"""Coordinates are metres; pose rotations use Rz @ Ry."""
from itertools import product
from .config import DEFAULTS
from .utils import require

RELATIONS = ("supported_by", "attached_to", "left_of", "right_of", "nearby", "faces", "oriented_with", "none")


def component_geometry(component):
    np = require("numpy")
    p = np.asarray(component["coord"])
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError("Geometry requires at least one 3D point")
    r = np.asarray(component.get("rotation", np.eye(3)))
    return {"lo": p.min(0), "hi": p.max(0), "front": np.asarray(component.get("front",r[:, 0])), "normal": np.asarray(component.get("normal_axis",r[:, 2]))}


def hard_relations(a, b, thresholds=None):
    np = require("numpy")
    t = dict(DEFAULTS["relation_thresholds"], **(thresholds or {}))
    lo_a, hi_a, lo_b, hi_b = [np.asarray(v) for v in (a["lo"], a["hi"], b["lo"], b["hi"])]
    eps = 1e-6
    size_a, size_b = np.maximum(hi_a-lo_a, eps), np.maximum(hi_b-lo_b, eps)
    ca, cb = (lo_a+hi_a)/2, (lo_b+hi_b)/2
    inter = np.maximum(np.minimum(hi_a, hi_b)-np.maximum(lo_a, lo_b), 0)
    overlap = np.prod(inter[:2]) / max(min(np.prod(size_a[:2]), np.prod(size_b[:2])), eps)
    volume_ratio = np.prod(inter)/max(min(np.prod(size_a), np.prod(size_b)), eps)
    align = abs(np.dot(a["front"], b["front"]))
    gap = abs(lo_a[2]-hi_b[2])
    left = np.clip((cb[0]-lo_a[0])/size_a[0], 0, 1)
    right = np.clip((hi_a[0]-cb[0])/size_a[0], 0, 1)
    dist = np.linalg.norm(cb-ca)
    face = np.dot(a["front"], cb-ca)/max(dist, eps)
    normal = np.dot(a["normal"], b["normal"])
    # Attachment uses OR (Table 1), even when only orientation matches.
    values = [overlap > t["overlap"] and gap <= t["support_gap"],
              volume_ratio > t["attach"] or align > t["direction"],
              left > t["left"], right > t["right"], dist <= t["near"],
              face > t["face"], overlap > t["overlap"] and normal > t["normal"]]
    return np.array(values + [not any(values)], dtype=np.float32)


def observed_graph(components, dynamic_count, neighbors=10, thresholds=None):
    """Directed foreground k-NN plus all static anchors."""
    np = require("numpy")
    n = len(components)
    edges = np.zeros((n, n, 8), dtype=np.float32)
    observed = np.zeros((n, n), dtype=bool)
    geometries = [component_geometry(c) for c in components]
    centers = np.array([(g["lo"]+g["hi"])/2 for g in geometries])
    for i in range(dynamic_count):
        order = sorted((j for j in range(dynamic_count) if j != i),
                       key=lambda j: (float(np.linalg.norm(centers[i]-centers[j])), j))[:neighbors]
        for j in order + list(range(dynamic_count, n)):
            edges[i, j] = hard_relations(geometries[i], geometries[j], thresholds)
            observed[i, j] = True
    return edges, observed


def rotation_matrices(poses):
    torch = require("torch")
    yaw, pitch = poses[:, 3], poses[:, 4]
    cy, sy, cp, sp = yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin()
    z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
    rz = torch.stack((cy,-sy,z,sy,cy,z,z,z,o), -1).reshape(-1,3,3)
    ry = torch.stack((cp,z,sp,z,o,z,-sp,z,cp), -1).reshape(-1,3,3)
    return rz @ ry


def tensor_geometry(local_bounds, poses, base_front, base_normal):
    torch = require("torch")
    corners = torch.tensor(list(product((0,1), repeat=3)), device=poses.device, dtype=torch.long)
    # local_bounds: [N,2,3]; rotate all corners before taking axis-aligned bounds.
    xyz = torch.stack([local_bounds[:, corners[:, d], d] for d in range(3)], dim=-1)
    r = rotation_matrices(poses)
    xyz = xyz @ r.transpose(1,2) + poses[:,None,:3]
    return {"lo": xyz.amin(1), "hi": xyz.amax(1),
            "front": (r @ base_front[:,:,None]).squeeze(-1),
            "normal": (r @ base_normal[:,:,None]).squeeze(-1)}


def pair_quantities(g):
    torch = require("torch")
    lo, hi = g["lo"], g["hi"]
    size = (hi-lo).clamp_min(1e-6)
    center = (hi+lo)/2
    inter = (torch.minimum(hi[:,None],hi[None,:])-torch.maximum(lo[:,None],lo[None,:])).clamp_min(0)
    area, vol = size[:,:2].prod(-1), size.prod(-1)
    overlap = inter[:,:,:2].prod(-1)/torch.minimum(area[:,None],area[None,:]).clamp_min(1e-6)
    volume = inter.prod(-1)
    delta = center[None,:,:]-center[:,None,:]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    face = (g["front"][:,None,:]*delta).sum(-1)/distance.clamp_min(1e-6)
    return {"overlap": overlap, "volume": volume,
            "volume_ratio": volume/torch.minimum(vol[:,None],vol[None,:]).clamp_min(1e-6),
            "gap": (lo[:,None,2]-hi[None,:,2]).abs(),
            "align": (g["front"] @ g["front"].T).abs().clamp(0,1),
            "left": ((center[None,:,0]-lo[:,None,0])/size[:,None,0]).clamp(0,1),
            "right": ((hi[:,None,0]-center[None,:,0])/size[:,None,0]).clamp(0,1),
            "distance": distance, "face": face.clamp(-1,1),
            "normal": (g["normal"] @ g["normal"].T).clamp(-1,1),
            "normalized_xy_distance": torch.linalg.vector_norm(delta[:,:,:2],dim=-1)/torch.minimum(area[:,None],area[None,:]).clamp_min(1e-6).sqrt()}


def soft_relations(g, thresholds=None):
    torch = require("torch")
    t = dict(DEFAULTS["relation_thresholds"], **(thresholds or {}))
    q = pair_quantities(g)
    sigmoid = lambda x: torch.sigmoid(x / t["temperature"])
    support = sigmoid(q["overlap"]-t["overlap"]) * sigmoid(t["support_gap"]-q["gap"])
    attach_a, attach_b = sigmoid(q["volume_ratio"]-t["attach"]), sigmoid(q["align"]-t["direction"])
    values = [support, 1-(1-attach_a)*(1-attach_b), sigmoid(q["left"]-t["left"]),
              sigmoid(q["right"]-t["right"]),sigmoid(t["near"]-q["distance"]),
              sigmoid(q["face"]-t["face"]),sigmoid(q["overlap"]-t["overlap"])*sigmoid(q["normal"]-t["normal"])]
    p = torch.stack(values, -1)
    p = torch.cat((p,(1-p).prod(-1,keepdim=True)), -1)
    return p * (1-torch.eye(len(p),device=p.device,dtype=p.dtype))[:,:,None]


def local_losses(g, target_edges, dynamic_count, support_normals, thresholds=None, collision_thickness=0.02):
    torch = require("torch")
    t = dict(DEFAULTS["relation_thresholds"], **(thresholds or {}))
    q = pair_quantities(g)
    n = len(g["lo"])
    i = torch.arange(n, device=g["lo"].device)
    dynamic_pair = ((i[:,None]<dynamic_count)|(i[None,:]<dynamic_count)) & (i[:,None]<i[None,:])
    collision_lo, collision_hi = g["lo"].clone(), g["hi"].clone()
    static_center=(collision_lo[dynamic_count:]+collision_hi[dynamic_count:])/2
    static_size=(collision_hi[dynamic_count:]-collision_lo[dynamic_count:]).clamp_min(collision_thickness)
    collision_lo[dynamic_count:]=static_center-static_size/2
    collision_hi[dynamic_count:]=static_center+static_size/2
    intersection=(torch.minimum(collision_hi[:,None],collision_hi[None,:])-torch.maximum(collision_lo[:,None],collision_lo[None,:])).clamp_min(0)
    collision = intersection.prod(-1)[dynamic_pair].sum()
    alignment = (1-(g["normal"][:dynamic_count]*support_normals).sum(-1).abs()).sum()
    rel = torch.stack((
        (q["normalized_xy_distance"]-t["overlap"]).clamp_min(0)+(q["gap"]-t["support_gap"]).abs(),
        (1-torch.maximum(q["volume_ratio"],q["align"])).square(),
        (1-q["left"]).square(),(1-q["right"]).square(),
        (q["distance"]-t["near"]).clamp_min(0),(1-q["face"]).square(),
        (t["overlap"]-q["overlap"]).clamp_min(0)+(t["normal"]-q["normal"]).clamp_min(0)), -1)
    semantic = (target_edges[:,:,:7]*rel).sum()
    return {"collision": collision,"alignment": alignment,"geometric": collision+alignment,"semantic": semantic}
