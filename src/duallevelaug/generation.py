from pathlib import Path
from itertools import product
import math
from .utils import require,read_json,write_json,file_hash,snapshot
from .sampling import sample_counts,sample_edges
from .statistics import point_arrays
from .geometry import tensor_geometry,soft_relations,local_losses,rotation_matrices


def initialize_layout(objects,background,rng):
    np=require("numpy")
    bounds=np.asarray(background["scene_bounds"],float)
    poses=np.zeros((len(objects),5),dtype=np.float32)
    normals=np.tile([0.,0.,1.],(len(objects),1))
    supports=background.get("supports",[])
    for i,obj in enumerate(objects):
        centered=obj["coord"]-np.asarray(obj["centroid"])
        lo,hi=centered.min(0),centered.max(0)
        centered=np.array([[hi[d] if bit[d] else lo[d] for d in range(3)] for bit in product((0,1),repeat=3)])
        poses[i,3]=rng.uniform(-np.pi,np.pi)
        if supports:
            support=supports[int(rng.integers(len(supports)))]
            sb=np.asarray(support["bounds"])
            xy=rng.uniform(sb[0,:2],sb[1,:2])
            normal=np.asarray(support["normal"])
            height=-(float(support["offset"])+float(normal[:2]@xy))/max(float(normal[2]),1e-6)
            normals[i]=normal
        else:
            xy=rng.uniform(bounds[0,:2],bounds[1,:2])
            height=bounds[0,2]
        poses[i,:2]=xy
        poses[i,2]=height-centered[:,2].min()
        # Apply the same bounds to all variants, including zero-step random layouts.
        yaw=poses[i,3]
        rot=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        rotated=centered@rot.T
        if np.any(np.ptp(rotated,axis=0)>bounds[1]-bounds[0]+1e-5):
            poses[i,3]=0
            rotated=centered
        low=bounds[0]-rotated.min(0)
        high=bounds[1]-rotated.max(0)
        if np.any(low>high+1e-5):
            raise ValueError("Sampled object cannot fit within background bounds")
        poses[i,:3]=np.clip(poses[i,:3],low,high)
    return poses,normals


def optimize(objects,anchors,initial,support_normals,bounds,target,classes,cfg,encoder=None,cooccurrence=None):
    torch=require("torch")
    np=require("numpy")
    device=torch.device(cfg.get("device","cuda" if torch.cuda.is_available() else "cpu"))
    dynamic_count=len(objects)
    components=objects+anchors
    if not objects:
        return initial,[]
    local_bounds=[]
    fronts=[]
    normals=[]
    anchor_pose=[]
    for i,obj in enumerate(components):
        center=np.asarray(obj["centroid"])
        p=obj["coord"]-center
        local_bounds.append(np.stack((p.min(0),p.max(0))))
        fronts.append(obj.get("front",np.asarray(obj["rotation"])[:,0]))
        normals.append(obj.get("normal_axis",np.asarray(obj["rotation"])[:,2]))
        if i>=dynamic_count:
            anchor_pose.append(np.r_[center,0,0])
    tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
    local_bounds,fronts,normals=map(tensor,(local_bounds,fronts,normals))
    fixed=tensor(anchor_pose).reshape(-1,5)
    poses=torch.nn.Parameter(tensor(initial))
    support=tensor(support_normals)
    target=tensor(target)
    classes=torch.as_tensor(classes,dtype=torch.long,device=device)
    edge_weights=tensor(cooccurrence)[classes[:,None],classes[None,:]] if cooccurrence is not None else None
    scene_bounds=tensor(bounds)
    optimizer=torch.optim.Adam([poses],lr=cfg.get("pose_lr",.03))
    mode=cfg.get("constraints","full")
    if mode=="full" and encoder is None:
        raise ValueError("Full constraints require a pretrained encoder")
    weights={**{"geometric":1.,"semantic":1.,"topology":1.},**cfg.get("loss_weights",{})}
    history=[]
    for step in range(cfg.get("steps",300) if mode!="random" else 0):
        geom=tensor_geometry(local_bounds,torch.cat((poses,fixed)),fronts,normals)
        losses=local_losses(geom,target,dynamic_count,support,cfg.get("relation_thresholds"),cfg.get("collision_thickness",.02))
        loss=weights["geometric"]*losses["geometric"]
        if mode in ("local","full"):
            loss=loss+weights["semantic"]*losses["semantic"]
        if mode=="full":
            from .graph import graph_loss
            edges=soft_relations(geom,cfg.get("relation_thresholds"))
            losses["topology"]=graph_loss(encoder,classes,target,classes,edges,target_weights=edge_weights,current_weights=edge_weights)
            loss=loss+weights["topology"]*losses["topology"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite pose loss; no scene written")
        optimizer.zero_grad()
        loss.backward()
        previous_angles=poses[:,3:].detach().clone()
        optimizer.step()
        with torch.no_grad():
            poses[:,3].copy_(torch.remainder(poses[:,3]+math.pi,2*math.pi)-math.pi)
            poses[:,4].clamp_(-math.pi/2,math.pi/2)
            g=tensor_geometry(local_bounds[:dynamic_count],poses,fronts[:dynamic_count],normals[:dynamic_count])
            oversized=((g["hi"]-g["lo"])>scene_bounds[1]-scene_bounds[0]+1e-5).any(-1)
            poses[oversized,3:]=previous_angles[oversized]
            g=tensor_geometry(local_bounds[:dynamic_count],poses,fronts[:dynamic_count],normals[:dynamic_count])
            lower_shift=(scene_bounds[0]-g["lo"]).clamp_min(0)
            upper_shift=(g["hi"]-scene_bounds[1]).clamp_min(0)
            poses[:,:3].add_(lower_shift-upper_shift)
        if step==0 or (step+1)%10==0 or step+1==cfg.get("steps",300):
            history.append({"step":step+1,"total":float(loss.detach()),**{k:float(v.detach()) for k,v in losses.items()}})
    return poses.detach().cpu().numpy(),history


def transform_object(obj,pose,instance_id,component_id):
    np=require("numpy")
    yaw,pitch=pose[3:]
    cy,sy,cp,sp=np.cos(yaw),np.sin(yaw),np.cos(pitch),np.sin(pitch)
    r=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])@np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    arrays={k:v.copy() for k,v in point_arrays(obj).items()}
    arrays["coord"]=((obj["coord"]-np.asarray(obj["centroid"]))@r.T+pose[:3]).astype(np.float32)
    if "normal" in arrays:
        arrays["normal"]=arrays["normal"]@r.T
    arrays["instance"]=np.full(len(arrays["coord"]),instance_id,dtype=np.int64)
    arrays["source_component"]=np.full(len(arrays["coord"]),component_id,dtype=np.int64)
    arrays["generated_point"]=np.zeros(len(arrays["coord"]),dtype=np.uint8)
    return arrays


def concatenate(parts):
    np=require("numpy")
    keys=set.intersection(*(set(x) for x in parts))
    for key in ("color","strength","normal"):
        if any(key in p for p in parts) and not all(key in p for p in parts):
            raise ValueError("Inconsistent %s features among sampled components"%key)
    return {k:np.concatenate([p[k] for p in parts],axis=0) for k in keys}


def generate(cfg):
    np=require("numpy")
    from .data import load_scene,save_scene,load_manifest,split_hash,validate_manifest
    stats=read_json(cfg["stats"])
    library=read_json(stats["library"])
    if library["manifest_hash"]!=stats["manifest_hash"]:
        raise ValueError("Library and statistics split hashes disagree")
    if not library["backgrounds"]:
        raise ValueError("No background scenes in training library")
    original=load_manifest(stats["manifest"])
    from .data import canonical_dataset
    if canonical_dataset(cfg.get("dataset",stats["dataset"]))!=canonical_dataset(stats["dataset"]) or original["dataset"]!=stats["dataset"] or original["classes"]!=stats["classes"]:
        raise ValueError("Generation config, statistics and original dataset/classes differ")
    if file_hash(stats["manifest"])!=stats["manifest_hash"]:
        raise ValueError("Original manifest changed after statistics collection")
    if any(x.get("source_split")!="train" for x in library["objects"]+library["backgrounds"]):
        raise ValueError("Held-out component in generation library")
    train_ids={r["id"] for r in original["scenes"] if r["split"]=="train" and not r.get("generated",False)}
    if any(obj["source_scene"] not in train_ids for obj in library["objects"]) or any(bg["scene_id"] not in train_ids for bg in library["backgrounds"]):
        raise ValueError("Library component source is not an original training scene")
    output=Path(cfg["output"]).resolve()
    output.mkdir(parents=True,exist_ok=True)
    encoder=None
    if cfg.get("constraints","full")=="full":
        from .graph import load_encoder
        torch=require("torch")
        device=cfg.get("device","cuda" if torch.cuda.is_available() else "cpu")
        encoder=load_encoder(cfg.get("graph_checkpoint"),stats,device)
    by_class={i:[] for i in range(len(stats["classes"]))}
    for obj in library["objects"]:
        by_class[obj["class_id"]].append(obj)
    count=int(math.ceil(cfg.get("ratio",.25)*stats["n_scenes"]))
    records=[]
    for scene_index in range(count):
        # Per-scene seeds preserve ablation prefixes across ratios and constraints.
        rng=np.random.default_rng(np.random.SeedSequence([cfg.get("seed",0),scene_index]))
        for attempt in range(cfg.get("layout_attempts",64)):
            background=library["backgrounds"][int(rng.integers(len(library["backgrounds"])))]
            counts=sample_counts(stats,rng,cfg.get("candidates",64),cfg.get("gt_sampling",False))
            scene_span=np.diff(np.asarray(background["scene_bounds"]),axis=0)[0]
            fit_pools={}
            for cls,n in enumerate(counts):
                if n:
                    fit_pools[cls]=[m for m in by_class[cls] if np.all(np.asarray(m.get("world_extent",m["extent"]))<=scene_span+1e-5)]
            if all(fit_pools[cls] for cls,n in enumerate(counts) if n):
                break
        else:
            raise ValueError("No sampled class composition fits any candidate background after layout_attempts")
        objects=[]
        source_paths=[]
        for cls,n in enumerate(counts):
            if n and not by_class[cls]:
                raise ValueError("Sampled class has no training objects")
            for _ in range(n):
                meta=fit_pools[cls][int(rng.integers(len(fit_pools[cls])))]
                objects.append({**load_scene(meta["path"]),**meta})
                source_paths.append(meta["path"])
        anchors=[{**load_scene(m["path"]),**m} for m in background["components"]]
        classes=[o["class_id"] for o in objects+anchors]
        target=sample_edges(classes,stats,rng,cfg.get("edge_noise",.05),cfg.get("cooccurrence_threshold",0))
        # Anchor geometry and observed relations stay fixed.
        from .geometry import component_geometry,hard_relations
        for i in range(len(objects),len(classes)):
            for j in range(len(objects),len(classes)):
                if i!=j:
                    target[i,j]=hard_relations(component_geometry(anchors[i-len(objects)]),component_geometry(anchors[j-len(objects)]),cfg.get("relation_thresholds"))
        initial,support_normals=initialize_layout(objects,background,rng)
        poses,history=optimize(objects,anchors,initial,support_normals,background["scene_bounds"],target,classes,cfg,encoder,stats["normalized_cooccurrence"])
        fixed=load_scene(background["path"])
        if cfg.get("poisson",False):
            from .decomposition import complete_background
            options=cfg.get("poisson_options",{})
            fixed=complete_background(load_scene(background["raw_path"]),fixed,
                    depth=options.get("depth",9),distance_threshold=options.get("distance",.1),
                    normal_angle=options.get("normal_angle",20),density_trim=options.get("density_trim",.01),
                    noise_std=options.get("jitter",.002),seed=cfg.get("seed",0)+scene_index,
                    class_ids={"scannet":(0,1),"s3dis":(1,2),"semantickitti":(8,9,10,11,12),"stpls3d":(0,1)}[stats["dataset"]])
        fixed["source_component"]=np.full(len(fixed["coord"]),-1,dtype=np.int64)
        fixed.setdefault("generated_point",np.zeros(len(fixed["coord"]),dtype=np.uint8))
        parts=[fixed]+[transform_object(obj,pose,i,i) for i,(obj,pose) in enumerate(zip(objects,poses))]
        scene=concatenate(parts)
        name="generated_%06d"%scene_index
        path=output/(name+".npz")
        save_scene(path,scene)
        layout={"initial":initial.tolist(),"poses":poses.tolist(),"classes":classes,"target_edges":target.tolist(),
                "cooccurrence_weights":np.asarray(stats["normalized_cooccurrence"])[np.asarray(classes,dtype=int)[:,None],np.asarray(classes,dtype=int)[None,:]].tolist(),
                "object_paths":source_paths,"background_path":background["path"],"losses":history,
                "support_fallback":not bool(background.get("supports"))}
        write_json(output/(name+"_layout.json"),layout)
        records.append({"id":name,"path":str(path),"split":"train","generated":True,
                        "background_scene":background["scene_id"],"background_source_id":background["scene_id"],"background_area":background.get("area"),
                        "area":background.get("area"),"background_split":"train","source":source_paths,
                        "layout":str(output/(name+"_layout.json"))})
        print("generate %d/%d %s objects=%d"%(scene_index+1,count,name,len(objects)),flush=True)
    manifest={"schema_version":1,"dataset":stats["dataset"],"classes":stats["classes"],"scenes":records,
              "split_hash":split_hash(records),"original_manifest":stats["manifest"],"original_manifest_hash":stats["manifest_hash"],"original_split_hash":stats["split_hash"],
              "augmentation_count":count,"original_training_scenes":stats["n_scenes"],"seed":cfg.get("seed",0)}
    validate_manifest(manifest)
    write_json(output/"manifest.json",manifest)
    snapshot(output,cfg,manifest_hash=stats["manifest_hash"])
    return manifest
