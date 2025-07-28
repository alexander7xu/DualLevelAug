from pathlib import Path
from .utils import require, read_json, snapshot, file_hash

torch = require("torch")
nn = torch.nn


class GraphEncoder(nn.Module):
    def __init__(self, num_classes, hidden=128, layers=3):
        super().__init__()
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes+1,hidden)  # last ID is MASK
        self.messages = nn.ModuleList([nn.Sequential(nn.Linear(hidden*2+8,hidden),nn.ReLU(),nn.Linear(hidden,hidden)) for _ in range(layers)])
        self.attention = nn.ModuleList([nn.Linear(hidden,1) for _ in range(layers)])
        self.updates = nn.ModuleList([nn.GRUCell(hidden,hidden) for _ in range(layers)])
        self.classifier = nn.Linear(hidden,num_classes)
        self.relation_head = nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Linear(hidden,8))

    def forward(self, classes, edges, weights=None):
        h = self.embedding(classes)
        n = len(h)
        if n == 0:
            return h
        for message,attention,update in zip(self.messages,self.attention,self.updates):
            # Row i receives context along directed edge (i,j).
            pair = torch.cat((h[:,None,:].expand(n,n,-1),h[None,:,:].expand(n,n,-1),edges),-1)
            m = message(pair)
            connectivity = 1-(1-edges[:,:,:7]).prod(-1)
            if weights is not None:
                connectivity = connectivity * weights
            connectivity = connectivity*(1-torch.eye(n,device=h.device))
            scores = attention(m).squeeze(-1)
            scores = torch.exp(scores-scores.max(dim=1,keepdim=True).values) * connectivity
            alpha = scores/(scores.sum(dim=1,keepdim=True)+1e-8)
            h = update((alpha[:,:,None]*m).sum(1),h)
        return h

    def predictions(self, classes, edges, weights=None):
        h = self(classes,edges,weights)
        n = len(h)
        pairs = torch.cat((h[:,None,:].expand(n,n,-1),h[None,:,:].expand(n,n,-1)),-1)
        return self.classifier(h), self.relation_head(pairs)


def graph_loss(encoder, target_classes, target_edges, current_classes, current_edges,
               insertion_cost=1.0, deletion_cost=1.0, substitution_weight=1.0, structure_weight=1.0,
               target_weights=None, current_weights=None):
    """Detach only discrete assignment, retaining gradients through selected costs."""
    from scipy.optimize import linear_sum_assignment
    target_h = encoder(target_classes,target_edges,target_weights)
    current_h = encoder(current_classes,current_edges,current_weights)
    nt,nc = len(target_h),len(current_h)
    if min(nt,nc)==0:
        return current_edges.sum()*0 + insertion_cost*max(nt-nc,0)+deletion_cost*max(nc-nt,0)
    costs = torch.cdist(target_h,current_h).square()
    rows,cols = linear_sum_assignment(costs.detach().cpu().numpy())
    rows = torch.as_tensor(rows,device=costs.device,dtype=torch.long)
    cols = torch.as_tensor(cols,device=costs.device,dtype=torch.long)
    # Edge probability excludes 'none'; permute both adjacency axes.
    ta = 1-(1-target_edges[:,:,:7]).prod(-1)
    ca = 1-(1-current_edges[:,:,:7]).prod(-1)
    if target_weights is not None: ta=ta*target_weights
    if current_weights is not None: ca=ca*current_weights
    ta,ca = ta[rows][:,rows],ca[cols][:,cols]
    return (substitution_weight*costs[rows,cols].sum()
            + structure_weight*torch.linalg.vector_norm(ta-ca)
            + insertion_cost*max(nt-nc,0)+deletion_cost*max(nc-nt,0))


def pretrain(cfg):
    stats = read_json(cfg["stats"])
    records = read_json(stats["graphs"])["graphs"]
    if not records:
        raise ValueError("Graph pretraining requires training graphs")
    from .config import DEFAULTS
    options = dict(DEFAULTS["graph"],**cfg.get("graph",{}))
    if options["epochs"] < 1:
        raise ValueError("At least one pretraining epoch is required")
    torch.manual_seed(cfg.get("seed",0))
    device = torch.device(cfg.get("device","cuda" if torch.cuda.is_available() else "cpu"))
    encoder = GraphEncoder(len(stats["classes"]),options["hidden"],options["layers"]).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(),lr=options["lr"])
    losses=[]
    usable_steps=0
    for epoch in range(options["epochs"]):
        total=0.0
        steps=0
        for index in torch.randperm(len(records)).tolist():
            record=records[index]
            if "path" in record:
                import numpy as np
                with np.load(record["path"],allow_pickle=False) as stored:
                    n=int(stored["num_nodes"])
                    edge_array=np.zeros((n,n,8),dtype=np.float32)
                    observed_array=np.zeros((n,n),dtype=bool)
                    idx=stored["edge_index"]
                    edge_array[idx[:,0],idx[:,1]]=stored["edge_values"]
                    observed_array[idx[:,0],idx[:,1]]=True
                    record={"classes":stored["classes"].tolist(),"edges":edge_array,"observed":observed_array}
            classes=torch.tensor(record["classes"],dtype=torch.long,device=device)
            edges=torch.tensor(record["edges"],dtype=torch.float32,device=device)
            n=len(classes)
            if n<2:
                continue
            class_mask=torch.rand(n,device=device)<options["mask_ratio"]
            if not class_mask.any():
                class_mask[torch.randint(n,(1,),device=device)]=True
            observed=torch.tensor(record.get("observed",(edges.sum(-1)>0).tolist()),device=device,dtype=torch.bool)
            edge_mask=(torch.rand(n,n,device=device)<options["mask_ratio"]) & observed
            if not edge_mask.any() and observed.any():
                pos=observed.nonzero()[0]
                edge_mask[pos[0],pos[1]]=True
            masked_classes=classes.clone()
            masked_classes[class_mask]=encoder.num_classes
            masked_edges=edges.clone()
            masked_edges[edge_mask]=0
            matrix=torch.tensor(stats["normalized_cooccurrence"],device=device,dtype=torch.float32)
            weights=matrix[classes[:,None],classes[None,:]].clone()
            # Replace class-specific weights to hide masked class identities.
            unknown=class_mask[:,None] | class_mask[None,:]
            weights[unknown]=matrix.mean()
            cls_logits,edge_logits=encoder.predictions(masked_classes,masked_edges,weights)
            loss=nn.functional.cross_entropy(cls_logits[class_mask],classes[class_mask])
            if edge_mask.any():
                loss=loss+nn.functional.binary_cross_entropy_with_logits(edge_logits[edge_mask],edges[edge_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total+=float(loss.detach())
            steps+=1
        if not steps:
            raise ValueError("No usable graphs with at least two nodes for pretraining")
        usable_steps+=steps
        losses.append(total/steps)
        print("graph epoch %d/%d loss=%.6f"%(epoch+1,options["epochs"],losses[-1]),flush=True)
    output=Path(cfg["output"])
    output.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":encoder.state_dict(),"num_classes":len(stats["classes"]),
                "classes":stats["classes"],"hidden":options["hidden"],"layers":options["layers"],
                "trained_epochs":options["epochs"],"optimizer_steps":usable_steps,
                "manifest_hash":stats["manifest_hash"],"losses":losses},output/"graph.pt")
    snapshot(output,cfg,manifest_hash=stats["manifest_hash"],checkpoint_sha256=file_hash(output/"graph.pt"))
    return str(output/"graph.pt")


def load_encoder(path, stats, device="cpu"):
    if not path or not Path(path).is_file():
        raise FileNotFoundError("Full constraints require a pretrained graph checkpoint. Run pretrain-graph or supply graph_checkpoint.")
    state=torch.load(path,map_location=device,weights_only=True)
    if state.get("trained_epochs",0)<1 or state.get("optimizer_steps",0)<1:
        raise ValueError("Graph checkpoint is missing valid pretraining metadata")
    if state.get("classes")!=stats["classes"]:
        raise ValueError("Graph checkpoint class order differs from dataset")
    if state.get("manifest_hash")!=stats["manifest_hash"]:
        raise ValueError("Graph checkpoint training split hash differs from statistics")
    encoder=GraphEncoder(state["num_classes"],state["hidden"],state["layers"]).to(device)
    encoder.load_state_dict(state["state_dict"],strict=True)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder
