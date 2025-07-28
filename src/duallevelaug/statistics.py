from pathlib import Path
from .utils import require, read_json, write_json, file_hash, snapshot
from .geometry import RELATIONS, observed_graph


def point_arrays(component):
    np=require("numpy")
    n=len(component["coord"])
    keys=("coord","color","strength","normal","segment","instance","raw_segment","origin","generated_point","source_component")
    return {k:np.asarray(component[k]) for k in keys if k in component and len(component[k])==n}


def component_meta(component):
    np=require("numpy")
    p=component["coord"]
    return {"class_id":int(component["class_id"]),"centroid":np.asarray(component.get("centroid",p.mean(0))).tolist(),
            "rotation":np.asarray(component.get("rotation",np.eye(3))).tolist(),
            "extent":np.asarray(component.get("extent",p.max(0)-p.min(0))).tolist(),
            "world_extent":(p.max(0)-p.min(0)).tolist(),
            "front":np.asarray(component.get("front", [1,0,0])).tolist(),
            "normal_axis":np.asarray(component.get("normal_axis",[0,0,1])).tolist()}


def collect(cfg):
    np=require("numpy")
    from .data import load_manifest,load_scene,save_scene,canonical_dataset,split_hash
    from .decomposition import decompose
    manifest=load_manifest(cfg["manifest"])
    if cfg.get("dataset") and canonical_dataset(cfg["dataset"])!=manifest["dataset"]:
        raise ValueError("Statistics config dataset differs from manifest")
    scenes=[s for s in manifest["scenes"] if s["split"]=="train"]
    if not scenes:
        raise ValueError("Statistics require at least one original training scene")
    if any(s.get("generated",False) for s in scenes):
        raise ValueError("Statistics must use original training scenes only")
    output=Path(cfg["output"]).resolve()
    (output/"objects").mkdir(parents=True,exist_ok=True)
    (output/"backgrounds").mkdir(exist_ok=True)
    (output/"graphs").mkdir(exist_ok=True)
    c=len(manifest["classes"])
    class_counts=[]
    cooccur=np.zeros((c,c),dtype=np.float64)
    observed_pair_counts=np.zeros((c,c),dtype=np.int64)
    relation_count=np.zeros((c,c,8),dtype=np.float64)
    prob_sum=np.zeros_like(relation_count)
    prob_sq=np.zeros_like(relation_count)
    graphs=[]
    library={"dataset":manifest["dataset"],"manifest_hash":file_hash(cfg["manifest"]),"split_hash":split_hash(manifest["scenes"]),"objects":[],"backgrounds":[]}
    totals=np.zeros(8,dtype=np.int64)
    decomposition={"floor":0,"background":0,"foreground":0}
    for si,record in enumerate(scenes):
        scene=load_scene(record["path"])
        parts=decompose(scene,manifest["dataset"],seed=cfg.get("seed",0),min_points=cfg.get("min_points",20),component_radius=cfg.get("component_radius",.35))
        objects=parts["objects"]
        anchors=parts["background_components"]
        components=objects+anchors
        counts=np.bincount(np.asarray([x["class_id"] for x in objects],dtype=np.int64),minlength=c)
        class_counts.append(counts)
        all_counts=np.bincount(np.asarray([x["class_id"] for x in components],dtype=np.int64),minlength=c)
        # Ordered instance pairs exclude self-pairs.
        cooccur+=np.outer(all_counts,all_counts)-np.diag(all_counts)
        edges,observed=observed_graph(components,len(objects),neighbors=cfg.get("nearest_neighbors",10),thresholds=cfg.get("relation_thresholds"))
        scene_rel=np.zeros_like(relation_count)
        for i,j in np.argwhere(observed):
            scene_rel[components[i]["class_id"],components[j]["class_id"]]+=edges[i,j]
            observed_pair_counts[components[i]["class_id"],components[j]["class_id"]]+=1
        relation_count+=scene_rel
        denominator=scene_rel.sum(-1,keepdims=True)
        probs=np.divide(scene_rel,denominator,out=np.zeros_like(scene_rel),where=denominator>0)
        prob_sum+=probs
        prob_sq+=probs**2
        totals+=edges.sum((0,1)).astype(int)
        decomposition["foreground"]+=len(objects)
        decomposition["floor"]+=sum(bool(a.get("is_floor", a.get("role")=="floor")) for a in anchors)
        decomposition["background"]+=sum(not bool(a.get("is_floor", a.get("role")=="floor")) for a in anchors)
        object_paths=[]
        for oi,obj in enumerate(objects):
            path=output/"objects"/("%06d_%05d.npz"%(si,oi))
            save_scene(path,point_arrays(obj))
            meta={**component_meta(obj),"path":str(path),"source_scene":record["id"],"source_split":"train","area":record.get("area")}
            library["objects"].append(meta)
            object_paths.append(str(path))
        bgpath=output/"backgrounds"/("%06d.npz"%si)
        save_scene(bgpath,point_arrays(parts["background"]))
        anchor_meta=[]
        for ai,a in enumerate(anchors):
            apath=output/"backgrounds"/("%06d_anchor%04d.npz"%(si,ai))
            save_scene(apath,point_arrays(a))
            anchor_meta.append({**component_meta(a),"path":str(apath),"is_floor":bool(a.get("is_floor", a.get("role")=="floor"))})
        def jsonable(v):
            if isinstance(v,dict): return {k:jsonable(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)): return [jsonable(x) for x in v]
            if isinstance(v,np.ndarray): return v.tolist()
            if isinstance(v,np.generic): return v.item()
            return v
        library["backgrounds"].append({"path":str(bgpath),"scene_id":record["id"],"area":record.get("area"),
                "source_split":"train","raw_path":record["path"],"components":anchor_meta,
                "supports":jsonable([{k:v for k,v in plane.items() if k not in ("indices","source_indices")} for plane in parts["supports"]]),"scene_bounds":jsonable(parts["scene_bounds"])})
        graph_path=output/"graphs"/("%06d.npz"%si)
        np.savez_compressed(graph_path,classes=np.array([x["class_id"] for x in components],dtype=np.int64),
                            edge_index=np.argwhere(observed),edge_values=edges[observed],num_nodes=len(components))
        graphs.append({"scene_id":record["id"],"path":str(graph_path),
                       "dynamic_count":len(objects),"object_paths":object_paths})
        print("stats %d/%d %s objects=%d"%(si+1,len(scenes),record["id"],len(objects)),flush=True)
    counts=np.asarray(class_counts)
    denom=relation_count.sum(-1,keepdims=True)
    probability=np.divide(relation_count,denom,out=np.zeros_like(relation_count),where=denom>0)
    probability[denom[:,:,0]==0,7]=1
    w=cooccur/max(cooccur.sum(),1)
    degree=w.sum(-1)
    scale=np.divide(1,np.sqrt(degree),out=np.zeros_like(degree),where=degree>0)
    normalized=scale[:,None]*w*scale[None,:]
    relationships={name:{"total":int(total),"mean":float(total/len(scenes))} for name,total in zip(RELATIONS,totals)}
    relationships["furniture_instances"]={"total":int(counts.sum()),"mean":float(counts.sum()/len(scenes))}
    result={"dataset":manifest["dataset"],"classes":manifest["classes"],"manifest":str(Path(cfg["manifest"]).resolve()),
            "manifest_hash":library["manifest_hash"],"split_hash":library["split_hash"],"n_scenes":len(scenes),"mean":counts.mean(0).tolist(),
            "variance":counts.var(0).tolist(),"count_max":counts.max(0).tolist(),
            "cooccurrence":w.tolist(),"normalized_cooccurrence":normalized.tolist(),
            "relation_probability":probability.tolist(),"observed_pair_counts":observed_pair_counts.tolist(),
            "relation_std":np.sqrt(np.maximum(prob_sq/len(scenes)-(prob_sum/len(scenes))**2,0)).tolist(),
            "decomposition":decomposition,"relationships":relationships,
            "library":str(output/"library.json"),"graphs":str(output/"graphs.json")}
    write_json(output/"library.json",library)
    write_json(output/"graphs.json",{"graphs":graphs,"relations":RELATIONS})
    write_json(output/"stats.json",result)
    snapshot(output,cfg)
    return result
