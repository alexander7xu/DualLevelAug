from .utils import require


def js_divergence(a, b):
    np = require("numpy")
    a, b = np.asarray(a,dtype=float), np.asarray(b,dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0 if a.sum() == b.sum() else float("inf")
    a,b = a/a.sum(), b/b.sum()
    m = (a+b)/2
    def kl(p):
        keep = p>0
        return (p[keep]*np.log(p[keep]/m[keep])).sum()
    return float((kl(a)+kl(b))/2)


def sampling_distribution(stats, gt_sampling=False):
    np = require("numpy")
    mean, std = np.array(stats["mean"],float), np.sqrt(stats["variance"])
    scale = np.ones(len(mean))
    if gt_sampling:
        for name in ("picture", "refrigerator", "otherfurniture", "sink", "counter"):
            normalized = [str(c).lower().replace(" ", "").replace("_", "") for c in stats["classes"]]
            if name in normalized:
                scale[normalized.index(name)] = 3
    # Scale the JS target too, so selection preserves GT sampling weights.
    return mean*scale, std*scale, np.ceil(np.array(stats["count_max"])*scale).astype(int)


def sample_counts(stats, rng, candidates=64, gt_sampling=False):
    np = require("numpy")
    mean,std,maximum = sampling_distribution(stats,gt_sampling)
    counts = np.clip(np.rint(rng.normal(mean,std,size=(candidates,len(mean)))),0,maximum).astype(int)
    if counts.sum() == 0 and mean.sum() > 0:
        counts[0, int(np.argmax(mean))] = 1
    scores = [js_divergence(c,mean) for c in counts]
    return counts[int(np.argmin(scores))]


def sample_edges(classes, stats, rng, noise=0.05, threshold=0.0):
    np = require("numpy")
    probabilities = np.asarray(stats["relation_probability"])
    std = np.asarray(stats["relation_std"])
    observed = np.asarray(stats["observed_pair_counts"])
    weights = np.asarray(stats["normalized_cooccurrence"])
    n = len(classes)
    edges = np.zeros((n,n,8), dtype=np.float32)
    for i,a in enumerate(classes):
        for j,b in enumerate(classes):
            if i==j:
                continue
            p = probabilities[a,b]
            perturbed = np.maximum(rng.normal(p,std[a,b]+noise),0)
            # Noise must not create relations for unobserved class pairs.
            if observed[a,b] == 0 or weights[a,b] <= threshold or p.sum() == 0:
                edges[i,j,7] = 1
                continue
            if perturbed.sum()==0:
                perturbed = p.copy()
            r = rng.choice(8,p=perturbed/perturbed.sum())
            edges[i,j,r] = 1
    return edges
