import numpy as np
d = np.load("logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0.npz")
pm = d["pixel_mask"]; px = d["true_crack_pixels"]
pcas, ncomp, skel_len = [], [], []
from scipy import ndimage
from skimage import morphology
for i in range(200):
    p = px[i][pm[i]]
    if len(p) < 2: continue
    xy = np.stack([p[:,1]*128, p[:,0]*128], 1)
    img = np.zeros((128,128), bool)
    img[np.clip(xy[:,1].astype(int),0,127), np.clip(xy[:,0].astype(int),0,127)] = True
    ys, xs = np.where(img)
    pts = np.stack([xs, ys], 1).astype(float)
    cov = np.cov(pts.T); ev = np.linalg.eigvalsh(cov)
    pcas.append(np.sqrt(max(ev)/max(min(ev),1e-9)))
    lab, n = ndimage.label(img, structure=np.ones((3,3)))
    ncomp.append(n)
    sk = morphology.skeletonize(img)
    skel_len.append(sk.sum())
pcas=np.array(pcas); ncomp=np.array(ncomp); skel=np.array(skel_len)
print("PCA elong: p10=%.1f p50=%.1f p90=%.1f  >=3: %d  >=5: %d  <2: %d" % tuple(list(np.percentile(pcas,[10,50,90]))+[(pcas>=3).sum(),(pcas>=5).sum(),(pcas<2).sum()]))
print("components: p50=%d p90=%d max=%d  ==1: %d" % (np.median(ncomp), np.percentile(ncomp,90), ncomp.max(), (ncomp==1).sum()))
print("skeleton len: p50=%d p90=%d" % (np.median(skel), np.percentile(skel,90)))
i = 5
p = px[i][pm[i]]; img = np.zeros((128,128), bool)
img[np.clip((p[:,1]*128).astype(int),0,127), np.clip((p[:,0]*128).astype(int),0,127)] = True
rows = []
for r in range(0,128,4):
    rows.append("".join("#" if img[r,c] else "." for c in range(0,128,2)))
open("_gt_vis.txt","w").write("\n".join(rows))
print("sample 5 ASCII saved (rows y=0..124 step4, cols x=0..126 step2)")
