import sys; sys.path.insert(0,".")
import numpy as np
def shape_of(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) < 2: return "empty"
    h = ys.max()-ys.min()+1; w = xs.max()-xs.min()+1
    bf = ((ys==0)|(ys==mask.shape[0]-1)|(xs==0)|(xs==mask.shape[1]-1)).mean()
    return "npx=%d h=%d w=%d elong=%.1f border=%.2f" % (len(ys), h, w, max(h,w)/max(min(h,w),1), bf)
d = np.load("logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0.npz")
for i in [0, 5, 10]:
    p = d["true_crack_pixels"][i][d["pixel_mask"][i]]
    img = np.zeros((128,128), np.uint8); img[p[:,0].astype(int), p[:,1].astype(int)] = 255
    S = d["x_2d"][i][1]
    print("cached idx", i, "GT:", shape_of(img), "| S>0.9:", shape_of((S>0.9).astype(np.uint8)*255))
print()
from _dprime_generator import generate_dprime_sample
for i in [0, 5, 10]:
    s = generate_dprime_sample(i, seed_base=434242, threshold_MPa=60.0)
    cm = s["crack_mask"] if "crack_mask" in s else None
    print("fresh idx", i, "spikes:", s["spikes"][0][:3], "meta peak=%.0f" % s["metadata"]["peak_stress_MPa"])
    print("   GT kpts n=", s["true_keypoints"].shape[0], "px n=", s["true_crack_pixels"].shape[0])
