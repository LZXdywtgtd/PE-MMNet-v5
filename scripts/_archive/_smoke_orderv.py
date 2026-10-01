# -*- coding: utf-8 -*-
import sys; sys.path.insert(0, ".")
import torch
from training.ordered_kp_loss import resample_kpts_arclength, ordered_kpt_loss, validity_loss, OrderedKeypointLoss

# 1. straight line resample
gt = torch.zeros(1, 9, 2); gt[0, :, 1] = torch.linspace(0, 1, 9)
mask = torch.ones(1, 9, dtype=torch.bool)
out = resample_kpts_arclength(gt, mask, 16)
assert torch.allclose(out[0, :, 0], torch.zeros(16), atol=1e-6)
assert torch.allclose(out[0, :, 1], torch.linspace(0, 1, 16), atol=1e-5)
print("1. straight resample PASS")

# 2. polyline resample
gt2 = torch.zeros(1, 3, 2)
gt2[0, 0] = torch.tensor([0.0, 0.0]); gt2[0, 1] = torch.tensor([0.0, 0.5]); gt2[0, 2] = torch.tensor([0.5, 0.5])
out2 = resample_kpts_arclength(gt2, torch.ones(1, 3, dtype=torch.bool), 5)
exp = torch.tensor([[0, 0], [0, 0.25], [0, 0.5], [0.25, 0.5], [0.5, 0.5]])
assert torch.allclose(out2[0], exp, atol=1e-5), "got %s" % out2[0]
print("2. polyline resample PASS")

# 3. K=8 to 16
gt3 = torch.zeros(1, 8, 2); gt3[0, :, 1] = torch.linspace(0, 1, 8)
out3 = resample_kpts_arclength(gt3, torch.ones(1, 8, dtype=torch.bool), 16)
assert torch.allclose(out3[0, :, 1], torch.linspace(0, 1, 16), atol=1e-5)
print("3. K=8 resample PASS")

# 4. ordered loss fwd/bwd with mixed pos/neg batch
pk = torch.rand(2, 16, 2)
tk = torch.zeros(2, 16, 2); tk[0, :8] = gt3[0]; tk[1] = -1.0
km = torch.zeros(2, 16, dtype=torch.bool); km[0, :8] = True
l = ordered_kpt_loss(pk, tk, km)
assert l.item() > 0
l.backward()
assert pk.grad is not None and pk.grad.abs().sum() > 0
print("4. ordered loss PASS (l=%.4f)" % l.item())

# 5. validity loss
v = torch.rand(2, 16)
km2 = torch.ones(2, 16, dtype=torch.bool); km2[1] = False
lv = validity_loss(v, km2)
assert lv.item() > 0
print("5. validity loss PASS (l=%.4f)" % lv.item())

# 6. default weights 0 do not trigger
fn = OrderedKeypointLoss()
pb = torch.rand(2, 4); pkk = torch.rand(2, 16, 2); tb = torch.rand(2, 4); tcp = torch.rand(2, 50, 2)
ls = fn(pb, pkk, tb, tcp, pixel_mask=torch.ones(2, 50, dtype=torch.bool))
assert ls["ordered"].item() == 0.0 and ls["validity"].item() == 0.0
print("6. default-off PASS")
print("ALL PASS")
