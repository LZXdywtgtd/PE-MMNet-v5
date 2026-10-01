"""验证 crack 触发：低阈值 + fast_thermal"""
import time, sys
sys.path.insert(0, r'D:\New_team_project\projects\pe_mmnet\project_v5')

from data.patch_simulator_v5 import PatchSimulator

print('=== 裂纹触发验证 ===')
FAST = {'ramp_up_c_per_min': 1260.0, 'soak_temp_c': 1280.0, 'soak_duration_min': 1.0, 'cool_down_c_per_min': 1260.0}
for seed in [42, 7, 100, 2024]:
    sim = PatchSimulator(
        patch_size=256, thermal_profile=FAST,
        crack_stress_threshold_MPa=5.0, seed=seed,
    )
    t0 = time.time()
    out = sim.simulate()
    dt = time.time() - t0
    peak = out['metadata'].get('peak_stress_MPa', float('nan'))
    n_kpts = out['metadata'].get('n_keypoints', 0)
    bbox = out['crack_bbox']
    print(f"seed={seed:4d}  time={dt:6.2f}s  peak_stress={peak:7.2f}MPa  K={n_kpts:2d}  bbox={bbox.round(3)}")