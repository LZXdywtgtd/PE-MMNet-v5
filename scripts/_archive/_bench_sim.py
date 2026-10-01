"""V5A6 仿真器速度基准测试"""
import time, sys
sys.path.insert(0, r'D:\New_team_project\projects\pe_mmnet\project_v5')

from data.patch_simulator_v5 import PatchSimulator

print('=== PatchSimulator 速度基准 ===')
for cfg in [
    {'name': '64 + fast_thermal', 'patch_size': 64, 'thermal_profile': {'ramp_up_c_per_min': 1260.0, 'soak_temp_c': 1280.0, 'soak_duration_min': 1.0, 'cool_down_c_per_min': 1260.0}},
    {'name': '128 + fast_thermal', 'patch_size': 128, 'thermal_profile': {'ramp_up_c_per_min': 1260.0, 'soak_temp_c': 1280.0, 'soak_duration_min': 1.0, 'cool_down_c_per_min': 1260.0}},
    {'name': '256 + fast_thermal', 'patch_size': 256, 'thermal_profile': {'ramp_up_c_per_min': 1260.0, 'soak_temp_c': 1280.0, 'soak_duration_min': 1.0, 'cool_down_c_per_min': 1260.0}},
]:
    sim = PatchSimulator(patch_size=cfg['patch_size'], thermal_profile=cfg['thermal_profile'], seed=42)
    t0 = time.time()
    out = sim.simulate()
    dt = time.time() - t0
    kps = out['crack_keypoints'].shape[0]
    bbox = out['crack_bbox']
    print(f"{cfg['name']:30s}: {dt:6.2f}s  keypoints={kps}  bbox={bbox.round(2)}")
print('=== 完成 ===')