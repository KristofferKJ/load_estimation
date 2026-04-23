import csv
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np

# ---- CONFIG ----
path = "/home/rasmus-storm/Desktop/tests/20_04/rosbag_2/dataset"
csv_path = f"{path}.csv"

fs = 30.0  # Hz
dt = 1.0 / fs

# ---- LOAD CSV ----
p_D_W_mocap = []
q_D_W_mocap = []
p_L_C = []
q_L_C = []
p_L_W_mocap = []
q_L_W_mocap = []
R_G_G0 = []

with open(csv_path, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        p_D_W_mocap.append([float(row[0]), float(row[1]), float(row[2])])
        q_D_W_mocap.append([float(row[3]), float(row[4]), float(row[5]), float(row[6])])

        p_L_C.append([float(row[7]), float(row[8]), float(row[9])])
        q_L_C.append([float(row[10]), float(row[11]), float(row[12]), float(row[13])])

        p_L_W_mocap.append([float(row[14]), float(row[15]), float(row[16])])
        q_L_W_mocap.append([float(row[17]), float(row[18]), float(row[19]), float(row[20])])

        R_G_G0.append([float(row[21]), float(row[22]), float(row[23])])

# ---- SKIP FRAMES ----
skip_frames = 75
cut_frames = 1#1500
Allign_frames = 0#1500


p_D_W_mocap = p_D_W_mocap[skip_frames:-cut_frames-Allign_frames]
q_D_W_mocap = q_D_W_mocap[skip_frames:-cut_frames-Allign_frames]
p_L_C = p_L_C[skip_frames+Allign_frames:-cut_frames]
q_L_C = q_L_C[skip_frames+Allign_frames:-cut_frames]
p_L_W_mocap = p_L_W_mocap[skip_frames+Allign_frames:-cut_frames]
q_L_W_mocap = q_L_W_mocap[skip_frames+Allign_frames:-cut_frames]
R_G_G0 = R_G_G0[skip_frames+Allign_frames:-cut_frames]

# ---- ESTIMATION ----
p_L_W_est = []
q_L_W_est = []

for p_d_w, q_d_w, p_l_c, q_l_c, r_g_g0 in zip(
    p_D_W_mocap, q_D_W_mocap, p_L_C, q_L_C, R_G_G0
):
    p_d_w = np.array(p_d_w)
    r_d_w = R.from_quat(q_d_w)

    p_G0_D = np.array([0, 0, -0.107])
    q_G0_D = R.from_euler('y', 0, degrees=True).as_quat()

    p_G0_W = p_d_w + r_d_w.apply(p_G0_D)
    q_G0_W = r_d_w * R.from_quat(q_G0_D)

    q_G_G0 = R.from_euler('XY', [r_g_g0[0] + 2.17, r_g_g0[1] - 1.33], degrees=True).as_quat() #2.17 # 1.33

    p_G_W = p_G0_W
    q_G_W = q_G0_W * R.from_quat(q_G_G0)

    p_C_G = np.array([0.025, 0, 0])
    q_C_G = R.from_euler('xz', [-90, -90], degrees=True).as_quat()

    p_C_W = p_G_W + q_G_W.apply(p_C_G)
    q_C_W = q_G_W * R.from_quat(q_C_G)

    p_l_c = np.array(p_l_c)
    q_l_c = np.array(q_l_c)

    p_L_W = p_C_W + q_C_W.apply(p_l_c)
    q_L_W = q_C_W * R.from_quat(q_l_c)

    p_L_W_est.append(p_L_W)
    q_L_W_est.append(q_L_W.as_quat())

# ---- RMSE FUNCTION ----
def rmse(gt, est):
    return math.sqrt(sum((g - e) ** 2 for g, e in zip(gt, est)) / len(gt))

# ---- TRANSLATION RMSE ----
rmse_x = rmse([p[0] for p in p_L_W_mocap], [p[0] for p in p_L_W_est])
rmse_y = rmse([p[1] for p in p_L_W_mocap], [p[1] for p in p_L_W_est])
rmse_z = rmse([p[2] for p in p_L_W_mocap], [p[2] for p in p_L_W_est])

# ---- ROTATION ERROR (ANGLE) ----
R_gt = [R.from_quat(q) for q in q_L_W_mocap]
R_est = [R.from_quat(q) for q in q_L_W_est]

angle_err_deg = []

for r_gt, r_est in zip(R_gt, R_est):
    r_err = r_gt.inv() * r_est
    q_err = r_err.as_quat()

    w = np.clip(abs(q_err[3]), -1.0, 1.0)
    angle_err_deg.append(math.degrees(2 * math.acos(w)))

rmse_rot_angle = rmse([0] * len(angle_err_deg), angle_err_deg)

# ---- ALIGN QUATERNIONS ----
q_gt = np.array(q_L_W_mocap)
q_est = np.array(q_L_W_est)

for i in range(len(q_gt)):
    if np.dot(q_gt[i], q_est[i]) < 0:
        q_est[i] = -q_est[i]

qx_gt, qy_gt, qz_gt, qw_gt = q_gt.T
qx_est, qy_est, qz_est, qw_est = q_est.T

# ---- QUATERNION RMSE ----
rmse_qx = rmse(qx_gt, qx_est)
rmse_qy = rmse(qy_gt, qy_est)
rmse_qz = rmse(qz_gt, qz_est)
rmse_qw = rmse(qw_gt, qw_est)

print(f"RMSE X: {rmse_x:.4f}")
print(f"RMSE Y: {rmse_y:.4f}")
print(f"RMSE Z: {rmse_z:.4f}")
print(f"Rotation RMSE (angle): {rmse_rot_angle:.4f} deg")

print(f"Quaternion RMSE qx: {rmse_qx:.6f}")
print(f"Quaternion RMSE qy: {rmse_qy:.6f}")
print(f"Quaternion RMSE qz: {rmse_qz:.6f}")
print(f"Quaternion RMSE qw: {rmse_qw:.6f}")

# ---- TIME AXIS (SECONDS) ----
n = len(p_L_W_mocap)
time = np.arange(n) * dt

# ---- PLOTTING (4x2 GRID) ----
fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)

# LEFT: translation + angle error
left_plots = [
    ([p[0] for p in p_L_W_mocap], [p[0] for p in p_L_W_est], "X (m)", rmse_x),
    ([p[1] for p in p_L_W_mocap], [p[1] for p in p_L_W_est], "Y (m)", rmse_y),
    ([p[2] for p in p_L_W_mocap], [p[2] for p in p_L_W_est], "Z (m)", rmse_z),
]

for i, (gt, est, label, error) in enumerate(left_plots):
    ax = axes[i, 0]
    ax.plot(time, gt, label="GT")
    ax.plot(time, est, label="Est", alpha=0.7)
    ax.set_title(f"{label} (RMSE={error:.4f})")
    ax.set_ylabel(label)
    ax.grid(True)
    ax.legend()

axes[3, 0].plot(time, angle_err_deg)
axes[3, 0].set_title(f"Rotation Error Angle (RMSE={rmse_rot_angle:.4f} deg)")
axes[3, 0].set_ylabel("deg")
axes[3, 0].grid(True)

# RIGHT: quaternion components
quat_data = [
    (qx_gt, qx_est, "qx", rmse_qx),
    (qy_gt, qy_est, "qy", rmse_qy),
    (qz_gt, qz_est, "qz", rmse_qz),
    (qw_gt, qw_est, "qw", rmse_qw),
]

for i, (gt, est, label, error) in enumerate(quat_data):
    ax = axes[i, 1]
    ax.plot(time, gt, label="GT")
    ax.plot(time, est, label="Est", alpha=0.7)
    ax.set_title(f"{label} (RMSE={error:.5f})")
    ax.grid(True)
    ax.legend()

# ---- FINAL TOUCH ----
for ax in axes[-1]:
    ax.set_xlabel("Time (s)")

plt.tight_layout()
plt.show()

fig.savefig(f"{path}.png")