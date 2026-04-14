import csv
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ---- CONFIG ----
path = "/home/rasmus-storm/Desktop/tests/08_04_combined/dataset"  # path to the CSV file containing the data
csv_path = f"{path}.csv"

# ---- LOAD CSV ----
p_L_W_est = []
q_L_W_est = []
p_L_W_mocap = []
q_L_W_mocap = []

with open(csv_path, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        p_L_W_est.append([float(row[0]), float(row[1]), float(row[2])])
        q_L_W_est.append([float(row[3]), float(row[4]), float(row[5]), float(row[6])])

        p_L_W_mocap.append([float(row[7]), float(row[8]), float(row[9])])
        q_L_W_mocap.append([float(row[10]), float(row[11]), float(row[12]), float(row[13])])


# ---- RMSE FUNCTION ----
def rmse(gt, est):
    n = len(gt)
    return math.sqrt(sum((g - e) ** 2 for g, e in zip(gt, est)) / n)

# ---- TRANSLATION RMSE ----
rmse_x = rmse([p[0] for p in p_L_W_mocap], [p[0] for p in p_L_W_est])
rmse_y = rmse([p[1] for p in p_L_W_mocap], [p[1] for p in p_L_W_est])
rmse_z = rmse([p[2] for p in p_L_W_mocap], [p[2] for p in p_L_W_est])

print(f"RMSE X: {rmse_x:.4f}")
print(f"RMSE Y: {rmse_y:.4f}")
print(f"RMSE Z: {rmse_z:.4f}")

# ---- QUATERNION ROTATION ERROR ----
R_L_W_gt = [R.from_quat(q) for q in q_L_W_mocap]
R_L_W_est = [R.from_quat(q) for q in q_L_W_est]

quat_err = []
angle_err_deg = []

for R_gt, R_est in zip(R_L_W_gt, R_L_W_est):

    # relative rotation
    R_err = R_gt.inv() * R_est
    q_err = R_err.as_quat()  # [x, y, z, w]

    quat_err.append(q_err)

    # geodesic rotation angle
    w = abs(q_err[3])
    w = max(-1.0, min(1.0, w))  # numerical safety
    angle = 2.0 * math.acos(w)
    angle_err_deg.append(math.degrees(angle))

# split quaternion vector part
qx_err = [q[0] for q in quat_err]
qy_err = [q[1] for q in quat_err]
qz_err = [q[2] for q in quat_err]

# magnitude of vector part
vec_mag = [
    math.sqrt(x*x + y*y + z*z)
    for x, y, z in zip(qx_err, qy_err, qz_err)
]

# rotation RMSE (angle)
rmse_rot_angle = rmse([0]*len(angle_err_deg), angle_err_deg)

print(f"Rotation RMSE (angle): {rmse_rot_angle:.4f} deg")

# ---- SAMPLE INDICES ----
samples = list(range(1, len(p_L_W_mocap) + 1))

# ---- PLOTTING ----
fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)

translation_components = [
    ([p[0] for p in p_L_W_mocap], [p[0] for p in p_L_W_est], "X", rmse_x),
    ([p[1] for p in p_L_W_mocap], [p[1] for p in p_L_W_est], "Y", rmse_y),
    ([p[2] for p in p_L_W_mocap], [p[2] for p in p_L_W_est], "Z", rmse_z),
]

# --- Translation plots ---
for ax, (gt, est, label, error) in zip(axes[:3], translation_components):
    ax.plot(samples, gt, label="Ground Truth")
    ax.plot(samples, est, label="Estimated", alpha=0.7)
    ax.set_ylabel(label)
    ax.set_title(f"{label} Translation (RMSE={error:.4f})")
    ax.legend()
    ax.grid(True)

# --- Quaternion error angle ---
axes[3].plot(samples, angle_err_deg)
axes[3].set_ylabel("Angle (deg)")
axes[3].set_title(
    f"Quaternion Rotation Error Angle (RMSE={rmse_rot_angle:.4f} deg)"
)
axes[3].grid(True)

# --- Quaternion vector components ---
axes[4].plot(samples, qx_err, label="qx")
axes[4].plot(samples, qy_err, label="qy")
axes[4].plot(samples, qz_err, label="qz")
axes[4].set_ylabel("Vector part")
axes[4].set_title("Quaternion Error Vector Components")
axes[4].legend()
axes[4].grid(True)

# --- Quaternion vector magnitude ---
axes[5].plot(samples, vec_mag)
axes[5].set_ylabel("|q_vec|")
axes[5].set_title("Quaternion Error Vector Magnitude")
axes[5].grid(True)

axes[-1].set_xlabel("Sample Index")

plt.tight_layout()
plt.show()

# save the figure
fig.savefig(f"{path}_error_plots.png")