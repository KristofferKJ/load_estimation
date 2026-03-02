import csv
import math
import matplotlib.pyplot as plt

# ---- CONFIG ----
csv_path = "src/Test_8.csv"

# ---- LOAD CSV ----
gt_x, gt_y, gt_z = [], [], []
est_x, est_y, est_z = [], [], []

with open(csv_path, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        gx, gy, gz, ex, ey, ez = map(float, row)
        gt_x.append(gx)
        gt_y.append(gy)
        gt_z.append(gz)
        est_x.append(ex)
        est_y.append(ey)
        est_z.append(ez)

# ---- RMSE FUNCTION ----
def rmse(gt, est):
    n = len(gt)
    return math.sqrt(sum((g - e)**2 for g, e in zip(gt, est)) / n)

rmse_x = rmse(gt_x, est_x)
rmse_y = rmse(gt_y, est_y)
rmse_z = rmse(gt_z, est_z)

print(f"RMSE X: {rmse_x:.4f}")
print(f"RMSE Y: {rmse_y:.4f}")
print(f"RMSE Z: {rmse_z:.4f}")

# ---- SAMPLE INDICES ----
samples = list(range(1, len(gt_x)+1))

# ---- PLOTTING ----
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

components = [
    (gt_x, est_x, "X", rmse_x),
    (gt_y, est_y, "Y", rmse_y),
    (gt_z, est_z, "Z", rmse_z)
]

for ax, (gt, est, label, error) in zip(axes, components):
    ax.plot(samples, gt, label="Ground Truth", color="blue")
    ax.plot(samples, est, label="Estimated", color="orange", alpha=0.7)
    ax.set_ylabel(label)
    ax.set_title(f"{label} over samples (RMSE={error:.4f})")
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel("Sample Index")
plt.tight_layout()
plt.show()