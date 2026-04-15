import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Path
# -----------------------
path = "/home/rasmus-storm/Desktop/tests/14_04_bias_estimation/video_20/dataset_nmpc"
csv_path = f"{path}.csv"

# -----------------------
# USER नियंत्रLED WINDOW (EDIT THESE)
# -----------------------
start_idx = 4250      # <-- start sample
end_idx = 6000     # <-- end sample (None = use full length)

# -----------------------
# Load CSV
# -----------------------
data = np.loadtxt(csv_path, delimiter=',')
data = data.T  # columns

# -----------------------
# Split signals
# -----------------------
segment1 = data[:12]      # motion capture
segment2 = data[12:24]    # estimated
bias = np.squeeze(data[24])

# -----------------------
# Apply windowing
# -----------------------
if end_idx is None:
    end_idx = segment1.shape[1]

segment1 = segment1[:, start_idx:end_idx]
segment2 = segment2[:, start_idx:end_idx]
bias = bias[start_idx:end_idx]

# -----------------------
# Time base (50 Hz)
# -----------------------
fs = 50.0
dt = 1.0 / fs
N = segment1.shape[1]
t = np.arange(N) * dt

# -----------------------
# Normalize world vector (6:9)
# -----------------------
vec = segment1[6:9]
norm = np.linalg.norm(vec, axis=0)
norm[norm == 0] = 1e-8
segment1[6:9] = vec / norm

# -----------------------
# RMSE function
# -----------------------
def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.mean((a - b) ** 2))

# -----------------------
# Compute RMSE (NO bias)
# -----------------------
rmse_vals = np.array([
    rmse(segment1[i], segment2[i]) for i in range(12)
])

# -----------------------
# Plot layout
# -----------------------
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(5, 3)

def plot_pair(ax, i, title, ylabel):
    ax.plot(t, segment1[i], label="Motion Capture", linewidth=1)
    ax.plot(t, segment2[i], label="Estimated", linewidth=1)

    ax.set_title(f"{title} | RMSE: {rmse_vals[i]:.4f}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

# -----------------------
# Position (m)
# -----------------------
for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    plot_pair(ax, i, f"Position axis {i}", "Position [m]")

# -----------------------
# Velocity (m/s)
# -----------------------
for i in range(3, 6):
    ax = fig.add_subplot(gs[1, i - 3])
    plot_pair(ax, i, f"Velocity axis {i-3}", "Velocity [m/s]")

# -----------------------
# World vector
# -----------------------
for i in range(6, 9):
    ax = fig.add_subplot(gs[2, i - 6])
    plot_pair(ax, i, f"World Vector axis {i-6}", "Normalized direction")

# -----------------------
# Angular velocity (rad/s)
# -----------------------
for i in range(9, 12):
    ax = fig.add_subplot(gs[3, i - 9])
    plot_pair(ax, i, f"Angular Velocity axis {i-9}", "Angular velocity [rad/s]")

# -----------------------
# Bias (no RMSE)
# -----------------------
ax_bias = fig.add_subplot(gs[4, :])
ax_bias.plot(t, bias, label="Acceleration Bias", linewidth=2)

ax_bias.set_title("Estimated Acceleration Bias (m/s²)")
ax_bias.set_xlabel("Time [s]")
ax_bias.set_ylabel("Acceleration [m/s²]")
ax_bias.grid(True)
ax_bias.legend()

# -----------------------
# Final layout
# -----------------------
plt.tight_layout()
plt.show()