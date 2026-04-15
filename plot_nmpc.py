import csv
import matplotlib.pyplot as plt
import numpy as np

path = "/home/rasmus-storm/Desktop/tests/14_04_bias_estimation/video_20/dataset_nmpc"  # path to the CSV file containing the data

csv_path = f"{path}.csv"

# Load CSV file
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    data = [[float(val) for val in row] for row in reader]

# Transpose to get columns
columns = list(zip(*data))

# Split into two segments (first 12 and next 12 columns)
segment1 = columns[:12]
norm = np.linalg.norm(segment1[6:9], axis=0)
print(norm)
segment1[6:9] = np.divide(segment1[6:9], norm)
segment2 = columns[12:24]

estimated_bias = columns[24]

#error = np.subtract(np.array(segment2[5]), np.array(segment1[5]))
#mean_error = np.mean(error)

#segment2[5] = segment2[5] - mean_error

# Plot pairwise in increments of 3
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()

for i in range(0, 12, 3):
    for j in range(3):
        col_idx = i + j
        ax = axes[i + j]
        ax.plot(segment1[col_idx], label=f'Segment 1 - Col {col_idx}', marker='o')
        ax.plot(segment2[col_idx], label=f'Segment 2 - Col {col_idx}', marker='s')
        ax.set_title(f'Column {col_idx}')
        ax.legend()
        ax.grid(True)

plt.tight_layout()
plt.show()


fig2, ax2 = plt.subplots()
lines = ax2.plot(estimated_bias)

plt.show()
