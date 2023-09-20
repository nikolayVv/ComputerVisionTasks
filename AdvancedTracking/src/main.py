import matplotlib.pyplot as plt
import numpy as np
import math
from recursive_bayes import kalman_filter

# Spiral, Hypocycloid or Epicycloid
curve = "Spiral"
N = 40
a = 3
b = 2
v = np.linspace(5 * math.pi, 0, N)
# Number of points (time-steps) in the trajectory
if curve == 'Epicycloid':
    x = (a + b) * np.cos(v) - b * np.cos((a + b) / b * v)
    y = (a + b) * np.sin(v) - b * np.sin((a + b) / b * v)
elif curve == 'Hypocycloid':
    x = (a - b) * np.cos(v) + b * np.cos((a - b) / b * v)
    y = (a - b) * np.sin(v) - b * np.sin((a - b) / b * v)
else:
    x = np.cos(v) * v
    y = np.sin(v) * v

fig, ax = plt.subplots(5, 3)

# P=100, Q=1
# Calculate for each model
sx_RW, sy_RW = kalman_filter(x, y, 100, 1, 'RW')
sx_NCV, sy_NCV = kalman_filter(x, y, 100, 1, 'NCV')
sx_NCA, sy_NCA = kalman_filter(x, y, 100, 1, 'NCA')
# Draw RW plot
l1= ax[0][0].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
l2= ax[0][0].plot(sx_RW, sy_RW, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[0][0].set_title(f"RW: q=100, r=1")
ax[0][0].axes.get_yaxis().set_visible(False)
ax[0][0].axes.get_xaxis().set_visible(False)
# Draw NCV plot
ax[0][1].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[0][1].plot(sx_NCV, sy_NCV, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[0][1].set_title(f"NCV: q=100, r=1")
ax[0][1].axes.get_yaxis().set_visible(False)
ax[0][1].axes.get_xaxis().set_visible(False)
# Draw NCA plot
ax[0][2].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[0][2].plot(sx_NCA, sy_NCA, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[0][2].set_title(f"NCA: q=100, r=1")
ax[0][2].axes.get_yaxis().set_visible(False)
ax[0][2].axes.get_xaxis().set_visible(False)

# P=5, Q=1
# Calculate for each model
sx_RW, sy_RW = kalman_filter(x, y, 5, 1, 'RW')
sx_NCV, sy_NCV = kalman_filter(x, y, 5, 1, 'NCV')
sx_NCA, sy_NCA = kalman_filter(x, y, 5, 1, 'NCA')
# Draw RW plot
ax[1][0].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[1][0].plot(sx_RW, sy_RW, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[1][0].set_title(f"RW: q=5, r=1")
ax[1][0].axes.get_yaxis().set_visible(False)
ax[1][0].axes.get_xaxis().set_visible(False)
# Draw NCV plot
ax[1][1].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[1][1].plot(sx_NCV, sy_NCV, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[1][1].set_title(f"NCV: q=5, r=1")
ax[1][1].axes.get_yaxis().set_visible(False)
ax[1][1].axes.get_xaxis().set_visible(False)
# Draw NCA plot
ax[1][2].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[1][2].plot(sx_NCA, sy_NCA, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[1][2].set_title(f"NCA: q=5, r=1")
ax[1][2].axes.get_yaxis().set_visible(False)
ax[1][2].axes.get_xaxis().set_visible(False)

# P=1, Q=1
# Calculate for each model
sx_RW, sy_RW = kalman_filter(x, y, 1, 1, 'RW')
sx_NCV, sy_NCV = kalman_filter(x, y, 1, 1, 'NCV')
sx_NCA, sy_NCA = kalman_filter(x, y, 1, 1, 'NCA')
# Draw RW plot
ax[2][0].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[2][0].plot(sx_RW, sy_RW, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[2][0].set_title(f"RW: q=1, r=1")
ax[2][0].axes.get_yaxis().set_visible(False)
ax[2][0].axes.get_xaxis().set_visible(False)
# Draw NCV plot
ax[2][1].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[2][1].plot(sx_NCV, sy_NCV, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[2][1].set_title(f"NCV: q=1, r=1")
ax[2][1].axes.get_yaxis().set_visible(False)
ax[2][1].axes.get_xaxis().set_visible(False)
# Draw NCA plot
ax[2][2].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[2][2].plot(sx_NCA, sy_NCA, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[2][2].set_title(f"NCA: q=1, r=1")
ax[2][2].axes.get_yaxis().set_visible(False)
ax[2][2].axes.get_xaxis().set_visible(False)

# P=1, Q=5
# Calculate for each model
sx_RW, sy_RW = kalman_filter(x, y, 1, 5, 'RW')
sx_NCV, sy_NCV = kalman_filter(x, y, 1, 5, 'NCV')
sx_NCA, sy_NCA = kalman_filter(x, y, 1, 5, 'NCA')
# Draw RW plot
ax[3][0].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[3][0].plot(sx_RW, sy_RW, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[3][0].set_title(f"RW: q=1, r=5")
ax[3][0].axes.get_yaxis().set_visible(False)
ax[3][0].axes.get_xaxis().set_visible(False)
# Draw NCV plot
ax[3][1].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[3][1].plot(sx_NCV, sy_NCV, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[3][1].set_title(f"NCV: q=1, r=5")
ax[3][1].axes.get_yaxis().set_visible(False)
ax[3][1].axes.get_xaxis().set_visible(False)
# Draw NCA plot
ax[3][2].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[3][2].plot(sx_NCA, sy_NCA, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[3][2].set_title(f"NCA: q=1, r=5")
ax[3][2].axes.get_yaxis().set_visible(False)
ax[3][2].axes.get_xaxis().set_visible(False)

# P=1, Q=100
# Calculate for each model
sx_RW, sy_RW = kalman_filter(x, y, 1, 100, 'RW')
sx_NCV, sy_NCV = kalman_filter(x, y, 1, 100, 'NCV')
sx_NCA, sy_NCA = kalman_filter(x, y, 1, 100, 'NCA')
# Draw RW plot
ax[4][0].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[4][0].plot(sx_RW, sy_RW, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[4][0].set_title(f"RW: q=1, r=100")
ax[4][0].axes.get_yaxis().set_visible(False)
ax[4][0].axes.get_xaxis().set_visible(False)
# Draw NCV plot
ax[4][1].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[4][1].plot(sx_NCV, sy_NCV, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[4][1].set_title(f"NCV: q=1, r=100")
ax[4][1].axes.get_yaxis().set_visible(False)
ax[4][1].axes.get_xaxis().set_visible(False)
# Draw NCA plot
ax[4][2].plot(x, y, linestyle='-', marker='o', markersize=10, mfc='none', c='r')
ax[4][2].plot(sx_NCA, sy_NCA, linestyle='-', marker='o', markersize=10, mfc='none', c='b')
ax[4][2].set_title(f"NCA: q=1, r=100")
ax[4][2].axes.get_yaxis().set_visible(False)
ax[4][2].axes.get_xaxis().set_visible(False)
# Write the title
fig.text(0.12, 0.95, "Red curve:", ha="center", va="bottom", size="xx-large",color="red", fontweight='bold')
fig.text(0.35, 0.95, "measurements (observations)", ha="center", va="bottom", size="xx-large", fontweight='bold')
fig.text(0.62,0.95,"Blue curve: ", ha="center", va="bottom", size="xx-large",color="blue", fontweight='bold')
fig.text(0.81, 0.95, "filtered measurements", ha="center", va="bottom", size="xx-large", fontweight='bold')

# Show the figure

plt.show()