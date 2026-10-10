import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from src.methods.kmeans import KMeans
from src.utils import normalize_fn, accuracy_fn, macrof1_fn

np.random.seed(100)

os.makedirs("plots/kmeans", exist_ok=True)

# ── Load & normalize ──────────────────────────────────────────────────────────
data = np.load("data/features.npz", allow_pickle=True)
X_train, X_test = data["xtrain"], data["xtest"]
y_train, y_test = data["ytrainclassif"], data["ytestclassif"]

mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = normalize_fn(X_train, mean, std)
X_test  = normalize_fn(X_test,  mean, std)

# ── Run KMeans ────────────────────────────────────────────────────────────────
K_values = [3, 6, 9, 12, 15, 18, 21]
train_accs, test_accs, test_f1s, inertias = [], [], [], []

for K in K_values:
    print(f"  K={K}...")
    km = KMeans(K=K, max_iters=100)
    train_preds = km.fit(X_train, y_train)
    test_preds  = km.predict(X_test)

    train_accs.append(accuracy_fn(train_preds, y_train))
    test_accs.append(accuracy_fn(test_preds, y_test))
    test_f1s.append(macrof1_fn(test_preds, y_test))

    dists       = km.compute_distance(X_train, km.centers)
    assignments = km.find_closest_cluster(dists)
    inertia     = sum(
        np.sum((X_train[assignments == k] - km.centers[k]) ** 2)
        for k in range(K) if np.any(assignments == k)
    )
    inertias.append(inertia)

best_acc_idx = int(np.argmax(test_accs))
best_f1_idx  = int(np.argmax(test_f1s))

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Accuracy & F1 vs K
# ═════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(K_values, train_accs, "--o", color="steelblue", linewidth=1.5,
         markersize=6, label="Train accuracy")
ax1.plot(K_values, test_accs, "-o", color="steelblue", linewidth=2,
         markersize=8, label="Test accuracy")
ax1.scatter([K_values[best_acc_idx]], [test_accs[best_acc_idx]],
            s=150, color="steelblue", zorder=5)
ax1.annotate(f"K={K_values[best_acc_idx]}\n{test_accs[best_acc_idx]:.1f}%",
             xy=(K_values[best_acc_idx], test_accs[best_acc_idx]),
             xytext=(K_values[best_acc_idx] + 1, test_accs[best_acc_idx] - 3),
             fontsize=9, color="steelblue",
             arrowprops=dict(arrowstyle="->", color="steelblue"))

ax1.set_xlabel("Number of clusters K")
ax1.set_ylabel("Accuracy (%)", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax1.set_xticks(K_values)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(K_values, test_f1s, "-s", color="tomato", linewidth=2,
         markersize=8, label="Macro F1")
ax2.scatter([K_values[best_f1_idx]], [test_f1s[best_f1_idx]],
            s=150, color="tomato", zorder=5)
ax2.annotate(f"K={K_values[best_f1_idx]}\nF1={test_f1s[best_f1_idx]:.3f}",
             xy=(K_values[best_f1_idx], test_f1s[best_f1_idx]),
             xytext=(K_values[best_f1_idx] - 4, test_f1s[best_f1_idx] + 0.03),
             fontsize=9, color="tomato",
             arrowprops=dict(arrowstyle="->", color="tomato"))
ax2.set_ylabel("Macro F1", color="tomato")
ax2.tick_params(axis="y", labelcolor="tomato")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

plt.title("K-Means — Accuracy & Macro F1 vs K", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/kmeans/acc_f1_vs_k.png", dpi=150, bbox_inches="tight")
print("  Saved → plots/kmeans/acc_f1_vs_k.png")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Elbow curve
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(K_values, inertias, "-o", color="seagreen", linewidth=2, markersize=8)

drops = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
elbow_idx = int(np.argmax(drops)) + 1
ax.scatter([K_values[elbow_idx]], [inertias[elbow_idx]], s=200, color="seagreen", zorder=5)
ax.annotate(f"elbow  K={K_values[elbow_idx]}",
            xy=(K_values[elbow_idx], inertias[elbow_idx]),
            xytext=(K_values[elbow_idx] + 1.5, inertias[elbow_idx] + max(inertias)*0.04),
            fontsize=10, color="seagreen", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="seagreen"))
ax.axvline(K_values[elbow_idx], color="seagreen", linewidth=1, linestyle="--", alpha=0.5)

ax.set_xlabel("Number of clusters K")
ax.set_ylabel("Inertia (sum of squared distances)")
ax.set_xticks(K_values)
ax.grid(True, linestyle="--", alpha=0.4)
plt.title("K-Means — Elbow Curve", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/kmeans/elbow_curve.png", dpi=150, bbox_inches="tight")
print("  Saved → plots/kmeans/elbow_curve.png")
plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Summary bar chart
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, values, label, unit, highlight_idx, color, hi_color in zip(
    axes,
    [test_accs, test_f1s],
    ["Test Accuracy", "Macro F1"],
    ["%", ""],
    [best_acc_idx, best_f1_idx],
    ["#aec6e8", "#f4a9a8"],
    ["steelblue", "tomato"]
):
    colors = [hi_color if i == highlight_idx else color for i in range(len(K_values))]
    bars = ax.bar([str(k) for k in K_values], values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.1f}{unit}" if unit else f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("K")
    ax.set_ylabel(label + (f" ({unit})" if unit else ""))
    ax.set_title(label + " per K", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, max(values) * 1.15)

plt.suptitle("K-Means — Performance Summary", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/kmeans/summary_bars.png", dpi=150, bbox_inches="tight")
print("  Saved → plots/kmeans/summary_bars.png")
plt.close()

print(f"\nDone!")
print(f"  Best accuracy : K={K_values[best_acc_idx]} → {test_accs[best_acc_idx]:.2f}%")
print(f"  Best F1       : K={K_values[best_f1_idx]}  → {test_f1s[best_f1_idx]:.4f}")