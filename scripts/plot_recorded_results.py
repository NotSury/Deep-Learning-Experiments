"""Plot training curves from the *recorded* notebook outputs.

This script does not execute any notebook code or train anything. It reads the
saved stdout of the training cells in `notebooks/*.ipynb`, parses the per-epoch
metrics that were printed during the original Google Colab runs, and draws them.

Usage (from the repository root):
    python scripts/plot_recorded_results.py
Requires: matplotlib
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
OUT = ROOT / "figures"

# Palette (validated categorical order) and recessive ink colours
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2,
})

NUM = r"([\d.]+)"


def stream_of(notebook, marker):
    """Concatenated stdout of the unique code cell whose source contains `marker`."""
    nb = json.loads((NB / notebook).read_text(encoding="utf-8"))
    hits = [c for c in nb["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(hits) == 1, (notebook, marker, len(hits))
    return "".join("".join(o["text"]) for o in hits[0].get("outputs", []) if o["output_type"] == "stream")


def parse(text, pattern):
    rows = [tuple(map(float, m)) for m in re.findall(pattern, text)]
    assert rows, pattern
    return list(zip(*rows))  # columns


FFN_PAT = rf"Train Loss: {NUM}, Train Acc: {NUM}%, (?:Test|Val) Loss: {NUM}, (?:Test|Val) Acc: {NUM}%"
CNN_PAT = rf"train loss= {NUM} - train acc= {NUM}% - valid loss= {NUM} - valid acc= {NUM}%"


def end_label(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK2)
    ax.plot([x], [y], "o", ms=5, color=color, mec="white", mew=1.5, zorder=5)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------ MNIST optimisers
nb1 = "01_activations_and_manual_ffn_optimisers.ipynb"
runs = {
    "SGD": parse(stream_of(nb1, "myFFN.update_SGD(x, y"), FFN_PAT),
    "SGD + momentum": parse(stream_of(nb1, "myFFN_momentum.update_SGDwithMomentum"), FFN_PAT),
    "AdaGrad": parse(stream_of(nb1, "myFFN_adagrad.update_AdaGrad"), FFN_PAT),
}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for (name, (trl, tra, tel, tea)), col in zip(runs.items(), C):
    ep = range(1, len(tea) + 1)
    axes[0].plot(ep, tea, color=col, label=name)
    axes[1].plot(ep, tel, color=col, label=name)
axes[0].set(title="Test accuracy", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(89, 98.2))
axes[1].set(title="Test loss (cross-entropy)", xlabel="Epoch", ylabel="Loss")
axes[0].legend(loc="lower right")
fig.suptitle("MNIST, 784-100-100-10 ReLU MLP: hand-written optimisers (lr = 0.01, batch 64)",
             x=0.01, ha="left", color=INK, fontsize=12, fontweight="semibold")
fig.tight_layout()
save(fig, "mnist_manual_optimisers.png")

# ------------------------------------------------------------------ Fashion-MNIST SAM
nb2 = "02_ffn_grid_search_entropy_loss_and_sam.ipynb"
trl, tra, vl, va = parse(stream_of(nb2, "class SAM(torch.optim.Optimizer)"), FFN_PAT)
ep = range(1, len(va) + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(ep, tra, color=C[0], label="Train (running avg.)")
axes[0].plot(ep, va, color=C[1], label="Validation")
axes[1].plot(ep, trl, color=C[0], label="Train (running avg.)")
axes[1].plot(ep, vl, color=C[1], label="Validation")
axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
axes[1].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
axes[1].legend(loc="upper right")
fig.suptitle("Fashion-MNIST, 784-40-40-10 Tanh MLP trained with SAM (ρ = 0.05, SGD-momentum base)",
             x=0.01, ha="left", color=INK, fontsize=12, fontweight="semibold")
fig.tight_layout()
save(fig, "fashion_mnist_sam_curves.png")

# ------------------------------------------------------------------ CNN experiments
nb3 = "03_cnn_mixup_cutmix_ova_and_rotation.ipynb"
cnn = {
    "Baseline (CE)": parse(stream_of(nb3, "history = fit(model= my_cnn"), CNN_PAT),
    "Mixup": parse(stream_of(nb3, "mixup_history = fit_mixup("), CNN_PAT),
    "CutMix": parse(stream_of(nb3, "def rand_bbox"), CNN_PAT),
    "One-vs-All loss": parse(stream_of(nb3, "class OVALoss"), CNN_PAT),
    "CE + rotation (λ=0.1)": parse(stream_of(nb3, "class YourCNNRotation"), CNN_PAT),
}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
for (name, (trl, tra, vl, va)), col in zip(cnn.items(), C):
    ep = range(1, len(va) + 1)
    axes[0].plot(ep, va, color=col, label=name, marker="o", ms=4, mec="white", mew=1)
    if name != "One-vs-All loss":  # OVA's loss is a different objective (sum of 20 BCE terms)
        axes[1].plot(ep, vl, color=col, label=name, marker="o", ms=4, mec="white", mew=1)
axes[0].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)", xticks=range(1, 11))
axes[1].set(title="Validation cross-entropy (OVA omitted: different loss)", xlabel="Epoch",
            ylabel="Loss", xticks=range(1, 11))
axes[0].legend(loc="lower right", fontsize=9)
fig.suptitle("20-class image classification, residual CNN [16, 32, 64]: 10 epochs, single seed",
             x=0.01, ha="left", color=INK, fontsize=12, fontweight="semibold")
fig.tight_layout()
save(fig, "cnn_validation_curves.png")

# Best vs final-epoch validation accuracy (horizontal grouped bars)
names = list(cnn)
best = [max(cnn[n][3]) for n in names]
final = [cnn[n][3][-1] for n in names]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
h = 0.36
ys = range(len(names))
ax.barh([y - h / 2 - 0.01 for y in ys], best, height=h, color=C[0], label="Best epoch")
ax.barh([y + h / 2 + 0.01 for y in ys], final, height=h, color=C[1], label="Final epoch (10)")
for y, b, f in zip(ys, best, final):
    ax.text(b + 0.5, y - h / 2, f"{b:.2f}", va="center", fontsize=9, color=INK2)
    ax.text(f + 0.5, y + h / 2, f"{f:.2f}", va="center", fontsize=9, color=INK2)
ax.set_yticks(list(ys), names)
ax.invert_yaxis()
ax.set_xlim(0, 65)
ax.set_xlabel("Validation accuracy (%)")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
ax.set_title("Best vs final-epoch validation accuracy (947 validation images)", loc="left")
fig.tight_layout()
save(fig, "cnn_best_vs_final_accuracy.png")
