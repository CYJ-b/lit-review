"""Make the two supervisor figures from analysis.py outputs."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

UM = 0.377442
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BIN_COL = {"Contact (≤20 µm)": "#184f95", "Near (20–100 µm)": "#3987e5", "Distant (>100 µm)": "#9ec5f4"}
GRP_COL = {"CLR (better survival)": "#2a78d6", "DII (worse survival)": "#eb6834"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": BASE, "figure.facecolor": SURF, "axes.facecolor": SURF})

path = sys.argv[1] if len(sys.argv) > 1 else "CRC_clusters_neighborhoods_markers.csv"
df = (pd.read_pickle(path) if path.endswith(".pkl") else pd.read_csv(path)).rename(columns={"X:X": "x", "Y:Y": "y"})
cd8 = pd.read_csv("cd8_distances.csv"); pat = pd.read_csv("per_patient_summary.csv")

# ---- Figure 1: cell map of one region
counts = cd8.groupby("File Name").size()
tum = df[df.ClusterName == "tumor cells"].groupby("File Name").size()
score = (counts.clip(upper=400) * tum.reindex(counts.index).clip(upper=800)).sort_values()
spot = score.index[-1]
s = df[df["File Name"] == spot]; c = cd8[cd8["File Name"] == spot]
fig, ax = plt.subplots(figsize=(9, 7))
other = s[~s.ClusterName.isin(["tumor cells", "CD8+ T cells"])]
ax.scatter(other.x * UM, other.y * UM, s=5, c="#e1e0d9", lw=0, label="Other cells")
t = s[s.ClusterName == "tumor cells"]
ax.scatter(t.x * UM, t.y * UM, s=9, c="#898781", lw=0, label="Tumour cells")
for b, col in BIN_COL.items():
    cc = c[c.bin == b]
    ax.scatter(cc.x * UM, cc.y * UM, s=34, c=col, edgecolors=SURF, linewidths=1, label=f"CD8+ T cell, {b}: n={len(cc)}")
ax.set_aspect("equal"); ax.invert_yaxis()
ax.set_xlabel("µm"); ax.set_ylabel("µm")
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
grp = "CLR (better survival)" if s.groups.iloc[0] == 1 else "DII (worse survival)"
ax.set_title(f"One invasive-front region ({spot}, patient {s.patients.iloc[0]}, {grp.split(" (")[0]} group)\nCD8+ T cells coloured by distance to the nearest tumour cell",
             loc="left", fontsize=12, color=INK)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, markerscale=1.3)
fig.savefig("fig1_cell_map.png", dpi=200, bbox_inches="tight", pad_inches=0.2); plt.close(fig)

# ---- Figure 2: three panels
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
# A per-patient % contact
ax = axs[0]
for i, (g, col) in enumerate(GRP_COL.items()):
    v = pat[pat.group == g].pct_contact.values
    jitter = np.random.default_rng(1).uniform(-0.12, 0.12, len(v))
    ax.scatter(np.full(len(v), i) + jitter, v, s=46, c=col, edgecolors=SURF, linewidths=1.5, zorder=3)
    ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, c=INK, lw=2, zorder=4)
    ax.text(i + 0.28, np.median(v), f"median {np.median(v):.0f}%", va="center", fontsize=9, color=INK2)
ax.set_xticks([0, 1], [g.replace(" (", "\n(") for g in GRP_COL]); ax.set_xlim(-0.5, 1.9)
ax.set_ylabel("% of CD8+ T cells in contact (≤20 µm)")
ax.set_title("A. Contact per patient (each dot = 1 patient)", loc="left", fontsize=11)
# B distance distribution
ax = axs[1]; bins = np.arange(0, 205, 5)
for g, col in GRP_COL.items():
    d = cd8[cd8.group == g].dist_um.clip(upper=200)
    ax.hist(d, bins=bins, density=True, histtype="step", lw=2, color=col, label=g)
for x in (20, 100): ax.axvline(x, c=MUTED, lw=1, ls="--"); ax.text(x + 2, ax.get_ylim()[1] * 0.95, f"{x} µm", color=MUTED, fontsize=8)
ax.set_xlabel("Distance to nearest tumour cell (µm, capped at 200)"); ax.set_ylabel("Share of CD8+ T cells (density)")
ax.set_title("B. Distance distribution by patient group", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=9)
# C PD-1 / Ki67 by bin
ax = axs[2]
m = cd8.groupby("bin")[["CD8+PD-1+", "CD8+Ki67+"]].mean().reindex(list(BIN_COL)) * 100
x = np.arange(3); w = 0.36
for j, (colname, lab) in enumerate([("CD8+PD-1+", "PD-1+ (exhaustion marker)"), ("CD8+Ki67+", "Ki-67+ (proliferating)")]):
    vals = m[colname].values
    bars = ax.bar(x + (j - 0.5) * (w + 0.02), vals, w, color=list(BIN_COL.values()), edgecolor=SURF, linewidth=2,
                  hatch=None if j == 0 else "///")
    for xi, v in zip(x + (j - 0.5) * (w + 0.02), vals): ax.text(xi, v + 0.15, f"{v:.1f}%", ha="center", fontsize=8, color=INK2)
ax.set_xticks(x, [b.replace(" (", "\n(") for b in BIN_COL])
ax.set_ylabel("% of CD8+ T cells")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#9c9b96", label="PD-1+ (exhaustion marker)"), Patch(facecolor="#9c9b96", hatch="///", label="Ki-67+ (proliferating)")], frameon=False, fontsize=9)
ax.set_title("C. Activation state by distance (all patients)", loc="left", fontsize=11)
for ax in axs:
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.suptitle(f"CD8+ T cells at the colorectal cancer invasive front — {len(cd8):,} cells, {pat.patients.nunique()} patients, 140 regions (Schürch et al. 2020)",
             x=0.01, ha="left", fontsize=12.5)
fig.tight_layout(); fig.savefig("fig2_distance_summary.png", dpi=200); plt.close(fig)
print("spot", spot)