"""CD8+ T cell to tumour-cell distances at the CRC invasive front (Schürch et al. 2020 CODEX data)."""
import sys, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

UM_PER_PX = 0.377442          # paper STAR Methods: lateral resolution 377.442 nm/pixel
BINS = [0, 20, 100, np.inf]   # contact <=20 um, near 20-100 um, distant >100 um
LABELS = ["Contact (≤20 µm)", "Near (20–100 µm)", "Distant (>100 µm)"]
GROUP = {1: "CLR (better survival)", 2: "DII (worse survival)"}

path = sys.argv[1] if len(sys.argv) > 1 else "CRC_clusters_neighborhoods_markers.csv"
df = pd.read_pickle(path) if path.endswith(".pkl") else pd.read_csv(path)
df = df.rename(columns={"X:X": "x", "Y:Y": "y"})
rows = []
for spot, s in df.groupby("File Name"):
    tum = s[s.ClusterName == "tumor cells"]
    cd8 = s[s.ClusterName == "CD8+ T cells"]
    if len(tum) == 0 or len(cd8) == 0:
        continue
    d, _ = cKDTree(tum[["x", "y"]].values).query(cd8[["x", "y"]].values)
    out = cd8[["File Name", "patients", "groups", "x", "y", "CD8+PD-1+", "CD8+Ki67+"]].copy()
    out["dist_um"] = d * UM_PER_PX
    rows.append(out)
cd8 = pd.concat(rows)
cd8["bin"] = pd.cut(cd8.dist_um, BINS, labels=LABELS, include_lowest=True)
cd8["group"] = cd8.groups.map(GROUP)
cd8.to_csv("cd8_distances.csv", index=False)

pat = cd8.groupby(["patients", "group"]).agg(
    n=("dist_um", "size"), median_um=("dist_um", "median"),
    pct_contact=("bin", lambda b: 100 * (b == LABELS[0]).mean())).reset_index()
pat.to_csv("per_patient_summary.csv", index=False)
a, b = [pat[pat.group == g].pct_contact for g in GROUP.values()]
print(f"CD8 cells: {len(cd8)} in {cd8['File Name'].nunique()} regions, {pat.patients.nunique()} patients")
print(cd8.groupby("group").dist_um.median().round(1).to_dict())
print(pd.crosstab(cd8.group, cd8.bin, normalize="index").round(3))
print("PD-1+ / Ki67+ % by bin:\n", (cd8.groupby("bin", observed=True)[["CD8+PD-1+", "CD8+Ki67+"]].mean() * 100).round(1))
print("Per-patient % contact, CLR vs DII: Mann-Whitney p =", round(mannwhitneyu(a, b).pvalue, 3))