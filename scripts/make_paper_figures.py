"""Generate the symmetry-SCORING paper's figures + leaderboard table from results/
into the sibling image-symmetry-scoring-paper repo. Run from the imgsym repo root:

    conda run -n imgsym python scripts/make_paper_figures.py

Produces (in ../image-symmetry-scoring-paper/):
  figures/fig_leaderboard.pdf       single-axis skill, 13 methods, per-dataset range
  figures/fig_single_vs_multi.pdf   single vs multi mean skill (ordering preserved)
  figures/fig_sensitivity.pdf       skill vs shift% and vs rotation deg (all methods)
  figures/fig_stage_ablation.pdf    skill vs backbone stage (mid-scale; ViT flat)
  figures/fig_hog_cell_ablation.pdf HOG cell sweep (+ window sweep when available)
  figures/fig_speed_accuracy.pdf    cost-skill plane (ms per crop vs mean skill)
  data/leaderboard.csv              derived mean skills (single + multi)
  tables/tab_leaderboard.tex        LaTeX leaderboard (input into the manuscript)

STYLE: one system across all figures -- Okabe-Ito colorblind-safe palette, a fixed
(color, marker, linestyle) identity per method, and grayscale-safe encodings
(markers / dash patterns / hatching, never color alone).
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imgsym.evaluation import discrimination_skill
from imgsym.evaluation.scoring_metrics import infer_direction

PAPER = os.path.join("..", "image-symmetry-scoring-paper")
FIG, DATA, TAB = (os.path.join(PAPER, d) for d in ("figures", "data", "tables"))
for d in (FIG, DATA, TAB):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- shared style
plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "lines.linewidth": 1.4,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
SIZE_S = (5.4, 3.4)   # single-panel figures
SIZE_W = (7.6, 3.2)   # two-panel figures

# Okabe-Ito palette; every method keeps ONE identity (color/marker/linestyle)
# across all figures. Non-highlighted methods render as light context lines/bars.
C_DEEP, C_HOG, C_ALEX = "#0072B2", "#D55E00", "#009E73"
C_GABOR, C_EROS = "#CC79A7", "#56B4E9"
NEUTRAL, CONTEXT = "#B8B8B8", "#CCCCCC"
STYLE = {
    "deep_features": dict(color=C_DEEP,  marker="o", ls="-"),
    "hog":           dict(color=C_HOG,   marker="s", ls="-"),
    "alexnet":       dict(color=C_ALEX,  marker="^", ls=(0, (4, 1.5))),
    "gabor":         dict(color=C_GABOR, marker="D", ls=(0, (1, 1))),
    "eros":          dict(color=C_EROS,  marker="v", ls=(0, (5, 1, 1, 1))),
}
HI = {"deep_features": C_DEEP, "hog": C_HOG}            # leaderboard highlights

# Display names used in the manuscript (library ids stay in CSVs / result files).
DISPLAY = {
    "pixel_correlation": "PixCorr", "sliding_window": "SlideWin", "dct": "DCT",
    "eros": "EROS", "gradient": "Grad", "multi_scale_gradient": "MS-Grad",
    "hog": "HOG", "phog": "PHOG", "gabor": "Gabor", "weighted_binary": "WBS",
    "local_global": "PatchNN", "alexnet": "AlexNet-C2", "deep_features": "DeepFeat",
}


def disp(m):
    return DISPLAY.get(m, m)


def load_disc(path):
    """results/discrimination*.csv -> {method: {ds: (skill, lo, hi, margin)}}, [ds]."""
    rows = list(csv.reader(open(path)))
    dss = [c[:-6] for c in rows[0][1:] if c.endswith("_skill")]
    out = {}
    for r in rows[1:]:
        out[r[0]] = {ds: tuple(float(x) for x in r[1 + 4 * i:5 + 4 * i])
                     for i, ds in enumerate(dss)}
    return out, dss


single, DS_S = load_disc("results/discrimination.csv")
multi, DS_M = load_disc("results/discrimination_multi.csv")
methods = list(single)
mean_s = {m: float(np.mean([single[m][d][0] for d in DS_S])) for m in methods}
mean_m = {m: float(np.mean([multi[m][d][0] for d in DS_M])) for m in methods}
order = sorted(methods, key=lambda m: mean_s[m], reverse=True)        # by single skill


# --- FIG 1: single-axis leaderboard (mean skill; whiskers = per-dataset range) ---
fig, ax = plt.subplots(figsize=SIZE_S)
y = np.arange(len(order))[::-1]
vals = [mean_s[m] for m in order]
err = [[mean_s[m] - min(single[m][d][0] for d in DS_S) for m in order],
       [max(single[m][d][0] for d in DS_S) - mean_s[m] for m in order]]
colors = [HI.get(m, NEUTRAL) for m in order]
ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.4)
ax.errorbar(vals, y, xerr=err, fmt="none", ecolor="#333", elinewidth=0.8, capsize=2)
ax.set_yticks(y)
ax.set_yticklabels([disp(m) for m in order])
for lbl, m in zip(ax.get_yticklabels(), order):
    if m in HI:
        lbl.set_fontweight("bold")
ax.set_xlabel(r"discrimination skill $2\,\mathrm{AUC}-1$ (mean over single-axis datasets; whiskers span datasets)")
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.axvline(0, color="k", lw=0.5)
fig.savefig(os.path.join(FIG, "fig_leaderboard.pdf"))
plt.close(fig)


# --- FIG 2: single vs multi mean skill (all 13 methods; colors as in FIG 1,
#     solid vs hatched = B/W safe; whiskers span the per-dataset range) ---
from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=(6.0, 3.3))
x = np.arange(len(order))
bar_cols = [HI.get(m, NEUTRAL) for m in order]
err_s = [[mean_s[m] - min(single[m][d][0] for d in DS_S) for m in order],
         [max(single[m][d][0] for d in DS_S) - mean_s[m] for m in order]]
err_m = [[mean_m[m] - min(multi[m][d][0] for d in DS_M) for m in order],
         [max(multi[m][d][0] for d in DS_M) - mean_m[m] for m in order]]
ax.bar(x - 0.2, [mean_s[m] for m in order], 0.4,
       color=bar_cols, edgecolor="white", linewidth=0.4)
ax.errorbar(x - 0.2, [mean_s[m] for m in order], yerr=err_s, fmt="none",
            ecolor="#333", elinewidth=0.7, capsize=1.5)
ax.bar(x + 0.2, [mean_m[m] for m in order], 0.4,
       color="white", edgecolor=bar_cols, linewidth=0.9, hatch="///")
ax.errorbar(x + 0.2, [mean_m[m] for m in order], yerr=err_m, fmt="none",
            ecolor="#333", elinewidth=0.7, capsize=1.5)
ax.set_xticks(x)
ax.set_xticklabels([disp(m) for m in order], rotation=45, ha="right")
for lbl, m in zip(ax.get_xticklabels(), order):
    if m in HI:
        lbl.set_fontweight("bold")
ax.set_ylabel("mean discrimination skill")
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.legend(handles=[Patch(facecolor="#5A5A5A", label="single-axis"),
                   Patch(facecolor="white", edgecolor="#5A5A5A", hatch="///",
                         label="multi-axis")], frameon=False)
fig.savefig(os.path.join(FIG, "fig_single_vs_multi.pdf"))
plt.close(fig)


# --- FIG 3: stage ablation, easy + hard subsets (parse the .md tables) ---
def parse_stage_md(path):
    out = {}
    for line in open(path):
        if line.startswith("|") and "+" in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            sk = [float(t) for t in parts[1].split() if t[0] in "+-"]
            if sk:
                out[parts[0].split(".")[0]] = sk
    return out


stage_easy_md = next(p for p in ("results/network_stage_ablation_easy.md",
                                 "results/network_stage_ablation.md") if os.path.exists(p))
panels = [("easy subset (NYU + symComp17)", parse_stage_md(stage_easy_md))]
if os.path.exists("results/network_stage_ablation_hard.md"):
    panels.append(("hard subset (PIX2PER-art)",
                   parse_stage_md("results/network_stage_ablation_hard.md")))
fig, axes = plt.subplots(1, len(panels), figsize=SIZE_W if len(panels) == 2 else SIZE_S,
                         sharey=True)
axes = np.atleast_1d(axes)
max_stages = 0
for ax, (title, stage) in zip(axes, panels):
    max_stages = max(max_stages, max(len(sk) for sk in stage.values()))
    for bb, sk in stage.items():               # actual stage index; lines end at depth
        xs = np.arange(len(sk))
        if bb.startswith("vit"):
            ax.plot(xs, sk, color="#3B3B3B", ls=(0, (4, 2)), lw=1.5, marker="o",
                    ms=3.6, mfc="white", zorder=3)
        elif bb == "mambaout_base":            # DeepFeat's default backbone
            ax.plot(xs, sk, color=C_DEEP, ls="-", lw=1.8, marker="o", ms=3.4, zorder=4)
        else:
            ax.plot(xs, sk, color=CONTEXT, ls="-", lw=0.9, zorder=1)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlabel("backbone stage")
    ax.set_xticks(range(max_stages))
axes[0].set_ylabel("discrimination skill")
axes[0].set_ylim(0.3, 1.0)
axes[0].plot([], [], color=CONTEXT, lw=0.9, label="hierarchical backbones (14)")
axes[0].plot([], [], color=C_DEEP, lw=1.8, marker="o", ms=3.4,
             label="MambaOut-Base (DeepFeat default)")
axes[0].plot([], [], color="#3B3B3B", ls=(0, (4, 2)), lw=1.5, marker="o", ms=3.6,
             mfc="white", label="columnar ViT (2)")
axes[0].legend(frameon=False, loc="lower left")
fig.savefig(os.path.join(FIG, "fig_stage_ablation.pdf"))
plt.close(fig)


# --- Appendix table: the backbone pool with best stage/skill per subset ---
def parse_stage_best(path):
    """md table -> {full timm id: (n_stages, 'stageK', '+0.94 [+0.90,+0.97]')}."""
    out = {}
    for line in open(path):
        if line.startswith("|") and "+" in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            sk = [float(t) for t in parts[1].split() if t[0] in "+-"]
            if sk and len(parts) >= 4:
                out[parts[0]] = (len(sk), parts[2], parts[3])
    return out


easy_best = parse_stage_best(stage_easy_md)
hard_path = "results/network_stage_ablation_hard.md"
hard_best = parse_stage_best(hard_path) if os.path.exists(hard_path) else {}
if hard_best:
    def _hard_skill(b):
        return float(hard_best[b][2].split()[0]) if b in hard_best else -9.0
    bbs = sorted(easy_best, key=_hard_skill, reverse=True)
    with open(os.path.join(TAB, "tab_backbones.tex"), "w") as fh:
        fh.write("% auto-generated by scripts/make_paper_figures.py\n")
        fh.write("\\begin{tabularx}{\\fulllength}{Xccccc}\n\\toprule\n")
        fh.write("\\textbf{Backbone (\\texttt{timm} identifier)} & \\textbf{Stages}"
                 " & \\multicolumn{2}{c}{\\textbf{Easy subset}}"
                 " & \\multicolumn{2}{c}{\\textbf{Hard subset}}\\\\\n")
        fh.write(" & & best & skill [95\\% CI] & best & skill [95\\% CI]\\\\\n\\midrule\n")
        for b in bbs:
            n, e_st, e_sk = easy_best[b]
            h_st, h_sk = (hard_best[b][1], hard_best[b][2]) if b in hard_best else ("--", "--")
            bid = b.replace("_", "\\_")
            fh.write(f"\\texttt{{\\footnotesize {bid}}} & {n}"
                     f" & {e_st.replace('stage', '')} & {e_sk}"
                     f" & {h_st.replace('stage', '')} & {h_sk}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabularx}\n")
    print("  + tab_backbones.tex")


# --- FIG 4: sensitivity (skill vs shift% / rotation deg; ALL methods, 5 highlighted) ---
pis = json.load(open("results/per_image_scores.json"))
shifts, angles = pis["meta"]["shifts"], pis["meta"]["angles"]
pdata = pis["data"]


# Direction per method, fixed once from the standard pool -- per-slice
# re-inference silently flips the sign of near-chance cells.
_std = set(pis["meta"]["std_tags"])
_direction = {}
for m in methods:
    pool = [(rec["true"], [s for t, s in rec["wrong"] if t in _std and s is not None])
            for ds in pdata for rec in pdata[ds][m]]
    pool = [(t, w) for t, w in pool if w]
    _direction[m] = infer_direction([t for t, _ in pool],
                                    [x for _, w in pool for x in w])


def skill_for(tag):
    """pooled discrimination skill across datasets using only wrongs with this tag."""
    per_m = {}
    for m in methods:
        pairs = []
        for ds in pdata:
            for rec in pdata[ds][m]:
                w = [s for t, s in rec["wrong"] if t == tag and s is not None]
                if w:
                    pairs.append((rec["true"], w))
        per_m[m] = (discrimination_skill(pairs, higher_is_better=_direction[m])
                    if pairs else np.nan)
    return per_m


shift_sk = {s: skill_for(f"shift_{s:g}") for s in shifts}
rot_sk = {a: skill_for(f"rot_{a:g}") for a in angles}
SHOW = ["deep_features", "hog", "alexnet", "gabor", "eros"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=SIZE_W, sharey=True)
for m in methods:                                   # context: every other method
    if m in SHOW:
        continue
    a1.plot([s * 100 for s in shifts], [shift_sk[s][m] for s in shifts],
            color=CONTEXT, lw=0.8, zorder=1)
    a2.plot(angles, [rot_sk[a][m] for a in angles], color=CONTEXT, lw=0.8, zorder=1)
for m in SHOW:                                      # highlighted, fixed identities
    st = STYLE[m]
    a1.plot([s * 100 for s in shifts], [shift_sk[s][m] for s in shifts],
            color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.5, lw=1.6,
            zorder=3, label=disp(m))
    a2.plot(angles, [rot_sk[a][m] for a in angles],
            color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.5, lw=1.6,
            zorder=3, label=disp(m))
a1.set_xlabel("perpendicular shift (% of size)")
a2.set_xlabel("rotation (degrees)")
a1.set_ylabel("discrimination skill")
for a, ticks in ((a1, [s * 100 for s in shifts]), (a2, list(angles))):
    a.set_xscale("log")
    a.set_xticks(ticks)
    a.set_xticklabels([f"{t:g}" for t in ticks])
    a.minorticks_off()
    a.set_ylim(0, 1)
    a.set_yticks(np.arange(0, 1.01, 0.2))
a2.plot([], [], color=CONTEXT, lw=0.8, label="other methods")
a2.legend(frameon=False)
fig.savefig(os.path.join(FIG, "fig_sensitivity.pdf"))
plt.close(fig)


# --- FIG 5: HOG ablation (cell sweep; + window sweep when results exist) ---
hogp = "results/hog_ablation.csv"
winp = "results/hog_ablation_win.csv"
if os.path.exists(hogp):
    hrows = list(csv.DictReader(open(hogp)))
    two_panel = os.path.exists(winp)
    if two_panel:
        fig, (axc, axw) = plt.subplots(1, 2, figsize=SIZE_W, sharey=True)
    else:
        fig, axc = plt.subplots(figsize=SIZE_S)
        axw = None
    # deliberately avoids the DeepFeat/HOG identity colors (blue/vermillion):
    # this figure ablates HOG internals, it does not compare methods.
    for signed, col, mk, ls, lbl in [("False", "#7570B3", "o", "-", "unsigned"),
                                     ("True", "#666666", "s", (0, (4, 1.5)), "signed")]:
        pts = sorted((int(r["cell"]), float(r["skill"])) for r in hrows
                     if r["signed"] == signed and int(r["nbins"]) == 18)
        if pts:
            xs, ys = zip(*pts)
            axc.plot(xs, ys, color=col, marker=mk, ls=ls, ms=4, label=lbl)
    axc.set_xscale("log", base=2)
    axc.set_xticks([4, 8, 16, 32])
    axc.set_xticklabels([4, 8, 16, 32])
    axc.minorticks_off()
    axc.set_xlabel("HOG cell size (px), window 64")
    axc.set_ylabel("discrimination skill")
    axc.legend(frameon=False, title="orientation (18 bins)")
    if two_panel:
        wrows = list(csv.DictReader(open(winp)))
        shades = {32: "#CBC9E2", 64: "#8683BD", 128: "#4A3B8C"}   # purple ramp
        marks = {32: "o", 64: "s", 128: "^"}
        for win in (32, 64, 128):
            pts = sorted((int(r["cell"]), float(r["skill"])) for r in wrows
                         if int(r["win"]) == win)
            if pts:
                xs, ys = zip(*pts)
                axw.plot(xs, ys, color=shades[win], marker=marks[win], ms=4,
                         label=f"window {win}")
        axw.set_xscale("log", base=2)
        cells_all = sorted({int(r["cell"]) for r in wrows})
        axw.set_xticks(cells_all)
        axw.set_xticklabels(cells_all)
        axw.minorticks_off()
        axw.set_xlabel("HOG cell size (px), unsigned")
        axw.legend(frameon=False, title="descriptor window (px)")
    fig.savefig(os.path.join(FIG, "fig_hog_cell_ablation.pdf"))
    plt.close(fig)
    print("  + fig_hog_cell_ablation.pdf" + (" (with window panel)" if two_panel else ""))
else:
    print("  (skipped fig_hog_cell_ablation: run scripts/run_hog_ablation.py first)")


# --- FIG 6: speed vs accuracy frontier (median ms/crop vs mean single-axis skill) ---
timp = "results/method_timings.csv"
if os.path.exists(timp):
    trows = {r["method"]: r for r in csv.DictReader(open(timp))}
    tm = [m for m in methods if m in trows]
    fig, ax = plt.subplots(figsize=SIZE_S)
    ms = {m: float(trows[m]["median_s"]) * 1e3 for m in tm}
    ax.errorbar([ms[m] for m in tm], [mean_s[m] for m in tm],
                xerr=[[ms[m] - float(trows[m]["p25_s"]) * 1e3 for m in tm],
                      [float(trows[m]["p75_s"]) * 1e3 - ms[m] for m in tm]],
                fmt="none", ecolor="#BBBBBB", elinewidth=0.7, zorder=1)
    for m in tm:
        big = m in HI
        st = STYLE.get(m)
        ax.scatter(ms[m], mean_s[m], s=48 if big else 22,
                   color=HI.get(m, "#888888"),
                   marker=(st["marker"] if st else "o"), zorder=3)
        ax.annotate(disp(m), (ms[m], mean_s[m]), textcoords="offset points",
                    xytext=(5, -2), fontsize=8 if big else 6.5,
                    fontweight="bold" if big else "normal",
                    color=HI.get(m, "#555555"))
    ax.set_xscale("log")
    ax.set_xlim(min(ms.values()) * 0.4, max(ms.values()) * 8)
    ax.set_xlabel("median scoring time per crop (ms, CPU; whiskers = IQR)")
    ax.set_ylabel("mean discrimination skill (single-axis)")
    fig.savefig(os.path.join(FIG, "fig_speed_accuracy.pdf"))
    plt.close(fig)
    print("  + fig_speed_accuracy.pdf")
else:
    print("  (skipped fig_speed_accuracy: run scripts/time_methods.py first)")


# --- derived data CSV + LaTeX leaderboard table ---
with open(os.path.join(DATA, "leaderboard.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["method", "single_mean", "multi_mean"]
               + [f"single_{d}" for d in DS_S] + [f"multi_{d}" for d in DS_M])
    for m in order:
        w.writerow([m, f"{mean_s[m]:.3f}", f"{mean_m[m]:.3f}"]
                   + [f"{single[m][d][0]:.3f}" for d in DS_S]
                   + [f"{multi[m][d][0]:.3f}" for d in DS_M])

with open(os.path.join(TAB, "tab_leaderboard.tex"), "w") as fh:
    fh.write("% auto-generated by scripts/make_paper_figures.py\n")
    fh.write("\\begin{tabularx}{\\textwidth}{lCC}\n\\toprule\n")
    fh.write("\\textbf{Method} & \\textbf{Single-axis} & \\textbf{Multi-axis}\\\\\n\\midrule\n")
    for m in order:
        nm = disp(m).replace("_", "\\_")
        bold = "\\textbf{%s}" if m in HI else "%s"
        fh.write(f"{bold % nm} & {mean_s[m]:+.2f} & {mean_m[m]:+.2f}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabularx}\n")

print("wrote figures + leaderboard.csv + tab_leaderboard.tex to", PAPER)
print("single order:", " > ".join(f"{m} {mean_s[m]:.2f}" for m in order[:4]), "...")
print("multi  order:", " > ".join(f"{m} {mean_m[m]:.2f}" for m in sorted(methods, key=lambda m: mean_m[m], reverse=True)[:4]), "...")
