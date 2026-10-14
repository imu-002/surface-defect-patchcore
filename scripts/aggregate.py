"""Aggregate per-category result JSONs into tables (markdown) and figures.

    python scripts/aggregate.py --runs results/standin results/standin_subtle --names "Standard" "Subtle" --out results/report
"""
import argparse, glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3 (labels printed on bars: aqua is <3:1)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "text.color": INK})


def load(run):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run, "*.json")))]


def nm(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def bars(ax, groups, series, labels, ylim=(0, 1.08), fmt="{:.2f}", legend=True):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, (s, lab) in enumerate(zip(series, labels)):
        b = ax.bar(x + (i - (n - 1) / 2) * w, s, w * 0.92, color=C[i], label=lab)
        for r, v in zip(b, s):
            if not np.isnan(v):
                ax.text(r.get_x() + r.get_width() / 2, v + 0.012, fmt.format(v), ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(*ylim)
    if legend and n > 1:
        ax.legend(frameon=False, ncol=n, loc="lower center", bbox_to_anchor=(0.5, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablation", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {n: load(r) for n, r in zip(a.names, a.runs)}
    empty = [n for n, v in R.items() if not v]
    if empty:
        raise SystemExit(f"no result files found for {empty}: the experiment step did not finish. Fix the error above it first.")
    md = []

    for name, res in R.items():
        md.append(f"\n### {name}: detection, localisation and speed per category\n")
        md.append("| Category | Image AUROC | Pixel AUROC | AU-PRO | F1 @ calibrated thr. | FPR @ thr. | Recall @ thr. | Best F1 (oracle) | ms/image | FPS |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        rows = []
        for r in res:
            p = r["patchcore"]; t = p.get("at_calibrated_threshold", {}); L = r["latency_ms"]
            f = lambda v: "n/a" if v is None else f"{v:.3f}"
            md.append(f"| {r['category']} | {f(p['image_auroc'])} | {f(p['pixel_auroc'])} | {f(p['aupro'])} | {f(t.get('f1'))} | "
                      f"{f(t.get('false_positive_rate'))} | {f(t.get('recall'))} | {f(p['best_f1_oracle'])} | {L['total_ms']:.0f} | {L['fps']:.1f} |")
        m = lambda k: nm([r["patchcore"][k] for r in res])
        mt = lambda k: nm([r["patchcore"].get("at_calibrated_threshold", {}).get(k) for r in res])
        md.append(f"| **Mean** | **{m('image_auroc'):.3f}** | **{m('pixel_auroc'):.3f}** | **{m('aupro'):.3f}** | **{mt('f1'):.3f}** | "
                  f"**{mt('false_positive_rate'):.3f}** | **{mt('recall'):.3f}** | **{m('best_f1_oracle'):.3f}** | **{nm([r['latency_ms']['total_ms'] for r in res]):.0f}** | "
                  f"**{nm([r['latency_ms']['fps'] for r in res]):.1f}** |")

        md.append(f"\n**{name}: image AUROC by defect type (good vs that type)**\n")
        types = sorted({t for r in res for t in r["per_type"]})
        md.append("| Category | " + " | ".join(types) + " |")
        md.append("|---|" + "---|" * len(types))
        for r in res:
            md.append(f"| {r['category']} | " + " | ".join(f"{r['per_type'][t]['image_auroc']:.3f}" if t in r["per_type"] else "-" for t in types) + " |")

        fig, ax = plt.subplots(figsize=(8.2, 3.4))
        cats = [r["category"] for r in res]
        bars(ax, cats, [[r["patchcore"]["image_auroc"] for r in res], [r["patchcore"]["pixel_auroc"] for r in res],
                        [r["patchcore"]["aupro"] for r in res]], ["Image AUROC", "Pixel AUROC", "AU-PRO"])
        ax.set_ylabel("score"); fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"fig_detection_{name.lower()}.png"), dpi=140); plt.close(fig)

    # ---------- baselines ----------
    meths = ["PatchCore (ours)", "Global-feature kNN", "One-class SVM", "Isolation Forest"]
    md.append("\n### Image-level AUROC: PatchCore vs unsupervised global-feature baselines (mean over categories)\n")
    md.append("| Dataset | " + " | ".join(meths) + " |"); md.append("|---|" + "---|" * len(meths))
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ser = []
    for name, res in R.items():
        v = [nm([r["baselines_image_auroc"][k] for r in res]) for k in meths]; ser.append(v)
        md.append(f"| {name} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    bars(ax, [m.replace(" (ours)", "\n(ours)").replace("Global-feature kNN", "Global-feature\nkNN") for m in meths], ser, list(R)); ax.set_ylabel("mean image AUROC")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_baselines.png"), dpi=140); plt.close(fig)

    # ---------- supervised vs unseen ----------
    md.append("\n### Supervised classifiers vs PatchCore when the defect type was NOT in the training data (mean image AUROC)\n")
    md.append("| Dataset | Scenario | SVM (supervised) | Random Forest (supervised) | PatchCore (no defect labels) |"); md.append("|---|---|---|---|---|")
    fig, axs = plt.subplots(1, len(R), figsize=(4.4 * len(R), 3.2), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (name, res) in zip(axs, R.items()):
        vals = []
        for sc in ("seen", "unseen"):
            s = nm([r["supervised_vs_unseen"]["supervised"][sc]["SVM"] for r in res])
            f = nm([r["supervised_vs_unseen"]["supervised"][sc]["RandomForest"] for r in res])
            p = nm([r["supervised_vs_unseen"]["patchcore"][sc] for r in res])
            vals.append((s, f, p)); md.append(f"| {name} | {sc} defect types | {s:.3f} | {f:.3f} | {p:.3f} |")
        bars(ax, ["defect type\nin training", "defect type\nheld out"], [[v[i] for v in vals] for i in range(3)],
             ["SVM", "Random Forest", "PatchCore"], ylim=(0, 1.1)); ax.set_title(name, fontsize=10, color=INK, pad=26)
    axs[0].set_ylabel("mean image AUROC"); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_unseen.png"), dpi=140); plt.close(fig)

    # ---------- defect type classification ----------
    md.append("\n### Defect-type identification: 5-fold cross-validated accuracy on defective images (mean over categories)\n")
    md.append("| Dataset | Features | kNN | SVM | Random Forest |"); md.append("|---|---|---|---|---|")
    fig, axs = plt.subplots(1, len(R), figsize=(4.4 * len(R), 3.2), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (name, res) in zip(axs, R.items()):
        grid = {}
        for fs in ("hand", "deep", "both"):
            row = [nm([r["type_classification"].get(f"{m}/{fs}", {}).get("accuracy") for r in res]) for m in ("kNN", "SVM", "RandomForest")]
            grid[fs] = row
            md.append(f"| {name} | {dict(hand='shape+colour only', deep='deep features only', both='both')[fs]} | " + " | ".join(f"{x:.3f}" for x in row) + " |")
        bars(ax, ["kNN", "SVM", "Random\nForest"], [grid["hand"], grid["deep"], grid["both"]], ["shape+colour", "deep", "both"], ylim=(0, 1.1))
        ax.set_title(name, fontsize=10, color=INK, pad=26)
    axs[0].set_ylabel("accuracy"); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_types.png"), dpi=140); plt.close(fig)

    # ---------- confusion matrices (SVM, both) ----------
    for name, res in R.items():
        cs = [r for r in res if r["type_classification"].get("SVM/both", {}).get("confusion")]
        if not cs:
            continue
        cols = min(3, len(cs)); rows_ = int(np.ceil(len(cs) / cols))
        fig, axs = plt.subplots(rows_, cols, figsize=(3.3 * cols, 3.2 * rows_)); axs = np.atleast_1d(axs).ravel()
        for ax, r in zip(axs, cs):
            tc = r["type_classification"]["SVM/both"]; cm = np.array(tc["confusion"], float); lab = tc["labels"]
            ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1); ax.grid(False)
            ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=7)
            for i in range(len(lab)):
                for j in range(len(lab)):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7, color="white" if cm[i, j] / cm[i].sum() > 0.5 else INK)
            ax.set_title(f"{r['category']}  (acc {tc['accuracy']:.2f})", fontsize=9)
        for ax in axs[len(cs):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"fig_confusion_{name.lower()}.png"), dpi=130); plt.close(fig)

    # ---------- ablation ----------
    for path in a.ablation:
        rows = json.load(open(path)); tag = os.path.splitext(os.path.basename(path))[0]
        md.append(f"\n### Ablation ({tag}): embedding size x memory-bank size\n")
        md.append("| Category | Embedding dim | Coreset ratio | Bank patches | Image AUROC | Pixel AUROC | AU-PRO | Search ms | Est. FPS |"); md.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['category']} | {r['pca_dim']} | {r['coreset_ratio']} | {r['bank']} | {r['image_auroc']:.3f} | {r['pixel_auroc']:.3f} | {r['aupro']:.3f} | {r['search_ms']:.0f} | {r['est_fps']:.1f} |")
        cats = sorted({r["category"] for r in rows})
        fig, axs = plt.subplots(1, len(cats), figsize=(4.6 * len(cats), 3.3), sharey=True); axs = np.atleast_1d(axs)
        for ax, c in zip(axs, cats):
            for i, d in enumerate(sorted({r["pca_dim"] for r in rows})):
                pts = sorted([r for r in rows if r["category"] == c and r["pca_dim"] == d], key=lambda r: r["est_fps"])
                ax.plot([p["est_fps"] for p in pts], [p["aupro"] for p in pts], "-o", color=C[i], lw=2, ms=6, label=f"{d}-dim embedding")
                for p in pts:
                    ax.annotate(f"{p['coreset_ratio']:.0%}", (p["est_fps"], p["aupro"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
            ax.set_xlabel("estimated end-to-end FPS (2 CPU cores)"); ax.set_title(c, fontsize=10); ax.legend(frameon=False, fontsize=8)
        axs[0].set_ylabel("AU-PRO"); fig.tight_layout(); fig.savefig(os.path.join(a.out, f"fig_ablation_{tag}.png"), dpi=140); plt.close(fig)

    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
