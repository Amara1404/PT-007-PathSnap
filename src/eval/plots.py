"""Model-comparison figures for the report, drawn only from results/*.csv and
results/preds/*.parquet (actual experiment outputs).

    python -m src.eval.plots
Writes PNGs to results/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval.metrics import row_confusion  # noqa: E402

RES = Path("results")
FIG = RES / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1, 2
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def _hbar(ax, labels, values, color=BLUE, err=None, fmt="{:.3f}", label_x=None):
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.6, color=color, xerr=err,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3} if err is not None else None)
    ends = label_x if label_x is not None else [v + (e or 0) for v, e in
                                                 zip(values, err if err is not None else [0] * len(values))]
    for yi, v, x in zip(y, values, ends):
        ax.text(x + 0.008 * ax.get_xlim()[1], yi, fmt.format(v), va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()


def cv_f1(cmp, folds):
    d = cmp.sort_values("cv_macro_f1_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 1.2))
    ax.set_xlim(0.5, 1.0)
    fold_max = [max(folds[folds.model == m].val_macro_f1.max(), mu + sd)
                for m, mu, sd in zip(d.model, d.cv_macro_f1_mean, d.cv_macro_f1_std)]
    _hbar(ax, d.model, d.cv_macro_f1_mean, err=d.cv_macro_f1_std, label_x=fold_max)
    for i, m in enumerate(d.model):
        f = folds[folds.model == m].val_macro_f1
        ax.scatter(f, np.full(len(f), i), s=10, color=INK, zorder=3, alpha=0.7)
    ax.set_xlabel("Macro F1 (candidate rows), mean ± std over 5 trip-grouped folds; dots = folds")
    ax.set_title("Cross-validation macro F1 by model")
    _save(fig, "cv_macro_f1.png")


def student_test(cmp):
    d = cmp.sort_values("st_top1_acc", ascending=False)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(d) + 1.2))
    h = 0.36
    ax.barh(y - h / 2 - 0.01, d.st_top1_acc, height=h, color=BLUE, label="Top-1 accuracy (observations)")
    ax.barh(y + h / 2 + 0.01, d.st_macro_f1, height=h, color=ORANGE, label="Macro F1 (candidate rows)")
    for yi, a, f in zip(y, d.st_top1_acc, d.st_macro_f1):
        ax.text(a + 0.005, yi - h / 2, f"{a:.3f}", va="center", fontsize=8, color=INK2)
        ax.text(f + 0.005, yi + h / 2, f"{f:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, d.model); ax.invert_yaxis(); ax.set_xlim(0.4, 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=9)
    ax.set_title("Student-test Top-1 accuracy and macro F1 (10 trips)", pad=26)
    _save(fig, "student_test_top1_f1.png")


def logloss(cmp):
    d = cmp.dropna(subset=["st_log_loss"]).sort_values("st_log_loss")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 1.2))
    _hbar(ax, d.model, d.st_log_loss)
    ax.set_xlabel("Log loss on student-test candidate rows (lower is better)")
    ax.set_title("Probability quality: student-test log loss")
    _save(fig, "log_loss.png")


def overfit(cmp):
    d = cmp.sort_values("cv_macro_f1_mean", ascending=False)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 1.4))
    ax.hlines(y, d.cv_macro_f1_mean, d.train_macro_f1_mean, color=GRID, linewidth=2)
    ax.scatter(d.train_macro_f1_mean, y, s=40, color=ORANGE, label="Training folds", zorder=3)
    ax.scatter(d.cv_macro_f1_mean, y, s=40, color=BLUE, label="Validation folds", zorder=3)
    ax.set_yticks(y, d.model); ax.invert_yaxis(); ax.set_xlim(0.5, 1.01)
    ax.set_xlabel("Macro F1, mean over 5 folds (gap = overfitting)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Training vs validation macro F1")
    _save(fig, "train_vs_validation_f1.png")


def timing(cmp):
    d = cmp.sort_values("inference_ms_per_1k_obs")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 1.2))
    _hbar(ax, d.model, d.inference_ms_per_1k_obs, fmt="{:.1f}")
    ax.set_xscale("log")
    ax.set_xlabel("Inference time per 1,000 GPS observations (ms, log scale)")
    ax.set_title("Inference speed on student-test trips")
    _save(fig, "inference_time.png")


def confusions(cmp):
    names = list(cmp.sort_values("cv_macro_f1_mean", ascending=False).model)
    n = len(names); cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.9 * rows), squeeze=False)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    for ax, m in zip(axes.flat, names):
        p = pd.read_parquet(RES / "preds" / f"{m}_student_test.parquet")
        cm = row_confusion(p.label, p.prob)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.1%}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["pred: other", "pred: match"]); ax.set_yticks([0, 1], ["true: other", "true: match"])
        ax.grid(False); ax.set_title(m, fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Student-test confusion matrices (candidate rows, threshold 0.5; colour = row share)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "confusion_matrices.png")


def histories():
    for f in sorted((RES / "history").glob("*_history.csv")):
        h = pd.read_csv(f)
        x = h.columns[0]
        if "val_loss" in h:      # same measure for both curves -> one panel
            fig, ax = plt.subplots(figsize=(6.5, 3.4))
            ax.plot(h[x], h.train_loss, color=BLUE, linewidth=2, label="Training loss")
            ax.plot(h[x], h.val_loss, color=ORANGE, linewidth=2, label="Validation loss")
            ax.set_ylabel("Log loss"); ax.legend(fontsize=9); axes = [ax]
        else:                    # loss vs accuracy: different measures -> two panels, never a dual axis
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
            axes[0].plot(h[x], h.train_loss, color=BLUE, linewidth=2); axes[0].set_ylabel("Training log loss")
            if "val_accuracy" in h:
                axes[1].plot(h[x], h.val_accuracy, color=ORANGE, linewidth=2)
                axes[1].set_ylabel("Validation accuracy (held-out 10%)")
        for a in axes:
            a.grid(axis="y"); a.grid(axis="x", visible=False); a.set_xlabel(x)
        name = f.stem.replace("_history", "")
        fig.suptitle(f"{name}: training history (final fit on all training trips)", x=0.01, ha="left",
                     fontsize=11, fontweight="bold", color=INK)
        fig.tight_layout()
        _save(fig, f"history_{name}.png")


def tuning_history(trials):
    t = trials.sort_values("number")
    done = t[t.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.grid(axis="y"); ax.grid(axis="x", visible=False)
    ax.scatter(done.number, done.value, s=22, color=BLUE, label="Completed trial (CV Top-1)", zorder=3)
    pr = t[t.state == "PRUNED"]
    if len(pr):
        ax.scatter(pr.number, np.full(len(pr), done.value.min()), s=18, marker="x", color=INK2,
                   label="Pruned trial (stopped early)", zorder=3)
    best = done.set_index("number").value.cummax().reindex(t.number).ffill()
    ax.step(best.index, best.values, where="post", color=ORANGE, linewidth=2, label="Best so far")
    d0 = done[done.number == 0]
    if len(d0):
        ax.annotate("untuned defaults", (0, d0.value.iat[0]), xytext=(10, 10), textcoords="offset points",
                    fontsize=9, color=INK2)
    ax.set_xlabel("Trial"); ax.set_ylabel("Mean CV Top-1 accuracy")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Bayesian optimisation of HistGradientBoosting (Optuna TPE)")
    _save(fig, "tuning_history.png")


def tuning_params(trials):
    done = trials[trials.state == "COMPLETE"]
    params = [c for c in done.columns if c.startswith("params_") and c != "params_limit_depth"]
    logs = {"params_learning_rate", "params_max_leaf_nodes", "params_min_samples_leaf", "params_l2_regularization"}
    cols = 3; rows = int(np.ceil(len(params) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows), squeeze=False, sharey=True)
    best = done.loc[done.value.idxmax()]
    for ax, p in zip(axes.flat, params):
        d = done.dropna(subset=[p])
        ax.grid(axis="y"); ax.grid(axis="x", visible=False)
        ax.scatter(d[p], d.value, s=16, color=BLUE, alpha=0.8)
        if pd.notna(best[p]):
            ax.scatter([best[p]], [best.value], s=60, facecolors="none", edgecolors=ORANGE, linewidths=2)
        if p in logs:
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(p.replace("params_", ""))
    for ax in axes[:, 0]:
        ax.set_ylabel("CV Top-1")
    for ax in list(axes.flat)[len(params):]:
        ax.axis("off")
    fig.suptitle("Trial score by hyperparameter value (orange ring = best trial)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "tuning_params.png")


def tuned_vs_untuned(cmp):
    c = cmp.set_index("model").loc[["hist_gb", "hist_gb_tuned"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [3, 1.3]})
    metrics = [("cv_macro_f1_mean", "CV macro F1"), ("cv_top1_mean", "CV Top-1"),
               ("st_top1_acc", "Student-test Top-1"), ("st_macro_f1", "Student-test macro F1")]
    y = np.arange(len(metrics)); h = 0.36
    for off, (m, color, lab) in zip((-h / 2 - 0.01, h / 2 + 0.01), (("hist_gb", INK2, "Untuned"), ("hist_gb_tuned", BLUE, "Tuned"))):
        vals = [c.loc[m, k] for k, _ in metrics]
        a1.barh(y + off, vals, height=h, color=color, label=lab)
        for yi, v in zip(y, vals):
            a1.text(v + 0.002, yi + off, f"{v:.4f}", va="center", fontsize=8, color=INK2)
    a1.set_yticks(y, [n for _, n in metrics]); a1.invert_yaxis()
    lo = min(c[[k for k, _ in metrics]].min()) - 0.03
    a1.set_xlim(lo, 1.0); a1.legend(fontsize=9, loc="lower right")
    a1.set_title("Untuned vs tuned HistGradientBoosting")
    ll = [c.loc["hist_gb", "st_log_loss"], c.loc["hist_gb_tuned", "st_log_loss"]]
    a2.barh([0, 1], ll, height=0.6, color=[INK2, BLUE])
    for yi, v in enumerate(ll):
        a2.text(v * 1.01, yi, f"{v:.4f}", va="center", fontsize=8, color=INK2)
    a2.set_yticks([0, 1], ["Untuned", "Tuned"]); a2.invert_yaxis()
    a2.set_xlim(0, max(ll) * 1.3); a2.set_title("Student-test log loss", fontsize=10)
    fig.tight_layout()
    _save(fig, "tuned_vs_untuned.png")


def main():
    cmp = pd.read_csv(RES / "model_comparison.csv")
    folds = pd.read_csv(RES / "cv_fold_scores.csv")
    cv_f1(cmp, folds); student_test(cmp); logloss(cmp); overfit(cmp); timing(cmp); confusions(cmp); histories()
    if (RES / "optuna_trials.csv").exists():
        trials = pd.read_csv(RES / "optuna_trials.csv")
        tuning_history(trials); tuning_params(trials)
    if "hist_gb_tuned" in set(cmp.model):
        tuned_vs_untuned(cmp)


if __name__ == "__main__":
    main()
