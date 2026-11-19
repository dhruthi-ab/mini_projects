import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def make_plots(S, sweep, ks, out, names):
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cols = ["MAP", "MRR", "P@10", "R@10", "R@100", "nDCG@10"]
    w = 0.8 / len(names); x = np.arange(len(cols))
    for i, m in enumerate(names):
        ax.bar(x + i * w, S.loc[m, cols], w, label=m, color=colors[i % 4])
    ax.set_xticks(x + w * (len(names) - 1) / 2); ax.set_xticklabels(cols); ax.set_ylabel("score")
    ax.set_title("Method comparison (held-out test topics)"); ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{out}/comparison_bar.png", dpi=200); plt.close(fig)

    for metric, fname in (("P", "precision_at_k"), ("R", "recall_at_k"), ("nDCG", "ndcg_at_k")):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for i, m in enumerate(names):
            ax.plot(ks, [S.loc[m, f"{metric}@{k}"] for k in ks], marker="o", label=m, color=colors[i % 4])
        ax.set_xscale("log"); ax.set_xticks(list(ks)); ax.set_xticklabels(ks)
        ax.set_xlabel("K"); ax.set_ylabel(f"{metric}@K"); ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
        fig.savefig(f"{out}/{fname}.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep.index, sweep.values, marker="o", color=colors[2])
    ax.set_xlabel("alpha (weight on BM25)"); ax.set_ylabel("validation MAP"); ax.set_title("Fusion weight sweep")
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fig.savefig(f"{out}/alpha_sweep.png", dpi=200); plt.close(fig)
