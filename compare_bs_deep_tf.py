"""
Comparaison Black-Scholes delta hedge vs Deep Hedging TensorFlow.

Le script entraine un modele Deep Hedging pour chaque mesure de risque,
avec couts de transaction, puis sauvegarde :
  - un histogramme BS hedge vs Deep hedge pour chaque mesure de risque,
  - un graphe delta_k contre S_k a chaque date t_k pour chaque mesure,
  - un fichier JSON de statistiques.

Exemple :
    python compare_bs_deep_tf.py --cost 0.01 --epochs 30
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from deep_hedging import (
    DeepHedger,
    delta_hedge_pnl,
    simulate_bs,
    summary_stats,
    train_hedger,
)


RISK_CONFIGS = [
    {"key": "cvar50", "risk": "cvar", "alpha": 0.50, "lam": 1.0, "label": "Deep CVaR 50%"},
    {"key": "cvar99", "risk": "cvar", "alpha": 0.99, "lam": 1.0, "label": "Deep CVaR 99%"},
    {"key": "entropic", "risk": "entropic", "alpha": 0.50, "lam": 1.0, "label": "Deep entropique"},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs_tf", help="Dossier de sortie.")
    parser.add_argument("--cost", type=float, default=0.01, help="Cout de transaction proportionnel.")
    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--T", type=float, default=30 / 365)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--n-train", type=int, default=60000)
    parser.add_argument("--n-test", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-paths", type=int, default=2500)
    return parser.parse_args()


def as_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def make_histogram(bs_pnl, deep_pnl, risk_label, cost, out_file):
    lo = float(min(np.quantile(bs_pnl, 0.005), np.quantile(deep_pnl, 0.005)))
    hi = float(max(np.quantile(bs_pnl, 0.995), np.quantile(deep_pnl, 0.995)))
    bins = np.linspace(lo, hi, 70)

    plt.figure(figsize=(8.5, 5.2))
    plt.hist(bs_pnl, bins=bins, alpha=0.62, density=True, label="BS delta hedge")
    plt.hist(deep_pnl, bins=bins, alpha=0.62, density=True, label=risk_label)
    plt.axvline(np.mean(bs_pnl), color="C0", linewidth=1.5)
    plt.axvline(np.mean(deep_pnl), color="C1", linewidth=1.5)
    plt.title(f"PnL terminal avec cout de transaction c = {cost:g}")
    plt.xlabel("PnL terminal")
    plt.ylabel("Densite")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=140)
    plt.close()


def make_delta_spot_grid(S, bs_delta, deep_delta, risk_label, out_file):
    """
    Affiche delta_k=f(S_k) pour toutes les dates t_k dans une grille.
    Pour limiter le bruit visuel, les trajectoires sont triees par spot.
    """
    S = np.asarray(S)
    bs_delta = np.asarray(bs_delta)
    deep_delta = np.asarray(deep_delta)
    n_steps = bs_delta.shape[1]

    n_cols = min(6, n_steps)
    n_rows = int(np.ceil(n_steps / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.45 * n_rows), sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for k in range(n_steps):
        ax = axes[k]
        order = np.argsort(S[:, k])
        spot = S[order, k]
        ax.plot(spot, bs_delta[order, k], color="black", linewidth=1.25, label="BS")
        ax.scatter(
            spot,
            deep_delta[order, k],
            s=5,
            alpha=0.22,
            color="C1",
            edgecolors="none",
            label=risk_label,
        )
        ax.set_title(f"t{k}")
        ax.set_xlim(np.quantile(spot, 0.01), np.quantile(spot, 0.99))
        ax.set_ylim(-0.15, 1.15)
        if k % n_cols == 0:
            ax.set_ylabel("position delta")
        if k >= (n_rows - 1) * n_cols:
            ax.set_xlabel("spot S_k")

    for ax in axes[n_steps:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Positions delta_k en fonction du spot S_k - {risk_label}", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_file, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Simulation des trajectoires...")
    S_train_2d = simulate_bs(args.n_train, args.steps, args.T, args.s0, args.sigma, seed=1)
    S_test_2d = simulate_bs(args.n_test, args.steps, args.T, args.s0, args.sigma, seed=2)
    S_train = S_train_2d[:, :, None]
    S_test = S_test_2d[:, :, None]
    Z_train = tf.maximum(S_train[:, -1, 0] - args.strike, 0.0)
    Z_test = tf.maximum(S_test[:, -1, 0] - args.strike, 0.0)

    bs_pnl, q0, bs_deltas = delta_hedge_pnl(
        S_test_2d,
        args.strike,
        args.sigma,
        args.T,
        transaction_cost=args.cost,
        return_deltas=True,
    )
    bs_pnl_np = as_numpy(bs_pnl)
    bs_deltas_np = as_numpy(bs_deltas)

    results = {
        "parameters": vars(args),
        "price_bs_q0": q0,
        "bs_delta_hedge": summary_stats(bs_pnl_np),
        "deep_hedging": {},
    }

    plot_n = min(args.plot_paths, args.n_test)
    S_plot = as_numpy(S_test_2d[:plot_n])
    bs_deltas_plot = bs_deltas_np[:plot_n]

    for cfg in RISK_CONFIGS:
        print(f"Entrainement {cfg['label']}...")
        hedger = DeepHedger(
            args.steps,
            d=1,
            risk=cfg["risk"],
            alpha=cfg["alpha"],
            lam=cfg["lam"],
            s0=args.s0,
        )
        history = train_hedger(
            hedger,
            S_train,
            Z_train,
            p0=q0,
            transaction_cost=args.cost,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            verbose=True,
        )

        deep_pnl, deep_deltas = hedger.compute_pnl(
            S_test,
            Z_test,
            p0=q0,
            transaction_cost=args.cost,
            training=False,
        )
        deep_pnl_np = as_numpy(deep_pnl)
        deep_deltas_np = as_numpy(deep_deltas[:, :, 0])

        np.save(out / f"pnl_bs_vs_{cfg['key']}_bs.npy", bs_pnl_np)
        np.save(out / f"pnl_bs_vs_{cfg['key']}_deep.npy", deep_pnl_np)
        np.save(out / f"deltas_{cfg['key']}_deep.npy", deep_deltas_np)

        make_histogram(
            bs_pnl_np,
            deep_pnl_np,
            cfg["label"],
            args.cost,
            out / f"hist_bs_vs_{cfg['key']}.png",
        )
        make_delta_spot_grid(
            S_plot,
            bs_deltas_plot,
            deep_deltas_np[:plot_n],
            cfg["label"],
            out / f"delta_vs_spot_bs_vs_{cfg['key']}.png",
        )

        results["deep_hedging"][cfg["key"]] = {
            "label": cfg["label"],
            "risk": cfg["risk"],
            "alpha": cfg["alpha"],
            "lambda": cfg["lam"],
            "stats": summary_stats(deep_pnl_np),
            "training_loss": history,
        }

    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Termine. Figures et statistiques dans: {out.resolve()}")


if __name__ == "__main__":
    main()
