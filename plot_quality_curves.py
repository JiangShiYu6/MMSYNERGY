import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def _plot_one_curve(df, x_col, y_col, y_std_col, title, xlabel, ylabel, out_fp):
    df = df.sort_values(x_col)
    x = df[x_col].values
    y = df[y_col].values
    yerr = df[y_std_col].fillna(0.0).values

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_fp, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser("Plot quality robustness curves from summary csv files")
    parser.add_argument("--edge-summary", type=str, required=True)
    parser.add_argument("--node-summary", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="output/quality_plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    edge_df = pd.read_csv(args.edge_summary)
    node_df = pd.read_csv(args.node_summary)

    _plot_one_curve(
        edge_df,
        x_col="edge_drop_rate",
        y_col="auc_mean",
        y_std_col="auc_std",
        title="Edge Removal Robustness: AUC",
        xlabel="Edge Drop Rate",
        ylabel="Test AUC",
        out_fp=os.path.join(args.out_dir, "edge_auc_curve.png"),
    )
    _plot_one_curve(
        edge_df,
        x_col="edge_drop_rate",
        y_col="ap_mean",
        y_std_col="ap_std",
        title="Edge Removal Robustness: AP",
        xlabel="Edge Drop Rate",
        ylabel="Test AP",
        out_fp=os.path.join(args.out_dir, "edge_ap_curve.png"),
    )

    _plot_one_curve(
        node_df,
        x_col="node_drop_rate",
        y_col="auc_mean",
        y_std_col="auc_std",
        title="Node Removal Robustness: AUC",
        xlabel="Node Drop Rate",
        ylabel="Test AUC",
        out_fp=os.path.join(args.out_dir, "node_auc_curve.png"),
    )
    _plot_one_curve(
        node_df,
        x_col="node_drop_rate",
        y_col="ap_mean",
        y_std_col="ap_std",
        title="Node Removal Robustness: AP",
        xlabel="Node Drop Rate",
        ylabel="Test AP",
        out_fp=os.path.join(args.out_dir, "node_ap_curve.png"),
    )

    print("Saved plots to:")
    print(os.path.join(args.out_dir, "edge_auc_curve.png"))
    print(os.path.join(args.out_dir, "edge_ap_curve.png"))
    print(os.path.join(args.out_dir, "node_auc_curve.png"))
    print(os.path.join(args.out_dir, "node_ap_curve.png"))


if __name__ == "__main__":
    main()
