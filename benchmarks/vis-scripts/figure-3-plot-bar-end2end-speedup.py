import pandas as pd
import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pathlib
import glob

TEXT_WIDTH = 506.295

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 10,
}


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.
    Parameters
    ----------
    width: float
    Document textwidth or columnwidth in pts
    fraction: float, optional
    Fraction of the width which you wish the figure to occupy
    Returns
    -------
    fig_dim: tuple
    Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio
    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim


plt.rcParams.update(tex_fonts)


def assign_label(row):
    if row["Compile"] == "aot":
        return "AOT"

    if row["Compile"] == "jitify":
        return "Jitify"

    if not row["Bounds"]:
        return "DROP"

    if not row["RuntimeConstprop"]:
        return "DROP"

    if row["StoredCache"]:
        return "Proteus+\$"

    return "Proteus"


def visualize(df, machine, plot_dir):
    plot_dir = pathlib.Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    std_df = (
        df.groupby(
            [
                "Input",
                "Benchmark",
                "Compile",
                "StoredCache",
                "Bounds",
                "RuntimeConstprop",
            ]
        )
        .std()
        .reset_index()
    )
    df = (
        df.groupby(
            [
                "Input",
                "Benchmark",
                "Compile",
                "StoredCache",
                "Bounds",
                "RuntimeConstprop",
            ]
        )
        .mean()
        .reset_index()
    )

    drop_columns = ["Compile", "StoredCache", "Bounds", "RuntimeConstprop", "Input"]

    # Keep only the data we want
    df["label"] = df.apply(assign_label, axis=1)
    df = df[df.label != "DROP"]
    std_df["label"] = std_df.apply(assign_label, axis=1)
    std_df = std_df[std_df.label != "DROP"]
    # end

    df = df.drop(columns=drop_columns)
    std_df = std_df.drop(columns=drop_columns)

    bar_order = ["Proteus", "Proteus+\$", "Jitify"]
    if machine == "amd":
        bar_order.remove("Jitify")

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    color = colors[:8]
    c_map = {
        "AOT": color[0],
        "Proteus": color[1],
        "Proteus+\$": color[2],
        "Jitify": color[7],
    }

    rename_bench = {
        "adam": "ADAM",
        "rsbench": "RSBENCH",
        "wsm5": "WSM5",
        "feynman-kac": "FEY-KAC",
        "lulesh": "LULESH",
        "sw4ck": "SW4CK",
    }

    df = df.replace({"Benchmark": rename_bench})
    std_df = std_df.replace({"Benchmark": rename_bench})

    tmp_df = (
        df.groupby(["Benchmark", "label"]).mean()[["ExeTime", "Speedup"]].reset_index()
    )
    elements = []

    for g, tmp in tmp_df.groupby("Benchmark"):
        if "Jitify" not in tmp.label.unique():
            elements.append(
                {
                    "Benchmark": g,
                    "label": "Jitify",
                    "ExecTime": np.nan,
                    "Speedup": np.nan,
                }
            )
    for e in elements:
        tmp_df.loc[len(tmp_df)] = e

    benchmarks_ordered = ["ADAM", "RSBENCH", "WSM5", "FEY-KAC", "LULESH", "SW4CK"]
    tmp_df = tmp_df.set_index("Benchmark")
    tmp_df = tmp_df.loc[benchmarks_ordered]
    tmp_df = tmp_df.reset_index()

    sizes = set_size(TEXT_WIDTH, 0.5)
    fig, ax = plt.subplots(figsize=sizes)
    bar_width = 0.3
    offset = 0
    spread = bar_width * (len(bar_order) + 1)
    ind = np.arange(0, spread * len(tmp_df.Benchmark.unique()), spread)
    for i, bar in enumerate(bar_order):
        rect = ax.bar(
            ind + offset,
            tmp_df[tmp_df.label == bar]["Speedup"],
            bar_width,
            color=c_map[bar],
            label=bar,
        )
        # NOTE: For whatever reason fmt (even when passed with %) does not format
        # to 2 digits. Rounding now explicitly through pandas functionality
        if bar != "AOT":
            bar_labels = ["%.2f" % v for v in tmp_df[tmp_df.label == bar]["Speedup"]]
            if bar == "Jitify":
                bar_labels[4] = "N/A"

            ax.bar_label(
                rect,
                fmt="%g",
                labels=bar_labels,
                padding=0.5,
                fontsize=8,
                rotation=90,
            )

        offset += bar_width
    # Add an "X" to the missing LULESH data point.
    if machine == "nvidia":
        ax.scatter(
            4 * spread + 2 * bar_width + 0.01, 0.1, marker="x", color=c_map["Jitify"]
        )
    ax.set_ylabel("Speedup over AOT")
    ax.yaxis.set_major_formatter("{x:.1f}")
    ax.set_xticks(ind + bar_width * (len(bar_order) - 1) / 2)
    ax.set_xticklabels(tmp_df.Benchmark.unique())
    yticks = ax.get_yticks()
    ax.set_ylim((yticks.min(), yticks.max()))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.xaxis.set_ticks_position('none')

    plt.xticks(rotation=15)
    plt.tight_layout()
    if machine == "nvidia":
        ax.legend(
            ncol=1,
            # loc="upper right",
            bbox_to_anchor=(0.58, 0.6),
            handlelength=1.0,
            handletextpad=0.5,
            # loc="upper right",
            fancybox=False,
            shadow=False,
            frameon=False,
        )
    fn = "{0}/figure-3-bar-end2end-speedup-{1}.pdf".format(plot_dir, machine)
    print(f"Storing to {fn}")
    fig.savefig(fn, bbox_inches="tight")
    plt.close(fig)


def compute_speedup(results):
    results.ExeTime.astype(float)
    for input_id in results.Input.unique():
        for repeat in results.repeat.unique():
            base = results[
                (results.Compile == "aot")
                & (results.Input == input_id)
                & (results.repeat == repeat)
            ].copy(True)

            # Input is unique for the same benchmark and input, rdiv to divide
            # the base (AOT) execution time with the JIT execution time.
            results.loc[
                (
                    (results.Input == input_id)
                    & (results.repeat == repeat)
                    & (results.Compile == "jitify"),
                    "Speedup",
                )
            ] = results.loc[
                (
                    (results.Input == input_id)
                    & (results.repeat == repeat)
                    & (results.Compile == "jitify")
                )
            ].ExeTime.rdiv(
                base.set_index("repeat").ExeTime
            )
    return results


def main():
    parser = argparse.ArgumentParser(description="Print results")
    parser.add_argument(
        "--dir", help="path to directory containing result files", required=True
    )

    parser.add_argument("--plot-dir", help="directory to store plots in", required=True)

    parser.add_argument(
        "-m",
        "--machine",
        help="which machine to run on: amd|nvidia",
        choices=("amd", "nvidia"),
        required=True,
    )

    args = parser.parse_args()
    dfs = list()
    for fn in glob.glob(f"{args.dir}/{args.machine}*-results.csv"):
        df = pd.read_csv(fn, index_col=0)
        found = False
        for sz in ["large", "mid", "small", "default"]:
            if sz in df.Input.unique():
                df = df[df.Input == sz]
                found = True
                break
        assert found, f"In benchmark {df.Benchmark.unique()} we could not deduce input"
        dfs.append(df)

    df = pd.concat(dfs)

    for b in df["Benchmark"].unique():
        bench_results = df[df["Benchmark"] == b]
        if "jitify" in bench_results["Compile"].unique():
            df.loc[df.Benchmark == b] = compute_speedup(bench_results)

    visualize(df, args.machine, args.plot_dir)


if __name__ == "__main__":
    main()
