import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pathlib

TEXT_WIDTH = 505.89

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 12,
    "font.size": 12,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
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


def prune_columns(df, profiler):
    if profiler == "rocprof":
        to_drop = [
            "gpu-id",
            "queue-id",
            "queue-index",
            "pid",
            "tid",
            "wave_size",
            "sig",
            "obj",
            "DispatchNs",
            "BeginNs",
            "EndNs",
            "CompleteNs",
            "ExeSize",
            "Hash",
            "RunIndex",
        ]
    elif profiler == "nvprof":
        to_drop = [
            "flop_hp_efficiency",
            "flop_sp_efficiency",
            "flop_dp_efficiency",
            "sm_efficiency",
            "tex_cache_hit_rate",
            "nvlink_overhead_data_received",
            "shared_utilization",
            "tensor_precision_fu_utilization",
            "sysmem_read_utilization",
            "sysmem_write_utilization",
            "sysmem_utilization",
            "dram_utilization",
            "l2_utilization",
            "tex_utilization",
            "ldst_fu_utilization",
            "cf_fu_utilization",
            "tex_fu_utilization",
            "special_fu_utilization",
            "half_precision_fu_utilization",
            "single_precision_fu_utilization",
            "double_precision_fu_utilization",
            "Device",
            "Context",
            "Stream",
            "Correlation_ID",
            "ExeSize",
            "RunIndex",
        ]
        df = df.rename(columns={"Kernel": "Name"})
    else:
        raise Exception("Expected profiler rocprof or nvprof")

    df["Name"] = df["Name"].apply(lambda x: x.split("(")[0])
    df = df.drop(columns=to_drop, errors="ignore")
    # NOTE: dropna may be too eager to drop rows, especially for nvprof which
    # creates numerous columns that are not applicable to every entry.
    df = df.dropna()
    return df


def assign_label(row):
    if row["Compile"] == "aot":
        return "AOT"

    rename_map = {
        True: {True: "LB+RCF", False: "LB"},
        False: {True: "RCF", False: "None"},
    }

    return rename_map[row["Bounds"]][row["RuntimeConstprop"]]


def analyze(df, bench, profiler, counter):
    plot_dir = pathlib.Path("plots/")
    plot_dir.mkdir(parents=True, exist_ok=True)
    df = df[(df.Benchmark == bench)]
    df = prune_columns(df, profiler)
    df = (
        df.groupby(
            [
                "Name",
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
    df.drop(df[df.StoredCache == True].index, inplace=True)
    df.Name = df.Name.str[:8]

    label_columns = [
        "Compile",
        "StoredCache",
        "Bounds",
        "RuntimeConstprop",
    ]

    df["label"] = df.apply(assign_label, axis=1)

    df = df.drop(columns=label_columns)
    df = df.drop(columns=["Ctime", "ExeTime", "repeat"])

    inputs = df.Input.unique()
    largest = "large"
    for i in ["large", "mid", "small", "default"]:
        if i in inputs:
            largest = i
            break
    df = df[df.Input == largest]

    id_vars = ["Name", "Input", "Benchmark", "label", counter]
    df = df[id_vars]
    return df


def visualize(df, bench, profiler, prefix, machine, counter, kernel=None):
    plot_dir = pathlib.Path("plots/")
    plot_dir.mkdir(parents=True, exist_ok=True)
    df = df[(df.Benchmark == bench)]

    bar_order = [
        "AOT",
        "None",
        "LB",
        "RCF",
        "LB+RCF",
    ]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    sizes = set_size(TEXT_WIDTH, 0.5)
    fig, ax = plt.subplots(figsize=sizes)
    df = df.set_index("label")
    df = df.loc[bar_order]
    ax.barh(bar_order, df[counter], color=colors[: len(bar_order)])

    ax.set_xlabel(f"{counter}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    plt.tight_layout()

    if kernel is None:
        fn = f"{plot_dir}/{prefix}-barh-kernel-metric-{machine}-{df['Benchmark'].unique()[0]}-{counter.lower()}.pdf"
    else:
        fn = f"{plot_dir}/{prefix}-barh-kernel-metric-{machine}-{df['Benchmark'].unique()[0]}-{kernel}-{counter.lower()}.pdf"

    print(f"Storing to {fn}")
    fig.savefig(fn)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Print results")
    parser.add_argument("--file", help="path to results csv file", required=True)

    parser.add_argument(
        "-p",
        "--prefix",
        help="prefix to prepend on the output file",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--machine",
        help="which machine to run on: amd|nvidia",
        choices=("amd", "nvidia"),
        required=True,
    )

    parser.add_argument(
        "-c", "--counter", help="Which performance counter to plot", required=True
    )

    args = parser.parse_args()
    df = pd.read_csv(args.file, index_col=0, dtype={"Hash": "str"})

    if args.machine == "amd":
        profiler = "rocprof"
    elif args.machine == "nvidia":
        profiler = "nvprof"
    else:
        raise Exception("Invalid machine type")

    for bench in df.Benchmark.unique():
        df = analyze(df, bench, profiler, args.counter)
        if len(df.Name.unique()) > 1:
            for kernel, kdf in df.groupby("Name"):
                visualize(
                    kdf,
                    bench,
                    profiler,
                    args.prefix,
                    args.machine,
                    args.counter,
                    kernel,
                )
        else:
            visualize(df, bench, profiler, args.prefix, args.machine, args.counter)


if __name__ == "__main__":
    main()
