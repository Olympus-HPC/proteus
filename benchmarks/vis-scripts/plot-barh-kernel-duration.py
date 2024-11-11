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
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 5,
    "xtick.labelsize": 10,
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


def prune_columns(df, profiler):
    if profiler == "rocprof":
        to_drop = [
            "gpu-id",
            "queue-id",
            "queue-index",
            "pid",
            "tid",
            "grd",
            "wgr",
            "lds",
            "scr",
            "arch_vgpr",
            "accum_vgpr",
            "sgpr",
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
            "Start",
            "SrcMemType",
            "DstMemType",
            "Device",
            "Context",
            "Stream",
            "Correlation_ID",
            "Throughput",
            "Grid X",
            "Grid Y",
            "Grid Z",
            "Block X",
            "Block Y",
            "Block Z",
            "Registers Per Thread",
            "Static SMem",
            "Dynamic SMem",
            "Size",
            "Hash",
            "ExeSize",
            "RunIndex",
        ]
    else:
        raise Exception("Expected profiler rocprof or nvprof")

    df["Name"] = df["Name"].apply(lambda x: x.split("(")[0])
    df = df.drop(columns=to_drop)
    # NOTE: dropna may be too eager to drop rows, especially for nvprof which
    # creates numerous columns that are not applicable to every entry.
    df = df.dropna()
    return df


def rename_lulesh(row):
    if "CalcVolumeForceForElems_kernel" in row.Name:
        return "VolumeForce"
    elif "CalcKinematicsAndMonotonicQGradient_kernel" in row.Name:
        return "KinematicQGradient"
    return "None"


def assign_label(row):
    if row["Compile"] == "aot":
        return "AOT"

    rename_map = {
        True: {True: "LB+RCF", False: "LB"},
        False: {True: "RCF", False: "None"},
    }

    return rename_map[row["Bounds"]][row["RuntimeConstprop"]]


def visualize(df, bench, profiler, prefix, machine):
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
                "repeat",
            ]
        )
        .mean()
        .reset_index()
    )

    label_columns = [
        "Compile",
        "StoredCache",
        "Bounds",
        "RuntimeConstprop",
    ]

    df["label"] = df.apply(assign_label, axis=1)

    if bench.upper() == "LULESH":
        df = df[
            df.Name.str.contains("CalcVolumeForceForElems_kernel")
            | df.Name.str.contains("CalcKinematicsAndMonotonicQGradient_kernel")
        ]
        df["Name"] = df.apply(rename_lulesh, axis=1)

    # NOTE: I am NOT dropping here the "StoredCache", for us to investigate more whether
    # there are secondary effects. However, those are likely to be 'measuring bugs' and not actual
    # contributions. We will need to diagnose them. When those are fixed, we need to
    # merge those measurements as well (or drop the rows).

    df = df.drop(columns=label_columns)

    indexes = np.arange(len(df.Name.unique()))

    bar_order = [
        "AOT",
        "None",
        "LB",
        "RCF",
        "LB+RCF",
    ]

    df["Duration"] = df["Duration"] / 10e9

    # Here we need to get mean/std
    df = (
        df.groupby(
            [
                "Name",
                "Input",
                "Benchmark",
                "label",
            ]
        )
        .agg({"Duration": ["mean", "std"], "Speedup": ["mean", "std"]})
        .reset_index()
    )

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    for kname, per_kernel_df in df.groupby("Name"):
        for key, tmp_df in per_kernel_df.groupby(["Input"]):
            if not isinstance(key, str):
                key = key[0]

            # Setting to 0.25 Cause I am planing to add 2 plots in a single column
            sizes = set_size(TEXT_WIDTH, 0.5)
            fig, ax = plt.subplots(figsize=sizes)
            tmp_df = tmp_df.set_index("label")
            tmp_df = tmp_df.loc[bar_order]
            ax.barh(
                bar_order, tmp_df["Duration"]["mean"], color=colors[: len(bar_order)]
            )

            ax.set_xlabel("Kernel Duration (s)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            plt.tight_layout()

            if len(df.Name.unique()) > 1:
                fn = f"{plot_dir}/{prefix}-barh-kernel-duration-{machine}-{tmp_df['Benchmark'].unique()[0]}-{kname}-{key}.pdf"
            else:
                fn = f"{plot_dir}/{prefix}-barh-kernel-duration-{machine}-{tmp_df['Benchmark'].unique()[0]}-{key}.pdf"
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

    args = parser.parse_args()
    df = pd.read_csv(args.file, index_col=0, dtype={"Hash": "str"})

    if args.machine == "amd":
        profiler = "rocprof"
    elif args.machine == "nvidia":
        profiler = "nvprof"
    else:
        raise Exception("Invalid machine type")

    for bench in df.Benchmark.unique():
        visualize(df, bench, profiler, args.prefix, args.machine)


if __name__ == "__main__":
    main()
