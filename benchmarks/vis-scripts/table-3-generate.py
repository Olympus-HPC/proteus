import pandas as pd
import argparse
import numpy as np
import glob
import pathlib

TEXT_WIDTH = 506.295

benchmark_order = ["adam", "rsbench", "wsm5", "feynman-kac", "lulesh", "sw4ck"]


def main():
    parser = argparse.ArgumentParser(description="Print results")
    parser.add_argument(
        "--dir", help="path to directory containing result files", required=True
    )
    parser.add_argument(
        "--outdir", help="path to directory to store the .tex output", required=True
    )

    args = parser.parse_args()

    output = r"\resizebox{\columnwidth}{!}{" + "\n"
    output += r"\begin{tabular}{r" + "c" * len(benchmark_order) + "}\n"
    output += r"\toprule" + "\n"
    output += (
        r"\textbf{Machine} & \multicolumn{"
        + str(len(benchmark_order))
        + r"}{c}{\textbf{Program}} \\"
        + "\n"
    )
    output += r"\cmidrule(lr){2-7}" + "\n"

    for b in benchmark_order:
        if b == "feynman-kac":
            b = "FEY-KAC"
        else:
            b = b.upper()
        output += r"& \textbf{" + b + r"} "
    output += r"\\" + "\n"
    output += r"\midrule" + "\n"

    machines = ["nvidia", "amd"]
    for machine in machines:
        output += r"\textbf{" + machine.upper() + r"} "
        dfs = list()
        for fn in [
            f
            for f in glob.glob(f"{args.dir}/{machine}*-caching.csv")
            if "jitify" not in f
        ]:
            df = pd.read_csv(fn, index_col=0)
            found = False
            for sz in ["large", "mid", "small", "default"]:
                if sz in df.Input.unique():
                    df = df[df.Input == sz]
                    found = True
                    break
            assert (
                found
            ), f"In benchmark {df.Benchmark.unique()} we could not deduce input"
            dfs.append(df)

        df = pd.concat(dfs)
        # print("df\n", df)
        df = df[
            (df.StoredCache == True)
            & (df.Bounds == True)
            & (df.RuntimeConstprop == True)
        ]
        df = df[["Benchmark", "CacheSize"]]
        df = df.groupby(["Benchmark"]).mean().reset_index()

        for b in benchmark_order:
            values = df[df.Benchmark == b.lower()]["CacheSize"].values
            assert len(values) == 1, "Expected single CacheSize value"
            output += r"& " + f"{(values[0]/1024.0).round(1)}KB "
        output += r"\\" + "\n"

    output += r"\bottomrule " + "\n"
    output += r"\end{tabular}" + "\n"
    output += "}"
    print("=== TABULAR OUTPUT ===")
    print(output)
    print("=== END OF TABULAR OUTPUT ===")
    plot_dir = pathlib.Path(args.outdir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fn = f"{args.outdir}/table-3-maximal-cache-size.tex"
    print(f"Storing table in {fn} ...")
    with open(f"{fn}", "w") as f:
        f.write(output)


if __name__ == "__main__":
    main()
