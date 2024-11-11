import pandas as pd
import argparse
import numpy as np
import glob
import pathlib

TEXT_WIDTH = 506.295

benchmark_order = ["ADAM", "RSBENCH", "WSM5", "FEY-KAC", "LULESH", "SW4CK"]


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


def generate_table(df, machine):
    in_df = df
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

    sem_df = (
        in_df.groupby(
            [
                "Input",
                "Benchmark",
                "Compile",
                "StoredCache",
                "Bounds",
                "RuntimeConstprop",
            ]
        )
        .sem()
        .reset_index()
    )

    sem_df.ExeTime = 100.0 * sem_df.ExeTime
    df["RelSem"] = sem_df.ExeTime / df.ExeTime

    drop_columns = ["Compile", "StoredCache", "Bounds", "RuntimeConstprop", "Input"]

    df["label"] = df.apply(assign_label, axis=1)
    df = df[df.label != "DROP"]

    df = df.drop(columns=drop_columns)

    row_order = ["Proteus", "Proteus+\$", "Jitify"]
    if machine == "amd":
        row_order.remove("Jitify")

    rename_bench = {
        "adam": "ADAM",
        "rsbench": "RSBENCH",
        "wsm5": "WSM5",
        "feynman-kac": "FEY-KAC",
        "lulesh": "LULESH",
        "sw4ck": "SW4CK",
    }

    df = df.replace({"Benchmark": rename_bench})

    tmp_df = (
        df.groupby(["Benchmark", "label"])
        .mean()[["ExeTime", "RelSem", "Speedup"]]
        .reset_index()
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

    # print("==== DATAFRAME ===")
    # print(type(tmp_df.columns))
    tmp_df = tmp_df.set_index("Benchmark")
    tmp_df = tmp_df.loc[benchmark_order]
    tmp_df = tmp_df.reset_index()
    # print(tmp_df)
    # print("==== END DATAFRAME ===")

    output = (
        " & "
        + " & ".join([r"\textbf{" + b + r"}" for b in benchmark_order])
        + r" \\"
        + "\n"
    )
    output += r"\midrule" + "\n"
    exe_times = {}
    rel_sem = {}
    for method in ["AOT"] + row_order:
        exe_times[method] = [
            (lambda v: "N/A" if str(v) == "nan" else str(v))(v)
            for v in tmp_df[tmp_df.label == method]["ExeTime"].values.round(2)
        ]
        rel_sem[method] = [
            (lambda v: "N/A" if str(v) == "nan" else str(v))(v)
            for v in tmp_df[tmp_df.label == method]["RelSem"].values.round(2)
        ]

    def pos_argmin(i, d):
        mink = None
        minv = np.inf
        for k, v in d.items():
            if v[i] == "N/A":
                continue
            if float(v[i]) < minv:
                mink = k
                minv = float(v[i])

        return mink

    for i in range(len(benchmark_order)):
        k = pos_argmin(i, exe_times)
        exe_times[k][i] = r"\cellcolor{yellow!60}" + exe_times[k][i] + r""

    for method in ["AOT"] + row_order:
        output += (
            r"\textbf{"
            + f"{method}"
            + r"}"
            + " & "
            + " & ".join(
                [f"{x} $\pm {y}\%$" for x, y in zip(exe_times[method], rel_sem[method])]
            )
            + r" \\"
            + "\n"
        )

    return output


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
    parser.add_argument(
        "--outdir", help="path to directory to store the .tex output", required=True
    )

    args = parser.parse_args()

    output = r"\resizebox{\columnwidth}{!}{" + "\n"
    output += r"\begin{tabular}{" + "c" * (len(benchmark_order) + 1) + "}\n"
    output += r"\toprule" + "\n"

    machines = ["amd", "nvidia"]
    for machine in machines:
        output += (
            r"\textbf{"
            + f"{machine.upper()}"
            + r"}"
            + r" & \multicolumn{"
            + f"{len(benchmark_order)}"
            + r"}{c}{\textbf{End-to-end Execution Time (s)}} \\"
            + "\n"
        )
        output += r"\cmidrule{2-7}" + "\n"
        dfs = list()
        for fn in glob.glob(f"{args.dir}/{machine}*-results.csv"):
            # Skip cases of nvprof/rocprof files
            if "profiler" in fn:
                continue
            # SKip the performance metric files
            if "kernel" in fn:
                continue

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

        for b in df["Benchmark"].unique():
            bench_results = df[df["Benchmark"] == b]
            if "jitify" in bench_results["Compile"].unique():
                df.loc[df.Benchmark == b] = compute_speedup(bench_results)

        output += generate_table(df, machine)
        if machine != machines[-1]:
            output += r"\midrule" + "\n"

    output += r"\bottomrule " + "\n"
    output += r"\end{tabular}" + "\n"
    output += "}"
    print("=== TABULAR OUTPUT ===")
    print(output)
    print("=== END OF TABULAR OUTPUT ===")
    plot_dir = pathlib.Path(args.outdir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fn = f"{args.outdir}/table-2-exe-times.tex"
    print(f"Storing table in {fn} ...")
    with open(f"{fn}", "w") as f:
        f.write(output)


if __name__ == "__main__":
    main()
