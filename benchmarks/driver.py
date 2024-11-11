import argparse
import pandas as pd
from pathlib import Path
import pathlib
import subprocess
import json
import os
import cxxfilt
import time
import re

NUM_REPEATS = 3

env_configs = {
    # Setting same env vars used in jit for aot, jitify to simplify dataframe
    # creation.
    "aot": [
        {
            "ENV_PROTEUS_USE_STORED_CACHE": "0",
            "ENV_PROTEUS_SET_LAUNCH_BOUNDS": "0",
            "ENV_PROTEUS_SPECIALIZE_ARGS": "0",
        },
    ],
    "jitify": [
        {
            "ENV_PROTEUS_USE_STORED_CACHE": "0",
            "ENV_PROTEUS_SET_LAUNCH_BOUNDS": "0",
            "ENV_PROTEUS_SPECIALIZE_ARGS": "0",
        },
    ],
    "jit": [
        # Without using stored cache.
        {
            "ENV_PROTEUS_USE_STORED_CACHE": "0",
            "ENV_PROTEUS_SET_LAUNCH_BOUNDS": "0",
            "ENV_PROTEUS_SPECIALIZE_ARGS": "0",
        },
        {
            "ENV_PROTEUS_USE_STORED_CACHE": "0",
            "ENV_PROTEUS_SET_LAUNCH_BOUNDS": "0",
            "ENV_PROTEUS_SPECIALIZE_ARGS": "1",
        },
        {
            "ENV_PROTEUS_USE_STORED_CACHE": "0",
            "ENV_PROTEUS_SET_LAUNCH_BOUNDS": "1",
            "ENV_PROTEUS_SPECIALIZE_ARGS": "0",
        },
        {
            "ENV_PROTEUS_USE_STORED_CACHE": "0",
            "ENV_PROTEUS_SET_LAUNCH_BOUNDS": "1",
            "ENV_PROTEUS_SPECIALIZE_ARGS": "1",
        },
        # Using stored cache.
        # CAUTION: We need to create the cache jit binaries before running.
        # Especially, JIT launch bounds, runtime constprop will be baked into
        # the binary so we need a "warmup" run for each setting before taking
        # the measurement.
        {
            "ENV_PROTEUS_USE_STORED_CACHE": "1",
            "ENV_PROTEUS_SET_LAUNCH_BOUNDS": "0",
            "ENV_PROTEUS_SPECIALIZE_ARGS": "0",
        },
        {
            "ENV_PROTEUS_USE_STORED_CACHE": "1",
            "ENV_PROTEUS_SET_LAUNCH_BOUNDS": "0",
            "ENV_PROTEUS_SPECIALIZE_ARGS": "1",
        },
        {
            "ENV_PROTEUS_USE_STORED_CACHE": "1",
            "ENV_PROTEUS_SET_LAUNCH_BOUNDS": "1",
            "ENV_PROTEUS_SPECIALIZE_ARGS": "0",
        },
        {
            "ENV_PROTEUS_USE_STORED_CACHE": "1",
            "ENV_PROTEUS_SET_LAUNCH_BOUNDS": "1",
            "ENV_PROTEUS_SPECIALIZE_ARGS": "1",
        },
    ],
}


class rocprof:
    def __init__(self, metrics, cwd):
        self.metrics = metrics
        if metrics:
            metrics_file = f"{cwd}/vis-scripts/rocprof-metrics.txt"
            self.command = f"rocprof -i {metrics_file}" + " --timestamp on -o {0} {1}"
        else:
            self.command = "rocprof --timestamp on -o {0} {1}"

    def get_command(self, output, executable):
        return self.command.format(output, executable)

    def parse(self, fn):
        def get_hash(x):
            try:
                hash_pos = 2
                return cxxfilt.demangle(x.split("$")[hash_pos])
            except IndexError:
                return None

        df = pd.read_csv(fn, sep=",")
        # Rename to match output between rocprof, nvprof.
        df.rename(columns={"KernelName": "Name", "Index": "RunIndex"}, inplace=True)
        df["Duration"] = df["EndNs"] - df["BeginNs"]
        df["Name"] = df["Name"].str.replace(" [clone .kd]", "", regex=False)
        df["Hash"] = df.Name.apply(lambda x: get_hash(x))
        df["Name"] = df.Name.apply(lambda x: cxxfilt.demangle(x.split("$")[0]))
        return df


class nvprof:
    def __init__(self, metrics):
        if metrics:
            self.command = "nvprof --metrics inst_per_warp,stall_exec_dependency --print-gpu-trace --normalized-time-unit ns --csv --log-file {0} {1}"
        else:
            self.command = "nvprof --print-gpu-trace --normalized-time-unit ns --csv --log-file {0} {1}"
        self.metrics = metrics

    def get_command(self, output, executable):
        return self.command.format(output, executable)

    def parse(self, fn):
        def get_hash(x):
            try:
                hash_pos = 2
                return cxxfilt.demangle(x.split("$")[hash_pos])
            except IndexError:
                return None

        # Skip the first 3 (or 4 lines if metrics are collected) of nvprof
        # metadata info.
        skiprows = 4 if self.metrics else 3
        df = pd.read_csv(fn, sep=",", skiprows=skiprows)
        # Skip the first row after the header which contains units of metrics.
        df = df[1:]
        # Nvprof with metrics tracks only kernels.
        if self.metrics:
            df["Kernel"] = df.Kernel.apply(lambda x: cxxfilt.demangle(x.split("$")[0]))
            df.rename(columns={"Kernel": "Name"}, inplace=True)
        else:
            df["Hash"] = df.Name.apply(lambda x: get_hash(x))
            df["Name"] = df.Name.apply(lambda x: cxxfilt.demangle(x.split("$")[0]))

        return df


def execute_command(cmd, **kwargs):
    print("=> Execute", cmd)
    try:
        p = subprocess.run(cmd, check=True, text=True, **kwargs)
    except subprocess.CalledProcessError as e:
        print("Failed cmd", e.cmd)
        print("ret", e.returncode)
        print("stdout\n", e.stdout)
        print("stderr\n", e.stderr)
        print(e)
        raise e

    if "capture_output" in kwargs and kwargs["capture_output"]:
        return p.stdout, p.stderr

    return None, None


class Configuration:
    def __init__(self, path, cc, jit_path, config):
        self.path = path
        self.cc = cc
        self.jit_path = jit_path
        self.config = config

    def __str__(self):
        return f"{self.config} {self.path}"

    def clean(self):
        os.chdir(self.path)
        cmd = "make clean"
        out, err = execute_command(cmd, capture_output=True, shell=True)
        print("=========== stdout ===========")
        print(out)
        print("==============================")
        print("=========== stderr ===========")
        print(err)
        print("==============================")

    def build(self, do_jit):
        os.chdir(self.path)
        cmd = "make"
        env = os.environ.copy()
        env["JIT"] = "yes" if do_jit else "no"
        env["JIT_PATH"] = self.jit_path
        env["CC"] = self.cc
        t1 = time.perf_counter()
        print(
            "Build command",
            cmd,
            "CC=" + env["CC"],
            "JIT_PATH=" + env["JIT_PATH"],
            "JIT=" + env["JIT"],
        )
        out, err = execute_command(cmd, env=env, capture_output=True, shell=True)
        t2 = time.perf_counter()
        print("=========== stdout ===========")
        print(out)
        print("==============================")
        print("=========== stderr ===========")
        print(err)
        print("==============================")
        return t2 - t1

    def build_and_run(self, profiler):
        os.chdir(self.path)

        results = pd.DataFrame()
        caching = pd.DataFrame()
        for btype, exe in self.config["executables"].items():
            self.clean()
            print("BUILD", self.path, "type", btype)
            assert (
                btype == "aot" or btype == "jit" or btype == "jitify"
            ), "Expected aot or jit or jitify for btype"
            ctime = self.build(btype != "aot")
            exe_size = Path(f"{self.path}/{exe}").stat().st_size
            print("=> BUILT")

            for repeat in range(0, NUM_REPEATS):
                for input_id, args in self.config["inputs"].items():
                    for env in env_configs[btype]:
                        cmd_env = os.environ.copy()
                        for k, v in env.items():
                            cmd_env[k] = v
                        cmd = f"./{exe} {args}"

                        set_launch_bounds = (
                            False
                            if env["ENV_PROTEUS_SET_LAUNCH_BOUNDS"] == "0"
                            else True
                        )
                        use_stored_cache = (
                            False
                            if env["ENV_PROTEUS_USE_STORED_CACHE"] == "0"
                            else True
                        )
                        specialize_args = (
                            False if env["ENV_PROTEUS_SPECIALIZE_ARGS"] == "0" else True
                        )

                        if btype == "jit":
                            print("Proteus JIT env", env)

                        if profiler is not None:
                            if profiler.metrics:
                                # Skip profiler with metrics (GPU counters) runs for
                                # jitify, as we don't use them.
                                if btype == "jitify":
                                    continue
                                # Skip measuring metrics (GPU counters) for the
                                # proteus jit stored cache variation since they are
                                # identical to the non-cached runs.
                                if btype == "jit" and use_stored_cache == True:
                                    continue

                        # Delete any previous generated JIT stored cache.
                        if use_stored_cache:
                            # Delete amy previous cache files in the command path.
                            for file in Path(self.path).glob(".proteus/cache-jit-*"):
                                file.unlink()

                        # Early on execute always a warmup run. If using the
                        # stored cache, this run will generate the files for a
                        # warm cache.
                        execute_command(
                            cmd,
                            env=cmd_env,
                            capture_output=True,
                            shell=True,
                            cwd=str(self.path),
                        )

                        stats = f"{os.getcwd()}/{btype}_{input_id}.csv"
                        if profiler:
                            # Execute with profiler on.
                            cmd = profiler.get_command(stats, cmd)

                        t1 = time.perf_counter()
                        out, err = execute_command(
                            cmd,
                            env=cmd_env,
                            capture_output=True,
                            shell=True,
                            cwd=str(self.path),
                        )
                        t2 = time.perf_counter()

                        # Cleanup from a stored cache run, removing cache files.
                        cache_size = 0
                        if use_stored_cache:
                            for file in Path(self.path).glob(".proteus/cache-jit-*.o"):
                                # Size in bytes.
                                cache_size += file.stat().st_size
                            # Delete amy previous cache files in the command path.
                            for file in Path(self.path).glob(".proteus/cache-jit-*"):
                                file.unlink()

                        print("=========Error=================")
                        print(err)
                        print("=========== stdout ===========")
                        print(out)
                        print("==============================")

                        if profiler:
                            df = profiler.parse(stats)
                            # Add new columns to the existing dataframe from the
                            # profiler.
                            df["Benchmark"] = self.config["name"]
                            df["Input"] = input_id
                            df["Compile"] = btype
                            df["Ctime"] = ctime
                            df["StoredCache"] = use_stored_cache
                            df["Bounds"] = set_launch_bounds
                            df["RuntimeConstprop"] = specialize_args
                            df["ExeSize"] = exe_size
                            df["ExeTime"] = t2 - t1
                            # Drop memcpy operations (because JIT adds DtoH copies
                            # to read kernel bitcodes that interfere with unique
                            # indexing and add RunIndex for nvprof to uniquely
                            # identify kernel invocations.
                            if isinstance(profiler, nvprof):
                                df.drop(
                                    df[df.Name.str.contains("CUDA memcpy")].index,
                                    inplace=True,
                                )
                                # Reset index to sequential, integer index.
                                df.reset_index(drop=True, inplace=True)
                                df["RunIndex"] = df.index
                        else:
                            # Create a new dataframe row.
                            df = pd.DataFrame(
                                {
                                    "Benchmark": [self.config["name"]],
                                    "Input": [input_id],
                                    "Compile": [btype],
                                    "Ctime": [ctime],
                                    "StoredCache": [use_stored_cache],
                                    "Bounds": [set_launch_bounds],
                                    "RuntimeConstprop": [specialize_args],
                                    "ExeSize": [exe_size],
                                    "ExeTime": [t2 - t1],
                                }
                            )
                        df["repeat"] = repeat
                        results = pd.concat((results, df), ignore_index=True)

                        # Skip parsing caching stats when running AOT.
                        if btype != "jit":
                            continue

                        # Parse JIT caching info.
                        matches = re.findall(
                            "HashValue ([0-9]+) NumExecs ([0-9]+) NumHits ([0-9]+)",
                            out,
                        )
                        cache_df = pd.DataFrame(
                            {
                                "HashValue": [str(m[0]) for m in matches],
                                "NumExecs": [int(m[1]) for m in matches],
                                "NumHits": [int(m[2]) for m in matches],
                            }
                        )
                        cache_df["Benchmark"] = self.config["name"]
                        cache_df["Input"] = input_id
                        cache_df["StoredCache"] = use_stored_cache
                        cache_df["Bounds"] = set_launch_bounds
                        cache_df["RuntimeConstprop"] = specialize_args
                        cache_df["repeat"] = repeat
                        cache_df["CacheSize"] = cache_size

                        caching = pd.concat((caching, cache_df))

        return results, caching


def postprocess_profiler(results):
    print("POSTPROCESS PROFILER")
    assert (
        len(results.Benchmark.unique()) == 1
    ), "Postprocessing expects a dataframe of a single benchmark"

    # Make sure Duration is a float.
    results.Duration = results.Duration.astype(float)
    for input_id in results.Input.unique():
        for repeat in results.repeat.unique():
            base = results[
                (results.Compile == "aot")
                & (results.Input == input_id)
                & (results.repeat == repeat)
            ]

            # RunIndex is unique for the same benchmark and input, rdiv to divide
            # the base (AOT) duration with the JIT duration.
            results.loc[
                (results.Input == input_id) & (results.repeat == repeat),
                "Speedup",
            ] = results.Duration.rdiv(
                results["RunIndex"].map(base.set_index("RunIndex").Duration)
            )

    return results


def postprocess(results):
    print("POSTPROCESS")
    assert (
        len(results.Benchmark.unique()) == 1
    ), "Postprocessing expects a dataframe of a single benchmark"

    # Make sure ExeTime is a float.
    results.ExeTime = results.ExeTime.astype(float)
    for input_id in results.Input.unique():
        for repeat in results.repeat.unique():
            base = results[
                (results.Compile == "aot")
                & (results.Input == input_id)
                & (results.repeat == repeat)
            ]
            # Input is unique for the same benchmark and input, rdiv to divide
            # the base (AOT) execution time with the JIT execution time.
            results.loc[
                (results.Input == input_id) & (results.repeat == repeat),
                "Speedup",
            ] = results.ExeTime.rdiv(
                results["Input"].map(base.set_index("Input").ExeTime)
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build, run and collect measurements for a benchmark program"
    )
    parser.add_argument(
        "-g",
        "--glob",
        default=str,
        help="glob pattern of benchmark directories",
        required=True,
    )
    parser.add_argument(
        "--compiler", help="path to the compiler executable", required=True
    )
    parser.add_argument(
        "-j",
        "--jit-path",
        help="path to the jit build directory",
        required=True,
    )
    parser.add_argument(
        "-x",
        "--exemode",
        help="execution model",
        choices=("direct", "profiler", "metrics"),
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
    cwd = os.getcwd()

    res_dir = pathlib.Path(f"{cwd}/results/")
    res_dir.mkdir(parents=True, exist_ok=True)

    experiments = list()
    # Each benchmark directory is expected to have a dictionary json file of the form:
    # {
    #   "name" : <benchmark name>
    #   "executables" : {
    #             "aot" : <aot executable>,
    #             "jit" : <jit executable>
    #         },
    #    "inputs" : {
    #        <input id> : <input arguments>
    #        ...
    #    }
    # }
    experiments_json = []
    basedir = Path(f"{args.glob}").parent
    for directory in Path(".").glob(f"{args.glob}"):
        if not Path(f"{directory}").is_dir():
            raise Exception(f"Cannot find directory {directory}")
        if not Path(f"{directory}/experiments.json").is_file():
            raise Exception(f"Directory {directory} is missing experiments.json")

        experiments_json.append(Path(f"{directory}/experiments.json"))

    if not experiments_json:
        raise Exception("Glob pattern did not produce any experiments")

    for exp_config in experiments_json:
        with open(exp_config, "r") as f:
            print("Loading", exp_config)
            config = json.load(f)
            print("config", config)
        experiments.append(
            Configuration(
                Path(exp_config).absolute().parent,
                args.compiler,
                args.jit_path,
                config,
            )
        )

    def gather_profiler_results(metrics):
        if args.machine == "amd":
            eresults_profiler, ecaching_profiler = e.build_and_run(
                rocprof(metrics, cwd)
            )
        elif args.machine == "nvidia":
            eresults_profiler, ecaching_profiler = e.build_and_run(nvprof(metrics))
        else:
            raise Exception("Expected amd or nvidia machine")

        print("=== results ===")
        print(eresults_profiler)
        print("=== eof results ===")
        if not metrics:
            # Postprocess profiler runs, add speedup.
            eresults_profiler = postprocess_profiler(eresults_profiler)
        # Store the intermediate, benchmark results.
        metrics_suffix = "-metrics" if metrics else ""
        eresults_profiler.to_csv(
            f"{res_dir}/{args.machine}-{basedir}-{e.config['name']}-results-profiler{metrics_suffix}.csv"
        )
        ecaching_profiler.to_csv(
            f"{res_dir}/{args.machine}-{basedir}-{e.config['name']}-caching-profiler{metrics_suffix}.csv"
        )

    def gather_results():
        eresults, ecaching = e.build_and_run(None)
        # Postprocess non-profiler runs, add speedup.
        eresults = postprocess(eresults)
        # Store the intermediate, benchmark results.
        eresults.to_csv(
            f"{res_dir}/{args.machine}-{basedir}-{e.config['name']}-results.csv"
        )
        ecaching.to_csv(
            f"{res_dir}/{args.machine}-{basedir}-{e.config['name']}-caching.csv"
        )

    # Build, run, and collect results for each experiment as gathered by glob
    # directories. Do profiler runs with and without metrics, and a run without
    # the profiler for end-to-end execution times.
    for e in experiments:
        # Gather results without the profiler.
        if args.exemode == "direct":
            gather_results()
        # Gather results with the machine-specific profiler WITHOUT metrics (gpu
        # counters)
        if args.exemode == "profiler":
            gather_profiler_results(metrics=False)
        # Gather results with the machine-specific profiler WITH metrics (gpu
        # counters).
        if args.exemode == "metrics":
            gather_profiler_results(metrics=True)

    print("Results are stored in ", res_dir)


if __name__ == "__main__":
    main()
