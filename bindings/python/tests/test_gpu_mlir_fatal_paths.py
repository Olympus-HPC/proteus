from test_support import expect_subprocess_failure, get_gpu_target, get_mlir_gpu_source


def main():
    target = get_gpu_target()
    if target is None:
        return

    source = get_mlir_gpu_source()
    missing_kernel = expect_subprocess_failure(
        f"""
import proteus
source = {source!r}
target = {target!r}
mod = proteus.compile(source, frontend="mlir", target=target, verify=True)
mod.get_kernel("missing_kernel", [proteus.ptr])
"""
    )
    assert missing_kernel.returncode != 0

    print("python_gpu_mlir_fatal_paths: ok")


if __name__ == "__main__":
    main()
