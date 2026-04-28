import proteus

from test_support import expect_raises


def main():
    source = "define i32 @forty_two() { ret i32 42 }"

    expect_raises(
        ValueError,
        lambda: proteus.compile(
            source, frontend="llvmir", target="host", compiler="nvcc"
        ),
        "LLVMIR frontend does not support compiler='nvcc'",
    )
    expect_raises(
        ValueError,
        lambda: proteus.compile(
            source, frontend="llvmir", target="host", extra_args=["-O3"]
        ),
        "LLVMIR frontend does not support extra_args",
    )

    print("python_host_llvmir_validation: ok")


if __name__ == "__main__":
    main()
