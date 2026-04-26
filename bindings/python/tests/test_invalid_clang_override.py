import os
import subprocess
import sys


def main():
    source = 'extern "C" int forty_two() { return 42; }'
    env = os.environ.copy()
    env["PROTEUS_CLANGXX_BIN"] = "/definitely/missing/clang++"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import proteus; "
                f'proteus.compile({source!r}, frontend="cpp", target="host")'
            ),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, output
    assert "PROTEUS_CLANGXX_BIN" in output, output
    assert "/definitely/missing/clang++" in output, output

    print("test_invalid_clang_override: ok")


if __name__ == "__main__":
    main()
