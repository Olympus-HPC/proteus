import pathlib
import tempfile

import proteus


def main():
    source = r"""
define i32 @forty_two() {
entry:
  ret i32 42
}

define i32 @plus1(i32 %x) {
entry:
  %sum = add i32 %x, 1
  ret i32 %sum
}
"""

    mod = proteus.compile(source, frontend="llvmir", target="host")
    assert mod.get_function_address("forty_two") != 0
    plus1 = mod.get_function("plus1", restype=proteus.i32, argtypes=[proteus.i32])
    assert repr(plus1) == "<proteus.Function name='plus1' restype=proteus.i32 argtypes=[proteus.i32]>"
    assert plus1(41) == 42
    assert proteus.compile(source, frontend="llvmir").get_function_address("forty_two") != 0

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "kernel.ll"
        path.write_text(source)
        mod = proteus.compile(path, frontend="llvmir", target="host")
        assert mod.get_function_address("forty_two") != 0
        assert mod.get_function("plus1", restype=proteus.i32, argtypes=[proteus.i32])(41) == 42

    print("python_host_llvmir_smoke: ok")


if __name__ == "__main__":
    main()
