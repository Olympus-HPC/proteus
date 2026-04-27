import pathlib
import tempfile

import proteus


def main():
    source = 'extern "C" int forty_two() { return 42; } extern "C" int plus1(int x) { return x + 1; }'

    mod = proteus.compile(source, frontend="cpp", target="host")
    assert mod.get_function_address("forty_two") != 0
    plus1 = mod.get_function("plus1", restype=proteus.i32, argtypes=[proteus.i32])
    assert repr(plus1) == "<proteus.Function name='plus1' restype=proteus.i32 argtypes=[proteus.i32]>"
    assert plus1(41) == 42
    assert proteus.compile(source, frontend="cpp").get_function_address("forty_two") != 0

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "kernel.cpp"
        path.write_text(source)
        mod = proteus.compile(path, frontend="cpp", target="host")
        assert mod.get_function_address("forty_two") != 0
        assert mod.get_function("plus1", restype=proteus.i32, argtypes=[proteus.i32])(41) == 42

    print("python_host_cpp_smoke: ok")


if __name__ == "__main__":
    main()
