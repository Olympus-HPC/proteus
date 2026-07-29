import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import proteus


CORE_IR = r'''
source_filename = "roundtrip.ll"
target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@keep = externally_initialized global i32 7, align 4
@drop = global i32 9, align 4

define i32 @identity(i32 %value) #0 {
entry:
  ret i32 %value
}

attributes #0 = { nounwind }

!proteus.test = !{!0}
!0 = !{!"roundtrip metadata"}
'''


LINKED_IR = r'''
source_filename = "linked.ll"
target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @linked_function() {
entry:
  ret void
}
'''


OPTIMIZE_IR = r'''
target triple = "x86_64-unknown-linux-gnu"

define i32 @remove_zero(i32 %value) {
entry:
  %result = add i32 %value, 0
  ret i32 %result
}
'''


def _assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_serialization_and_verification():
    module = proteus.llvm.Module.from_ir(CORE_IR, name="roundtrip.ll")
    module.verify()

    bitcode = module.to_bitcode()
    assert isinstance(bitcode, bytes)
    assert bitcode.startswith(b"BC\xc0\xde")
    assert b"\x00" in bitcode

    reparsed = proteus.llvm.Module.from_bitcode(bitcode)
    reparsed.verify()
    text = reparsed.to_ir()
    assert 'target triple = "x86_64-unknown-linux-gnu"' in text
    assert "target datalayout" in text
    assert "nounwind" in text
    assert "!proteus.test" in text
    assert "roundtrip metadata" in text

    _assert_raises(RuntimeError, lambda: proteus.llvm.Module.from_bitcode(b"not bitcode"))
    _assert_raises(
        TypeError, lambda: proteus.llvm.Module.from_bitcode(bytearray(bitcode))
    )


def test_clone_link_prune_and_internalize():
    original = proteus.llvm.Module.from_ir(CORE_IR)
    cloned = original.clone()
    assert cloned is not original

    assert cloned.prune() is cloned
    assert "externally_initialized" not in cloned.to_ir()
    assert "externally_initialized" in original.to_ir()

    assert cloned.internalize(preserve=["identity"]) is cloned
    cloned_text = cloned.to_ir()
    assert "define i32 @identity" in cloned_text
    assert "@drop = internal global" in cloned_text
    assert "@drop = global" in original.to_ir()

    linked_input = proteus.llvm.Module.from_ir(LINKED_IR)
    linked = proteus.llvm.Module.link([original, linked_input])
    linked.verify()
    linked_text = linked.to_ir()
    assert "@identity" in linked_text
    assert "@linked_function" in linked_text
    assert "@linked_function" not in original.to_ir()

    _assert_raises(ValueError, lambda: proteus.llvm.Module.link([]))
    _assert_raises(KeyError, lambda: original.internalize(preserve=["missing_symbol"]))


def test_config_and_optimization():
    config = proteus.Config.current().codegen()
    assert config.method in {"rtc", "serial", "parallel"}
    assert config.opt_level in {"O0", "O1", "O2", "O3", "Os", "Oz"}
    assert 0 <= config.codegen_opt_level <= 3
    assert isinstance(config.specialize_arguments, bool)
    assert isinstance(config.specialize_dimensions, bool)
    assert isinstance(config.specialize_dimension_ranges, bool)
    assert isinstance(config.set_launch_bounds, bool)

    updated = config.replace(
        method="RTC",
        pipeline="default<O2>",
        opt_level="O2",
        codegen_opt_level=2,
        specialize_arguments=False,
        tuned_max_threads=256,
        min_blocks_per_sm=2,
    )
    assert updated is not config
    assert updated.method == "rtc"
    assert updated.pipeline == "default<O2>"
    assert updated.opt_level == "O2"
    assert updated.codegen_opt_level == 2
    assert not updated.specialize_arguments
    assert updated.tuned_max_threads == 256
    assert updated.min_blocks_per_sm == 2

    assert config.replace(pipeline=None).pipeline is None
    assert config.replace(tuned_max_threads=None).tuned_max_threads is None
    _assert_raises(ValueError, lambda: config.replace(method="invalid"))
    _assert_raises(ValueError, lambda: config.replace(opt_level="O4"))
    _assert_raises(ValueError, lambda: config.replace(codegen_opt_level=4))
    _assert_raises(TypeError, lambda: config.replace(unknown=True))
    _assert_raises(AttributeError, lambda: setattr(config, "method", "rtc"))

    module = proteus.llvm.Module.from_ir(OPTIMIZE_IR)
    assert module.optimize("x86-64", config=updated) is module
    module.verify()
    assert "add i32" not in module.to_ir()

    invalid_pipeline = updated.replace(pipeline="not-a-proteus-pass")
    _assert_raises(
        RuntimeError,
        lambda: proteus.llvm.Module.from_ir(OPTIMIZE_IR).optimize(
            "x86-64", config=invalid_pipeline
        ),
    )


def test_environment_and_per_kernel_config():
    if proteus.has_cuda or proteus.has_hip:
        return

    tuned = {
        "replay_kernel": {
            "CodeGen": "parallel",
            "Pipeline": "default<O1>",
            "OptLevel": "1",
            "CodeGenOptLevel": 2,
            "SpecializeArgs": False,
            "LaunchBounds": False,
            "SpecializeDims": False,
            "SpecializeDimsRange": False,
            "TunedMaxThreads": 128,
            "MinBlocksPerSM": 3,
        }
    }
    with tempfile.TemporaryDirectory() as directory:
        config_path = Path(directory) / "tuned.json"
        config_path.write_text(json.dumps(tuned), encoding="utf-8")

        environment = os.environ.copy()
        environment.update(
            {
                "PROTEUS_CODEGEN": "serial",
                "PROTEUS_OPT_PIPELINE": "default<O2>",
                "PROTEUS_OPT_LEVEL": "2",
                "PROTEUS_CODEGEN_OPT_LEVEL": "1",
                "PROTEUS_SPECIALIZE_ARGS": "1",
                "PROTEUS_SPECIALIZE_LAUNCH_BOUNDS": "1",
                "PROTEUS_SPECIALIZE_DIMS": "1",
                "PROTEUS_SPECIALIZE_DIMS_RANGE": "1",
                "PROTEUS_TUNED_KERNELS": str(config_path),
            }
        )
        source = """
import json
import proteus

def fields(config):
    return {
        "method": config.method,
        "pipeline": config.pipeline,
        "opt_level": config.opt_level,
        "codegen_opt_level": config.codegen_opt_level,
        "specialize_arguments": config.specialize_arguments,
        "set_launch_bounds": config.set_launch_bounds,
        "specialize_dimensions": config.specialize_dimensions,
        "specialize_dimension_ranges": config.specialize_dimension_ranges,
        "tuned_max_threads": config.tuned_max_threads,
        "min_blocks_per_sm": config.min_blocks_per_sm,
    }

view = proteus.Config.current()
print(json.dumps({"global": fields(view.codegen()),
                  "kernel": fields(view.codegen("replay_kernel"))}))
"""
        output = subprocess.check_output(
            [sys.executable, "-c", source], env=environment, text=True
        )

    configs = json.loads(output)
    assert configs["global"] == {
        "method": "serial",
        "pipeline": "default<O2>",
        "opt_level": "O2",
        "codegen_opt_level": 1,
        "specialize_arguments": True,
        "set_launch_bounds": True,
        "specialize_dimensions": True,
        "specialize_dimension_ranges": True,
        "tuned_max_threads": None,
        "min_blocks_per_sm": 0,
    }
    assert configs["kernel"] == {
        "method": "parallel",
        "pipeline": "default<O1>",
        "opt_level": "O1",
        "codegen_opt_level": 2,
        "specialize_arguments": False,
        "set_launch_bounds": False,
        "specialize_dimensions": False,
        "specialize_dimension_ranges": False,
        "tuned_max_threads": 128,
        "min_blocks_per_sm": 3,
    }


def test_host_capability_errors():
    if proteus.has_cuda or proteus.has_hip:
        return

    module = proteus.llvm.Module.from_ir(CORE_IR)
    _assert_raises(
        NotImplementedError,
        lambda: module.specialize_launch_dimensions(
            grid=(1, 1, 1), block=(1, 1, 1)
        ),
    )
    _assert_raises(
        NotImplementedError,
        lambda: module.emit_object(
            "x86-64", config=proteus.Config.current().codegen()
        ),
    )


def main():
    test_serialization_and_verification()
    test_clone_link_prune_and_internalize()
    test_config_and_optimization()
    test_environment_and_per_kernel_config()
    test_host_capability_errors()
    print("test_llvm_module: ok")


if __name__ == "__main__":
    main()
