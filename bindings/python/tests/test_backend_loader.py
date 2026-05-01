import proteus


def main():
    available = proteus.available_backends()
    assert available, available
    assert set(available).issubset({"host", "cuda", "rocm"}), available
    assert proteus.active_backend in {"host", "cuda", "rocm"}, proteus.active_backend
    assert proteus.active_backend in available, (proteus.active_backend, available)
    assert proteus.active_backend_variant in proteus.available_backend_variants()
    if proteus.active_backend == "cuda":
        assert proteus.has_cuda is True
    if proteus.active_backend == "rocm":
        assert proteus.has_hip is True

    print("test_backend_loader: ok")


if __name__ == "__main__":
    main()
