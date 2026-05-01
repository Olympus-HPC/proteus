import proteus


def main():
    available = proteus.available_backends()
    assert "host" in available, available
    assert proteus.active_backend in {"host", "cuda12"}, proteus.active_backend
    if proteus.active_backend == "cuda12":
        assert proteus.has_cuda is True

    print("test_backend_loader: ok")


if __name__ == "__main__":
    main()
