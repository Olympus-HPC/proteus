def main():
    namespace = {}
    exec("from proteus import *", namespace, namespace)
    assert namespace["active_backend"] in {"host", "cuda", "rocm"}, namespace["active_backend"]
    assert namespace["active_backend_variant"], namespace
    assert callable(namespace["compile"]), namespace.keys()
    assert "i32" in namespace, namespace.keys()
    print("test_wildcard_import: ok")


if __name__ == "__main__":
    main()
