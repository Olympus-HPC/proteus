# User Interface

Proteus offers three ways to define JIT work.
All three eventually feed the same runtime specialization, caching, and
execution pipeline, but they differ in how JIT code is described and in their
build-time requirements.

| Interface | Best for | Specialization support | Runtime template instantiation | Compiler requirement |
| --- | --- | --- | --- | --- |
| [Code Annotations](annotations.md) | Incremental adoption in existing host, CUDA, or HIP code | Values, arrays, objects, and launch configuration | No | Application must be compiled with Clang |
| [C++ Frontend API](cpp-frontend.md) | Source-string JIT, runtime template instantiation, and compiler control | Values, arrays, objects, and launch configuration | Yes | Application can be compiled with any compatible compiler |
| [DSL API](dsl.md) | Programmatic IR construction and advanced runtime code generation | Values, arrays, and launch configuration | No | Application can be compiled with any compatible compiler |

For more detail on each path:

- [Code Annotations](annotations.md)
- [C++ Frontend API](cpp-frontend.md)
- [DSL API](dsl.md)

For runtime knobs that affect specialization, caching, and compilation, see
the [runtime configuration guide](config.md).
