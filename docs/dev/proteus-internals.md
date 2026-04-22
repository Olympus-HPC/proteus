# Proteus Internals

This section documents implementation-level behavior that cuts across multiple
frontends, target models, and runtime paths.

The user guide describes how to use Proteus APIs. These internals notes explain
how the runtime routes work, which components own compilation decisions, and
where configuration options take effect.

## Topics

- [Optimization Pipeline](optimization-pipeline.md): how
  `PROTEUS_OPT_PIPELINE` is selected, which compile paths use it, and how it
  participates in cache keys.
