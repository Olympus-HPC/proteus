from ._backend import load_native_module


_native = load_native_module()

__doc__ = _native.__doc__
__file__ = getattr(_native, "__file__", __file__)

for _name in dir(_native):
    if _name.startswith("__") and _name not in {"__doc__", "__file__"}:
        continue
    globals()[_name] = getattr(_native, _name)
