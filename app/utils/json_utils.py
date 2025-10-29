from __future__ import annotations

# JSON safety: recursively sanitize NaN/Inf and non-serializable values

def to_json_safe(obj):
    try:
        import numpy as _np
        import math as _math
    except Exception:
        _np = None
        import math as _math

    if obj is None:
        return None
    # Basic scalars
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        if _math.isfinite(obj):
            return float(obj)
        return None
    # Numpy scalars
    if _np is not None and isinstance(obj, (_np.integer,)):
        try:
            return int(obj)
        except Exception:
            return None
    if _np is not None and isinstance(obj, (_np.floating,)):
        try:
            val = float(obj)
            return val if _math.isfinite(val) else None
        except Exception:
            return None
    # Datetime-like
    try:
        from datetime import date, datetime as _dt
        if isinstance(obj, (date, _dt)):
            try:
                return obj.isoformat()
            except Exception:
                return str(obj)
    except Exception:
        pass
    # Lists/tuples
    if isinstance(obj, (list, tuple)):
        return [ to_json_safe(x) for x in obj ]
    # Dicts
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                ks = str(k)
            except Exception:
                ks = repr(k)
            out[ks] = to_json_safe(v)
        return out
    # Fallback: try float, then str
    try:
        f = float(obj)
        if _math.isfinite(f):
            return f
        return None
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None
