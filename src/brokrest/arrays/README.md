# Arrays

Arrays is a replacement for `TensorDict`, backed by `numpy.recarray`.

Previously, we used `tensordict.TensorClass` heavily, but,
since we are only using `torch` for indexing (and not much else, no GPU planned),
but too many of our dependencies use `numpy` directly,
e.g. `numba`, `shapely`, `bokeh`, `cv2` etc,
I have decided that we switch to `numpy` to avoid having to convert back and forth,
as converting is causing a lot of annoyance.

On top of that, we also wanted to implement jagged arrays,
which is not avialable in the base `TensorClass`,
which corresponds to `ArrayDict` here, and jagged array `ArrayList`.
