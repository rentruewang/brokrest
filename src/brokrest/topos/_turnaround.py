# Copyright (c) The BrokRest Authors - All Rights Reserved

import typing

import numpy as np
import tensordict as td

from .rects import Segment

if typing.TYPE_CHECKING:
    from .candles import Candle


def simple_keep_turnaround_segments(candles: "Candle") -> Segment:
    if candles.ndim != 1:
        raise ValueError("Only support 1D candles.")

    points = candles.center_points()
    segments = Segment.from_start_end(start=points[:-1], end=points[1:])

    # No segments to merge.
    if len(segments) <= 1:
        return segments

    # The result.
    segment_list: list[Segment] = []

    prev_segment = segments[0]
    segment = segments[-1]

    def _make_new_segment_and_append():
        # Edge case where there are no new updates.
        if prev_segment is segment:
            return

        new_segment = Segment.from_start_end(
            start=prev_segment.start, end=segment.start
        )
        segment_list.append(new_segment)

    for segment in segments[1:]:
        # Both pointing up or both pointing down.
        if segment.dy.sign() == prev_segment.dy.sign():
            continue

        # Merge until previous segment.
        _make_new_segment_and_append()
        prev_segment = segment

    _make_new_segment_and_append()
    return td.stack(segment_list)


def vectorized_keep_turnaround_points(candles: "Candle") -> Segment:
    """
    Merge nearby elements that are consecutively increasing / decreasing,
    and only preserve the turn around points (into segments).

    Do this with cumsum (with heisen sequence).
    """

    raise NotImplementedError


def cumsum_with_reset(tensor: np.ndarray, reset: np.ndarray):
    "Cumsum with resetting signals."

    if not np.isdtype(reset.dtype, "bool"):
        raise ValueError(f"Reset should be a boolean tensor. {reset.dtype=}.")

    scan = _AssociativeScan()
    invert = 1 - reset.astype("float64")
    result = scan(tensor, invert, axis=0)
    return result


class _AssociativeScan:
    """
    Code adapted from here:
    https://github.com/pytorch/pytorch/issues/53095#issuecomment-2102409471

    which is in turn based on heinen sequence:
    https://github.com/glassroom/heinsen_sequence

    Organize into a class to show scope of the adaptation.
    """

    def __call__(self, values: np.ndarray, coeffs: np.ndarray, axis: int) -> np.ndarray:
        log_values = self._complex_log(values.astype("float64"))
        log_coeffs = self._complex_log(coeffs.astype("float64"))
        a_star = np.cumsum(log_coeffs, axis=axis)

        # Here, `torch.logcumsumexp` is replaced with `np.logaddexp.accumulate`.
        log_x0_plus_b_star = np.logaddexp.accumulate(log_values - a_star, axis=axis)
        log_x = a_star + log_x0_plus_b_star
        result = np.real(np.exp(log_x))
        return result

    def _complex_log(self, float_input: np.ndarray, eps: float = 1e-6):
        eps_arr = np.ones_like(float_input) * eps
        real = np.log(np.maximum(abs(float_input), eps_arr))
        imag = (float_input < 0).astype(float_input.dtype) * np.pi
        return real + imag * 1j
