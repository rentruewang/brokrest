# Copyright (c) The BrokRest Authors - All Rights Reserved

import typing

import tensordict as td
import torch

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


def cumsum_with_reset(tensor: torch.Tensor, reset: torch.Tensor):
    "Cumsum with resetting signals."

    if reset.dtype != torch.bool:
        raise ValueError(f"Reset should be a boolean tensor. {reset.dtype=}.")

    scan = _AssociativeScan()
    invert = 1 - reset.float()
    result = scan(tensor, invert, dim=0)
    return result


class _AssociativeScan:
    """
    Code adapted from here:
    https://github.com/pytorch/pytorch/issues/53095#issuecomment-2102409471

    which is in turn based on heinen sequence:
    https://github.com/glassroom/heinsen_sequence

    Organize into a class to show scope of the adaptation.
    """

    def __call__(
        self, values: torch.Tensor, coeffs: torch.Tensor, dim: int
    ) -> torch.Tensor:
        log_values = self._complex_log(values.float())
        log_coeffs = self._complex_log(coeffs.float())
        a_star = torch.cumsum(log_coeffs, dim=dim)
        log_x0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=dim)
        log_x = a_star + log_x0_plus_b_star
        return torch.exp(log_x).real

    def _complex_log(self, float_input: torch.Tensor, eps: float = 1e-6):
        eps_tensor = float_input.new_tensor(eps)
        real = float_input.abs().maximum(eps_tensor).log()
        imag = (float_input < 0).to(float_input.dtype) * torch.pi
        return torch.complex(real, imag)
