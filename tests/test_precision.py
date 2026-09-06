"""Precision properties of `d`. See scripts/precision_budget.py for the full budget.

These lock in the architectural claim the accuracy target rests on: `d` rejects
error that is common to every run, and is sensitive only to error that differs
between runs. If that ever stops being true, centimetre accuracy stops being
affordable and these tests should fail loudly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from precision_budget import measure  # noqa: E402

CLEAN = dict(wander_m=0.0, reloc_drift_m=0.0, reloc_noise_m=0.0, common_drift_m=0.0)


def test_perfect_input_is_recovered_exactly():
    """The analysis stage must contribute no error of its own."""
    assert measure(**CLEAN)["rms_cm"] == pytest.approx(0.0, abs=1e-6)


def test_rider_wander_is_not_counted_as_error():
    """Wander is in both the measured and the reference profile, so it cancels.

    This is the distinction the whole accuracy target depends on: a rider
    varying between runs is not the pipeline being wrong.
    """
    assert measure(**{**CLEAN, "wander_m": 0.5})["rms_cm"] == pytest.approx(0.0, abs=1e-6)


def test_shared_map_error_is_rejected():
    """A metre of error common to every run must not move `d` by a centimetre.

    The load-bearing claim: absolute map accuracy is nearly irrelevant, so the
    requirement is run-to-run consistency, not a survey-grade reconstruction.
    """
    got = measure(**{**CLEAN, "common_drift_m": 1.0})["rms_cm"]
    assert got < 1.0, f"shared error leaked into d: {got:.2f}cm"


def test_per_run_error_transfers_roughly_linearly():
    """Per-run registration error is the one thing `d` cannot reject."""
    small = measure(**{**CLEAN, "reloc_drift_m": 0.05})["rms_cm"]
    large = measure(**{**CLEAN, "reloc_drift_m": 0.20})["rms_cm"]
    assert 3.0 < large / small < 5.0, f"expected ~4x, got {large / small:.1f}x"


def test_two_centimetre_registration_gives_sub_centimetre_d():
    """The engineering target, stated as a test."""
    got = measure(**{**CLEAN, "reloc_drift_m": 0.02})["rms_cm"]
    assert got < 1.0, f"2cm per-run registration gave {got:.2f}cm of d error"
