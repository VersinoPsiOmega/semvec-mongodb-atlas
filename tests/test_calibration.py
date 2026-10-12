"""Firewall thresholds are a measurement, not a constant.

The gate compares an incoming turn against exclusion anchors and blocks
above a cosine threshold. That number is a property of the *embedder*:
two models place the same pair of sentences at different distances, so
a threshold measured for one is meaningless for another. Copying it
across silently is how a firewall stops blocking without erroring.

So thresholds live in a registry of calibrated profiles, each carrying
the separation it was derived from, and an embedder with no profile is
refused rather than defaulted. :func:`calibrate_threshold` performs the
measurement — it is the same routine that produced the shipped
profile, which is what makes the shipped number reproducible.
"""

from __future__ import annotations

import pytest

from semvec_atlas.firewall import (
    CALIBRATED_PROFILES,
    CalibrationError,
    ThresholdProfile,
    calibrate_threshold,
    threshold_for,
)

MPNET = "sentence-transformers/all-mpnet-base-v2"


class TestProfileRegistry:
    def test_mpnet_profile_is_the_measured_one(self):
        profile = CALIBRATED_PROFILES[MPNET]
        assert isinstance(profile, ThresholdProfile)
        assert profile.threshold == pytest.approx(0.45)

    def test_profile_records_the_separation_it_came_from(self):
        """A threshold without its margin is an unfalsifiable number."""
        profile = CALIBRATED_PROFILES[MPNET]
        benign_max, attack_min = profile.separation
        assert benign_max < profile.threshold < attack_min

    def test_threshold_for_known_model(self):
        assert threshold_for(MPNET) == pytest.approx(0.45)

    def test_unknown_model_is_refused_not_defaulted(self):
        """Fail closed: an uncalibrated model must not inherit a number.

        ``voyage-3.5`` is the deliberate example — it is the model this
        package will want as its default, and it has not been measured
        yet (the calibration run needs a Voyage API key). When someone
        registers that profile this test fails, which is the intended
        reminder to update it rather than a false alarm.
        """
        with pytest.raises(CalibrationError) as exc:
            threshold_for("voyage-3.5")
        assert "calibrate" in str(exc.value).lower()


class TestCalibration:
    """The measurement routine, exercised with a separable toy embedder."""

    class _Embedder:
        """Two clusters: 'attack …' near one axis, everything else near another."""

        DIM = 8

        def get_dimension(self) -> int:
            return self.DIM

        def get_embedding(self, text: str):
            import numpy as np

            v = np.zeros(self.DIM, dtype=float)
            if text.startswith("attack"):
                v[0] = 1.0
                v[1] = 0.08 * (len(text) % 5)
            else:
                v[2] = 1.0
                v[3] = 0.08 * (len(text) % 5)
            return v / np.linalg.norm(v)

    ANCHORS = ["attack anchor one", "attack anchor two"]
    ATTACKS = ["attack variant alpha", "attack variant beta gamma"]
    BENIGN = ["ordinary business sentence", "another routine remark here"]

    def test_threshold_lands_between_the_two_clusters(self):
        profile = calibrate_threshold(
            self._Embedder(),
            anchors=self.ANCHORS,
            attacks=self.ATTACKS,
            benign=self.BENIGN,
            model="toy-model",
        )
        benign_max, attack_min = profile.separation
        assert benign_max < profile.threshold < attack_min
        assert profile.model == "toy-model"

    def test_inseparable_corpus_raises_instead_of_guessing(self):
        """No margin means no honest threshold — say so rather than pick one."""
        with pytest.raises(CalibrationError) as exc:
            calibrate_threshold(
                self._Embedder(),
                anchors=self.ANCHORS,
                # A "benign" sample that is really an attack: the clusters
                # now overlap and no cut separates them.
                attacks=self.ATTACKS,
                benign=[*self.BENIGN, "attack disguised as benign"],
                model="toy-model",
            )
        assert "separat" in str(exc.value).lower()

    def test_empty_sample_is_refused(self):
        with pytest.raises(CalibrationError):
            calibrate_threshold(
                self._Embedder(),
                anchors=self.ANCHORS,
                attacks=[],
                benign=self.BENIGN,
                model="toy-model",
            )
