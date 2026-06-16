import pathlib
import sys

import numpy as np


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "3D-Diffusion-Policy"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from diffusion_policy_3d.common.pybullet_validation import (  # noqa: E402
    _score_safety_sdf_candidate,
    _select_lowest_candidate_score_index,
)


def _score_key(score):
    return np.asarray(
        [
            score["has_pen"],
            score["pen_step_count"],
            score["pen_point_count"],
            score["neg_min_sdf"],
            score["neg_worstk_mean"],
            score["margin_violation"],
        ],
        dtype=np.float32,
    )


def test_safety_score_prefers_no_penetration():
    safe = _score_safety_sdf_candidate(
        np.asarray([[0.002, 0.006, 0.007]], dtype=np.float32),
        topk=2,
        d_select=0.005,
    )
    penetrated = _score_safety_sdf_candidate(
        np.asarray([[-0.001, 0.020, 0.030]], dtype=np.float32),
        topk=2,
        d_select=0.005,
    )

    selected = _select_lowest_candidate_score_index(
        np.stack([_score_key(penetrated), _score_key(safe)], axis=0)
    )

    assert selected == 1
    assert safe["has_pen"] == 0.0
    assert penetrated["has_pen"] == 1.0
    assert safe["pen_point_count"] == 0.0
    assert penetrated["pen_point_count"] == 1.0


def test_safety_score_prefers_larger_min_sdf_after_penetration_tie():
    better_clearance = _score_safety_sdf_candidate(
        np.asarray([[0.003, 0.006, 0.008]], dtype=np.float32),
        topk=2,
        d_select=0.005,
    )
    worse_clearance = _score_safety_sdf_candidate(
        np.asarray([[0.001, 0.020, 0.030]], dtype=np.float32),
        topk=2,
        d_select=0.005,
    )

    selected = _select_lowest_candidate_score_index(
        np.stack([_score_key(worse_clearance), _score_key(better_clearance)], axis=0)
    )

    assert selected == 1
    assert better_clearance["neg_min_sdf"] < worse_clearance["neg_min_sdf"]


def test_safety_score_topk_mean_and_margin_match_reference_logic():
    sdf_values = np.asarray([[0.001, 0.003, 0.006, 0.010]], dtype=np.float32)
    score = _score_safety_sdf_candidate(sdf_values, topk=2, d_select=0.005)

    assert score["has_pen"] == 0.0
    assert score["pen_step_count"] == 0.0
    assert score["pen_point_count"] == 0.0
    assert np.isclose(score["neg_min_sdf"], -0.001)
    assert np.isclose(score["neg_worstk_mean"], -0.002)
    assert np.isclose(score["margin_violation"], 0.003)


def test_safety_score_prefers_fewer_penetrating_steps_before_point_count():
    fewer_pen_steps = _score_safety_sdf_candidate(
        np.asarray(
            [
                [-0.002, -0.001, -0.001],
                [0.010, 0.020, 0.030],
            ],
            dtype=np.float32,
        ),
        topk=2,
        d_select=0.005,
    )
    more_pen_steps = _score_safety_sdf_candidate(
        np.asarray(
            [
                [-0.001, 0.010, 0.020],
                [-0.001, 0.010, 0.020],
            ],
            dtype=np.float32,
        ),
        topk=2,
        d_select=0.005,
    )

    selected = _select_lowest_candidate_score_index(
        np.stack([_score_key(more_pen_steps), _score_key(fewer_pen_steps)], axis=0)
    )

    assert selected == 1
    assert fewer_pen_steps["pen_step_count"] < more_pen_steps["pen_step_count"]
