from __future__ import annotations

from scripts.run_hkg_t24_online_residual_memory_refinement import refined_online_memory_specs


def test_refined_online_memory_specs_are_unique_and_bounded() -> None:
    specs = refined_online_memory_specs()
    ids = [spec.candidate_id for spec in specs]

    assert len(specs) == 36
    assert len(ids) == len(set(ids))
    assert {spec.context_set for spec in specs} == {"behavior", "seasonal_behavior", "all"}
    assert {spec.combine_mode for spec in specs} == {"lift_weighted"}
    assert {spec.halflife_rows for spec in specs} == {20.0, 45.0, 90.0}
