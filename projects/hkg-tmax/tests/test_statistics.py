from hkg_tmax.statistics import paired_moving_block_bootstrap


def test_bootstrap_difference_sign() -> None:
    result = paired_moving_block_bootstrap(
        candidate_losses=[0.8, 0.9, 1.0, 0.7, 0.8],
        baseline_losses=[1.0, 1.1, 1.2, 0.9, 1.0],
        block_length=2,
        repetitions=200,
        seed=7,
    )
    assert result.observed_mean < 0
    assert result.upper < 0
