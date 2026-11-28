import pytest

from the_loop.llm.pricing import estimate_cost_usd


def test_known_model_cost_is_exact():
    # gpt-4o-mini: $0.15 input / $0.60 output per 1M tokens.
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(
        0.75
    )
    assert estimate_cost_usd("gpt-4o-mini", 1_000, 500) == pytest.approx(
        (1_000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
    )


def test_longest_prefix_wins():
    assert estimate_cost_usd("gpt-4o-mini-2024-07-18", 1_000_000, 0) == pytest.approx(
        0.15
    )
    assert estimate_cost_usd("gpt-4o-2024-08-06", 1_000_000, 0) == pytest.approx(2.50)


@pytest.mark.parametrize(
    "model",
    [
        "qwen/qwen2.5-vl-7b",
        "lmstudio-community/qwen2.5-coder",
        "local-model",
        "llama-3.1-8b",
        "totally-unknown-model",
        "",
    ],
)
def test_local_and_unknown_models_are_free(model):
    assert estimate_cost_usd(model, 1_000_000, 1_000_000) == 0.0
