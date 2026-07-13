from knowmoredirt.evaluation import exact_match, semantic_match, token_f1


def test_semantic_scoring_distinguishes_formatting_from_real_errors():
    assert not exact_match("Officer Talen", "Talen")
    assert semantic_match("Officer Talen", "Talen")
    assert semantic_match("No", "No; later inspection found no crack.")
    assert semantic_match("has no stated translation", "unknown")
    assert not semantic_match("false", "unknown")
    assert token_f1("Plaintiff Harbor Coop", "Harbor Coop") >= 0.8


def test_semantic_scoring_accepts_equivalent_reported_modal_propositions():
    assert semantic_match(
        "the cache should expire every 8 minutes.",
        "It should expire every 8 minutes.",
    )
    assert not semantic_match(
        "the cache should expire every 20 minutes.",
        "It should expire every 8 minutes.",
    )
