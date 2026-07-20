from knowmoredirt.evaluation import answer_matches



def test_answer_matches_ignores_terminal_punctuation() -> None:
    assert answer_matches("Mist Rail was delayed.", "Mist Rail was delayed")


def test_answer_matches_accepts_concise_boolean_before_explanation() -> None:
    assert answer_matches("No", "No; later inspection found no crack in the tank wall.")
