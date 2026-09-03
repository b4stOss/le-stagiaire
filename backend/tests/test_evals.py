import pytest

from app.evals import _parse_number, grade_numeric


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("153,508", 153508.0),  # EN thousands
        ("20 215", 20215.0),  # FR thousands, plain space
        ("20 215", 20215.0),  # FR thousands, narrow no-break space
        ("14,7", 14.7),  # FR decimal
        ("4.7", 4.7),
        ("1,234.56", 1234.56),
        ("1.234,56", 1234.56),
        ("(22,332)", -22332.0),  # accounting negative
        ("-3", -3.0),
    ],
)
def test_parse_number(raw, expected):
    assert _parse_number(raw) == expected


def _gold(value, scale="million", tolerance_pct=0.5):
    return {"value": value, "unit": "EUR", "scale": scale, "tolerance_pct": tolerance_pct}


def test_numeric_matches_figure_in_millions():
    assert grade_numeric("Net revenue was EUR 153,508 million in 2025.", _gold(153508))


def test_numeric_normalizes_billions_to_millions():
    assert grade_numeric("Net revenue reached €153.5 billion.", _gold(153508))
    assert grade_numeric("Le chiffre d'affaires atteint 153,5 milliards d'euros.", _gold(153508))


def test_numeric_ignores_the_sign_carried_by_the_wording():
    assert grade_numeric("Stellantis reported a net loss of EUR 22,332 million.", _gold(-22332))


def test_numeric_rejects_wrong_figures_and_wrong_scales():
    assert not grade_numeric("Net revenue was EUR 143,508 million.", _gold(153508))
    assert not grade_numeric("Net revenue was EUR 153,508 thousand.", _gold(153508))


def test_numeric_percent_scale_compares_plain_values():
    assert grade_numeric("The CET1 ratio stood at 12.9% at year end.", _gold(12.9, scale="percent"))
    assert not grade_numeric("The CET1 ratio stood at 13.9% at year end.", _gold(12.9, scale="percent"))
