"""Every results table in README.md must be exactly what make_report.py derives from runs/."""
import make_report


def test_readme_tables_match_run_files():
    with open(make_report.README, encoding="utf-8") as f:
        current = f.read()
    assert make_report.render(current) == current, "README.md results are stale: run `python make_report.py`"


def test_every_table_has_markers():
    with open(make_report.README, encoding="utf-8") as f:
        text = f.read()
    for name in make_report.TABLES:
        assert f"<!-- BEGIN:{name} -->" in text and f"<!-- END:{name} -->" in text
