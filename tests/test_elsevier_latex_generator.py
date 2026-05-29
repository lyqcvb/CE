import importlib.util
import sys
from pathlib import Path


def load_generator_module():
    script = Path(__file__).resolve().parents[1] / "writing" / "elsevier_latex" / "generate_elsevier_latex.py"
    spec = importlib.util.spec_from_file_location("generate_elsevier_latex", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_full_manuscript_from_chapters_uses_article_title_and_rewrites_paths(tmp_path):
    generator = load_generator_module()
    writing = tmp_path / "writing"
    chapters = writing / "chapters"
    plan = writing / "plan"
    chapters.mkdir(parents=True)
    plan.mkdir(parents=True)

    (plan / "outline.md").write_text(
        "# Outline\n\n## Title\n\nDirect chapter assembly title\n",
        encoding="utf-8",
    )
    (chapters / "00_Abstract.md").write_text(
        "# Abstract\n\nAbstract body.\n\nKeywords: EEG; HOFC\n",
        encoding="utf-8",
    )
    (chapters / "01_Introduction.md").write_text(
        "# 1. Introduction\n\n![Fig. 1. Caption](../figures/fig1.png)\n",
        encoding="utf-8",
    )
    (chapters / "06_References.md").write_text(
        "# References\n\n[1] Example reference.\n",
        encoding="utf-8",
    )

    manuscript = generator.build_full_manuscript_from_chapters(writing)

    assert manuscript.startswith("# Direct chapter assembly title\n\n# Abstract")
    assert "![Fig. 1. Caption](figures/fig1.png)" in manuscript
    assert "# 1. Introduction" in manuscript
    assert "# References" in manuscript
    assert "# Outline" not in manuscript


def test_table_caption_with_period_after_number_is_supported():
    generator = load_generator_module()

    rows = [
        ["Metric", "Value"],
        ["---", "---"],
        ["A", "1.23456"],
    ]

    assert generator.clean_table_caption("Table 1. Caption text.") == "Caption text."

    latex = generator.convert_table(rows, "Table 1. Caption text.", 0)

    assert r"\caption{Caption text.}" in latex
    assert r"\label{tab:1}" in latex


def test_short_table_note_stays_inside_float_and_table1_is_barriered():
    generator = load_generator_module()

    rows = [
        ["Metric", "Value"],
        ["---", "---"],
        ["A", "1.23456"],
    ]

    latex = generator.convert_table(rows, "Table 1. Caption text.", 0, "Note: Example note.")

    assert r"\footnotesize Note: Example note." in latex
    assert latex.index(r"\footnotesize Note: Example note.") < latex.index(r"\end{table}")
    assert latex.rstrip().endswith(r"\FloatBarrier")


def test_appendix_table_caption_labels_are_supported():
    generator = load_generator_module()

    rows = [
        ["Metric", "Value"],
        ["---", "---"],
        ["A", "1.23456"],
    ]

    assert generator.clean_table_caption("Table A.1. Appendix caption.") == "Appendix caption."

    latex = generator.convert_table(rows, "Table A.1. Appendix caption.", 0)

    assert r"\caption{Appendix caption.}" in latex
    assert r"\label{tab:A.1}" in latex


def test_appendix_heading_resets_table_numbering_to_a_prefix():
    generator = load_generator_module()

    latex = generator.convert_heading("# Appendix")

    assert r"\appendix" in latex
    assert r"\section*{Appendix}" in latex
    assert r"\setcounter{table}{0}" in latex
    assert r"\renewcommand{\thetable}{A.\arabic{table}}" in latex


def test_table_caption_name_overrides_elsevier_default_period():
    generator = load_generator_module()

    frontmatter = generator.make_frontmatter(
        "Title",
        ["Background and Objective: Abstract body."],
        ["EEG"],
    )

    assert r"\renewcommand{\tablename}{Table}" in frontmatter
    assert r"\captionsetup[table]{name=Table,labelsep=period}" in frontmatter
    assert r"\usepackage{placeins}" in frontmatter


def test_preprint_journal_placeholder_footer_is_suppressed():
    generator = load_generator_module()

    frontmatter = generator.make_frontmatter(
        "Title",
        ["Background and Objective: Abstract body."],
        ["EEG"],
    )

    assert r"\journal{Biomedical Signal Processing and Control}" in frontmatter
    assert r"\def\ps@pprintTitle" in frontmatter


def test_flat_figure_asset_name_follows_visible_figure_label():
    generator = load_generator_module()

    assert (
        generator.figure_asset_name(
            Path("writing/figures/results/fig7_fc_stability.pdf"),
            "Fig. 6. Epoch based FC stability across groups and FC types.",
        )
        == "fig6_fc_stability.pdf"
    )


def test_preferred_figure_asset_uses_pdf_sibling(tmp_path):
    generator = load_generator_module()
    old_root = generator.ROOT
    try:
        generator.ROOT = tmp_path
        figure_dir = tmp_path / "writing" / "figures" / "results"
        figure_dir.mkdir(parents=True)
        (figure_dir / "fig7_fc_stability.png").write_bytes(b"png placeholder")
        (figure_dir / "fig7_fc_stability.pdf").write_bytes(b"%PDF placeholder")

        assert (
            generator.preferred_figure_asset("figures/results/fig7_fc_stability.png")
            == "figures/results/fig7_fc_stability.pdf"
        )
    finally:
        generator.ROOT = old_root
