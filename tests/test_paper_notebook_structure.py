from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_NOTEBOOKS = (
    REPO_ROOT / "papers/error_mcmc/notebook.py",
    REPO_ROOT / "papers/fastergs/notebook.py",
    REPO_ROOT / "papers/fastgs/notebook.py",
    REPO_ROOT / "papers/nht/notebook.py",
    REPO_ROOT / "papers/nht_fast_gs/notebook.py",
    REPO_ROOT / "papers/powerfoam/notebook.py",
    REPO_ROOT / "papers/radfoam/notebook.py",
    REPO_ROOT / "papers/scaffold_gs/notebook.py",
    REPO_ROOT / "papers/stoch3dgs/notebook.py",
    REPO_ROOT / "papers/stoch_fast_gs/notebook.py",
    REPO_ROOT / "papers/svraster/notebook.py",
    REPO_ROOT / "papers/triangle_splatting/notebook.py",
)

MARKDOWN_CELL_PATTERN = re.compile(
    r'mo\.md\("""\n(?P<body>.*?)\n    """\)',
    re.DOTALL,
)
COLUMN_DECORATOR_PATTERN = re.compile(
    r"@app\.(?:cell|function|class_definition)\([^)]*\bcolumn\s*=",
)
CELL_PATTERN = re.compile(
    r"@app\.cell[^\n]*\ndef _\((?P<args>.*?)\):(?P<body>.*?)(?=\n\n@app\.|\Z)",
    re.DOTALL,
)
FORBIDDEN_HEADINGS = {
    "# Training",
    "# Densification",
    "# Support",
    "# Support Code",
    "# Configuration",
    "# Configuration model",
    "## Config definition",
    "## Training controls",
    "## Training output",
    "## Training setup",
}
IMPLEMENTATION_MARKERS = (
    "    ## Method and config\n",
    "    ## Training assembly\n",
    "    ## IO wiring",
    "    ## Execution\n",
    "    ## Densification implementation\n",
    "    ## Scaffold-GS implementation\n",
    "    ## Utilities\n",
)


def markdown_headings(source: str) -> list[str]:
    """Return markdown headings rendered by notebook markdown cells."""
    headings: list[str] = []
    for match in MARKDOWN_CELL_PATTERN.finditer(source):
        for line in match.group("body").splitlines():
            stripped_line = line.strip()
            if stripped_line.startswith("#"):
                headings.append(stripped_line)
    return headings


def first_implementation_index(source: str) -> int:
    """Return the first implementation section after the IO heading."""
    io_index = source.index("    ## IO\n")
    indices = [
        source.index(marker, io_index)
        for marker in IMPLEMENTATION_MARKERS
        if marker in source[io_index:]
    ]
    assert indices, "training notebook has no implementation section"
    return min(indices)


def test_training_notebooks_use_io_first_single_column_layout() -> None:
    for notebook_path in TRAINING_NOTEBOOKS:
        source = notebook_path.read_text()
        headings = markdown_headings(source)

        assert 'app = marimo.App(width="medium")' in source, notebook_path
        assert 'width="columns"' not in source, notebook_path
        assert not COLUMN_DECORATOR_PATTERN.search(source), notebook_path
        assert "## IO" in headings, notebook_path

        title_index = next(
            index
            for index, heading in enumerate(headings)
            if heading.startswith("# ")
        )
        assert headings[title_index + 1] == "## IO", notebook_path
        assert FORBIDDEN_HEADINGS.isdisjoint(headings), notebook_path


def test_training_results_and_viewers_render_inside_io_section() -> None:
    for notebook_path in TRAINING_NOTEBOOKS:
        source = notebook_path.read_text()
        implementation_index = first_implementation_index(source)

        result_view_index = source.index("def _(training_result_view):")
        viewer_index = source.index("def _(training_viewer):")

        assert result_view_index < implementation_index, notebook_path
        assert viewer_index < implementation_index, notebook_path


def test_training_status_and_preview_cells_use_shared_refresh_boundaries() -> (
    None
):
    for notebook_path in TRAINING_NOTEBOOKS:
        source = notebook_path.read_text()
        cells = [
            (match.group("args"), match.group("body"))
            for match in CELL_PATTERN.finditer(source)
        ]
        result_cells = [
            (args, body)
            for args, body in cells
            if "return (training_result_view,)" in body
        ]
        viewer_cells = [
            (args, body)
            for args, body in cells
            if "return (training_viewer,)" in body
        ]

        assert "render_training_status_panel_from_handle" in source, (
            notebook_path
        )
        assert "metric_text_parts" not in source, notebook_path
        assert "format_duration(snapshot" not in source, notebook_path
        assert "training_viewer_handle.snapshot()" not in source, notebook_path

        assert len(result_cells) == 1, notebook_path
        result_args, result_body = result_cells[0]
        assert "training_status_refresh" in result_args + result_body, (
            notebook_path
        )
        assert "training_inspector_refresh" not in result_args + result_body, (
            notebook_path
        )
        assert "render_training_status_panel_from_handle" in result_body, (
            notebook_path
        )

        assert len(viewer_cells) == 1, notebook_path
        viewer_args, viewer_body = viewer_cells[0]
        assert "training_inspector_refresh" in viewer_args + viewer_body, (
            notebook_path
        )
        assert "training_status_refresh" not in viewer_args + viewer_body, (
            notebook_path
        )
        assert "training_result" in viewer_args + viewer_body, notebook_path
        assert "render_training_status_panel_from_handle" in viewer_body, (
            notebook_path
        )
        assert "preview_status_panel" in viewer_body, notebook_path
        assert "fixed_view_panel = training_inspector.panel" in viewer_body, (
            notebook_path
        )
        assert "mo.vstack" in viewer_body, notebook_path
