"""Tests for README code examples."""
import pytest
from pytest_examples import CodeExample, EvalExample


def test_readme_examples(readme_example: CodeExample, eval_example: EvalExample, save_example_plots, plot_output_dir, monkeypatch):
    """Test all code examples in README.md.

    This test is parametrized by pytest_generate_tests in conftest.py
    to extract and validate all Python code blocks from README.md.
    """
    # Set environment variable for examples that might need it
    monkeypatch.setenv("VARWG_PLOT_OUTPUT_DIR", str(plot_output_dir))
    eval_example.run(readme_example)
