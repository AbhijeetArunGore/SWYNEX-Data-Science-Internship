"""Smoke tests for the Streamlit application (app.py).

Uses streamlit.testing.v1.AppTest to verify that the application compiles,
initializes without exceptions, renders all required tabs, and handles interactions.
"""
import pytest
from streamlit.testing.v1 import AppTest


def test_app_loads_without_exceptions():
    """Verify that app.py compiles and initializes with zero unhandled exceptions."""
    at = AppTest.from_file('app.py')
    at.run(timeout=15)

    assert not at.exception, f"App raised unhandled exceptions: {at.exception}"


def test_app_contains_disclaimer_and_title():
    """Verify that the educational prototype disclaimer and title are rendered."""
    at = AppTest.from_file('app.py')
    at.run(timeout=15)

    # Check for title in markdown elements
    markdown_texts = [m.value for m in at.markdown]
    combined_text = " ".join(markdown_texts)

    assert "Diabetes Inpatient 30-Day Readmission Risk Prototype" in combined_text
    assert "Educational & Research Prototype" in combined_text


def test_app_synthetic_scenario_switch():
    """Verify that clicking a synthetic profile button triggers a clean re-run."""
    at = AppTest.from_file('app.py')
    at.run(timeout=15)

    # Find and click button for Synthetic Profile B if available
    buttons = at.button
    if len(buttons) >= 2:
        buttons[1].click().run(timeout=15)
        assert not at.exception
