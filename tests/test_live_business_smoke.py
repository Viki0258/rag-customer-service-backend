import runpy
from pathlib import Path

import pytest


MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/eval/live_business_smoke.py'))


def test_importing_live_evaluator_never_starts_network_calls():
    assert callable(MODULE['run'])


def test_case_selection_preserves_followup_context():
    cases = MODULE['select_cases'](['topic-switch-follow-up'])
    assert [case[0] for case in cases] == [
        'returns-en', 'follow-up', 'topic-switch', 'topic-switch-follow-up']


def test_case_selection_rejects_unknown_cases():
    with pytest.raises(ValueError):
        MODULE['select_cases'](['does-not-exist'])


def test_case_selection_defaults_to_full_suite():
    assert len(MODULE['select_cases'](None)) == 25
