"""When a conversation gets flagged for a human."""

import pytest

import qa
from server.escalation import ASKED_FOR_HUMAN, INCOMPLETE, NOT_IN_DOCUMENTS, flag_reason

ANSWERED = "Covered up to RM25,000 (Source: Benefits Schedule, page 2)."


@pytest.mark.parametrize("question", [
    "Can I talk to a real person please?",
    "I want to speak to someone about my claim",
    "Please get a human",
    "Can your team contact me?",
])
def test_asking_for_a_person_is_flagged(question):
    assert flag_reason(question, ANSWERED, {}) == ASKED_FOR_HUMAN


@pytest.mark.parametrize("question", [
    "Is a burst pipe covered on Silver tier?",
    "How do I make a complaint?",
    "Does the policy cover personal belongings?",
])
def test_ordinary_questions_are_not_flagged(question):
    assert flag_reason(question, ANSWERED, {}) is None


def test_decline_is_flagged():
    assert flag_reason("Do you cover my car?", "The provided documents do not answer this.", {}) == NOT_IN_DOCUMENTS


def test_incomplete_answer_is_flagged():
    assert flag_reason("Burst pipe, Silver?", qa.INCOMPLETE_ANSWER, {"incomplete": True}) == INCOMPLETE


def test_explicit_request_wins_over_other_reasons():
    assert flag_reason("Talk to someone, the bot can't help", qa.INCOMPLETE_ANSWER, {"incomplete": True}) == ASKED_FOR_HUMAN
