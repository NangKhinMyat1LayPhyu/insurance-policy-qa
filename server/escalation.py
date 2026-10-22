"""Decide when a conversation needs a human instead of the bot."""

import re

import qa

ASKED_FOR_HUMAN = "Customer asked for a person"
NOT_IN_DOCUMENTS = "Documents don't answer it"
INCOMPLETE = "Bot couldn't finish an answer"

# The exact refusal qa.py's system prompt tells the model to use.
DECLINE_PHRASE = "the provided documents do not answer this"

HUMAN_REQUEST = re.compile(
    r"\b(human|real person|live agent|representative|"
    r"(speak|talk|chat) (to|with) (someone|somebody|a person|a human|staff|an agent|your team)|"
    r"(call|contact) me)\b",
    re.IGNORECASE,
)


def flag_reason(question, answer, info):
    """Why this exchange needs a human, or None if the bot handled it."""
    if HUMAN_REQUEST.search(question):
        return ASKED_FOR_HUMAN
    if info.get("incomplete") or answer == qa.INCOMPLETE_ANSWER:
        return INCOMPLETE
    if DECLINE_PHRASE in answer.lower():
        return NOT_IN_DOCUMENTS
    return None
