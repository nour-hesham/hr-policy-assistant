"""Streamlit HTTP client for the stateless HR policy assistant."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
HEALTH_TIMEOUT_SECONDS = 3
ASK_TIMEOUT_SECONDS = 35


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)


def check_backend() -> tuple[bool, dict[str, Any] | None]:
    try:
        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=HEALTH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("ready") is True:
            return True, payload
    except (requests.RequestException, ValueError, TypeError):
        pass
    return False, None


def ask_backend(question: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/ask",
            json={"question": question},
            timeout=ASK_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return None, "The request took too long. Please try again."
    except requests.RequestException:
        return None, "Couldn't connect to the backend. Confirm that FastAPI is running."

    if response.status_code != 200:
        messages = {
            422: "Enter a question between 1 and 1,000 characters.",
            502: "The answer could not be validated. Please try again.",
            503: "The answer service is temporarily unavailable. Please try again later.",
            504: "The answer service timed out. Please try again.",
        }
        return None, messages.get(
            response.status_code,
            "The request could not be completed. Please try again.",
        )

    try:
        payload = response.json()
    except ValueError:
        return None, "The backend returned an invalid response."

    if not valid_answer_payload(payload):
        return None, "The backend returned an invalid response."
    return payload, None


def valid_answer_payload(payload: Any) -> bool:
    basic_shape_is_valid = (
        isinstance(payload, dict)
        and payload.get("status") in {"answered", "partial", "not_found"}
        and isinstance(payload.get("answer"), str)
        and bool(payload["answer"].strip())
        and isinstance(payload.get("sources"), list)
    )
    if not basic_shape_is_valid:
        return False

    sources = payload["sources"]
    if payload["status"] == "not_found":
        return sources == []
    if not sources:
        return False

    required_fields = {
        "source_id",
        "document_id",
        "document_title",
        "section_title",
        "version",
        "source_file",
        "evidence",
    }
    return all(
        isinstance(source, dict)
        and all(
            isinstance(source.get(field), str) and bool(source[field].strip())
            for field in required_fields
        )
        for source in sources
    )


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    st.subheader("Sources used")
    policy_count = len({source.get("document_id") for source in sources})
    st.caption(
        f"{len(sources)} cited section{'s' if len(sources) != 1 else ''} "
        f"from {policy_count} polic{'ies' if policy_count != 1 else 'y'}."
    )

    for index, source in enumerate(sources):
        label = (
            f"{source.get('source_id', 'Source')} · "
            f"{source.get('document_title', 'Policy')} · "
            f"{source.get('section_title', 'Section')}"
        )
        with st.expander(label, expanded=index == 0):
            st.caption(
                f"Document ID: {source.get('document_id', '—')}  |  "
                f"Version: {source.get('version', '—')}  |  "
                f"Source file: {source.get('source_file', '—')}"
            )
            st.markdown(source.get("evidence", ""))


st.title("HR Policy Assistant")
st.write(
    "Ask a question about the supplied workplace policies. Answers are generated "
    "only when relevant policy evidence is available."
)

steps = st.columns(4)
for column, number, title in zip(
    steps,
    ("01", "02", "03", "04"),
    ("Ask", "Retrieve", "Generate", "Cite"),
):
    with column:
        st.markdown(f"**{number} {title}**")

ready, health = check_backend()
if ready and health:
    st.caption(
        f"Backend ready · {health.get('loaded_chunk_count', 0)} policy sections loaded"
    )
else:
    st.error(
        "The backend is not available. Start FastAPI on port 8000, then refresh this page."
    )

with st.form("policy_question_form", clear_on_submit=False):
    question = st.text_area(
        "Your policy question",
        placeholder="For example: How many annual leave days do I receive after six years?",
        max_chars=1000,
        height=120,
        key="policy_question",
    )
    submitted = st.form_submit_button(
        "Ask question",
        type="primary",
        disabled=not ready,
        use_container_width=False,
    )

if submitted:
    cleaned_question = question.strip()
    if not cleaned_question:
        st.warning("Enter a policy question before submitting.")
    else:
        with st.spinner("Searching the policies and preparing an answer..."):
            answer_payload, error_message = ask_backend(cleaned_question)

        if error_message:
            st.error(error_message)
        elif answer_payload:
            status = answer_payload["status"]
            answer = answer_payload["answer"]
            sources = answer_payload["sources"]

            if status == "not_found":
                st.info(answer)
            else:
                if status == "partial":
                    st.warning(
                        "The supplied policies support only part of this answer."
                    )
                st.subheader("Answer")
                st.markdown(answer)
                render_sources(sources)
