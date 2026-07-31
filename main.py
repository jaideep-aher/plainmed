"""Streamlit app for PlainMed medical text simplification."""

import json
import os
from pathlib import Path
from typing import List, Optional

import streamlit as st
import textstat

from scripts.model import MODEL_ID_FILE, predict


APP_TITLE = "PlainMed"
APP_DESCRIPTION = "Simplify medical text into plain language a patient can understand."
DISCLAIMER = "Educational tool only. This is not medical advice."
TEST_FILE = Path("data/processed/test.jsonl")
DATASET_CITATION = (
    "Basu, C., Vasu, R., Yasunaga, M., & Yang, Q. (2023). "
    "Med-EASi: Finely Annotated Dataset and Models for Controllable Simplification of Medical Texts."
)
DATASET_LINK = "https://huggingface.co/datasets/cbasu/Med-EASi"
REPO_LINK = "https://github.com/"


def get_api_key() -> Optional[str]:
    """Read the OpenAI API key from Streamlit secrets or the environment."""
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        secret_key = None
    return secret_key or os.getenv("OPENAI_API_KEY")


def get_model_id() -> Optional[str]:
    """Read the fine-tuned model ID from disk with a MODEL_ID environment fallback."""
    if MODEL_ID_FILE.exists():
        model_id = MODEL_ID_FILE.read_text(encoding="utf-8").strip()
        if model_id:
            return model_id
    return os.getenv("MODEL_ID")


def load_examples(path: Path = TEST_FILE, count: int = 3) -> List[str]:
    """Load example expert sentences from the held-out test JSONL file."""
    if not path.exists():
        return [
            "Pancreatitis is common.",
            "Nausea, vomiting, constipation, severe prostration, restlessness, and irritability are common.",
            "Some patients have weight loss, rarely enough to become underweight.",
        ]

    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(row["expert"])
            if len(examples) == count:
                break

    return examples


def reading_grade(text: str) -> float:
    """Return the Flesch-Kincaid grade level for a text string."""
    return round(float(textstat.flesch_kincaid_grade(text)), 1)


def render_grade_badge(label: str, grade: float) -> None:
    """Render a simple reading-level badge."""
    st.markdown(f"**{label} reading level:** Grade {grade}")


def render_header() -> None:
    """Render the app title, description, and disclaimer."""
    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)
    st.warning(DISCLAIMER)


def render_example_buttons(examples: List[str]) -> None:
    """Render example buttons that prefill the input text area."""
    columns = st.columns(len(examples))
    for index, (column, example) in enumerate(zip(columns, examples), start=1):
        if column.button(f"Example {index}", use_container_width=True):
            st.session_state["medical_text"] = example


def simplify_text(text: str, model_id: str, api_key: str) -> str:
    """Call the fine-tuned model to simplify medical text."""
    return predict(text, model_id=model_id, api_key=api_key)


def render_result(original: str, simplified: str) -> None:
    """Render original and simplified text side by side with reading-level badges."""
    left, right = st.columns(2)

    with left:
        st.subheader("Original")
        st.write(original)
        render_grade_badge("Original", reading_grade(original))

    with right:
        st.subheader("Plain language")
        st.write(simplified)
        render_grade_badge("Plain language", reading_grade(simplified))


def render_about(model_id: Optional[str]) -> None:
    """Render metadata about the model and dataset."""
    with st.expander("About"):
        st.write(f"Fine-tuned model ID: `{model_id or 'Not configured'}`")
        st.write(DATASET_CITATION)
        st.link_button("Med-EASi dataset", DATASET_LINK)
        st.link_button("Repository", REPO_LINK)


def run_app() -> None:
    """Run the Streamlit application flow."""
    render_header()

    examples = load_examples()
    render_example_buttons(examples)

    text = st.text_area(
        "Medical text",
        key="medical_text",
        height=180,
        placeholder="Paste a medical sentence or paragraph here.",
    )

    model_id = get_model_id()
    api_key = get_api_key()

    if st.button("Simplify", type="primary"):
        if not text.strip():
            st.error("Please enter medical text to simplify.")
        elif not model_id:
            st.error("Fine-tuned model ID is not configured yet.")
        elif not api_key:
            st.error("OpenAI API key is not configured.")
        else:
            try:
                with st.spinner("Simplifying text..."):
                    simplified = simplify_text(text.strip(), model_id, api_key)
                render_result(text.strip(), simplified)
            except Exception as exc:
                st.error(f"Could not simplify the text right now: {exc}")

    render_about(model_id)


if __name__ == "__main__":
    run_app()
