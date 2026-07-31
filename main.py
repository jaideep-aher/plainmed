"""Streamlit app for PlainMed medical text simplification."""

import json
import os
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Optional

import streamlit as st
import textstat

from scripts.model import BASE_MODEL, MODEL_ID_FILE, predict


APP_TITLE = "PlainMed"
APP_DESCRIPTION = "Simplify medical text into plain language a patient can understand."
DISCLAIMER = "Educational tool only. This is not medical advice."
TEST_FILE = Path("data/processed/test.jsonl")
PRACTICAL_EXAMPLES = [
    "Patients in whom cancer is identified require CT of the chest and abdomen to determine extent of tumor spread.",
    "Since it has a half-life of 3 to 5 minutes, the infusion has to be continuous, and interruption can be fatal.",
    "If large portions of the body are affected, fluid and electrolyte loss may be significant.",
]
PATIENT_REWRITE_PROMPT = (
    "You rewrite medical text for a patient. Use plain everyday words, short sentences, "
    "and explain medical terms in context. Keep the meaning, do not add new facts, and do "
    "not copy the original wording unless it is already simple."
)
REPAIR_PROMPT = (
    "Rewrite this for a patient at about a 6th grade reading level. Use 2 to 4 short "
    "sentences. Explain abbreviations and medical terms. Keep all important warnings and "
    "clinical caveats. Do not add facts that are not in the original."
)
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
        return PRACTICAL_EXAMPLES[:count]

    examples = list(PRACTICAL_EXAMPLES[:count])
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            expert_text = row["expert"]
            if expert_text in examples or not is_practical_medical_example(expert_text):
                continue
            examples.append(expert_text)
            if len(examples) == count:
                break

    return examples[:count]


def is_practical_medical_example(text: str) -> bool:
    """Return whether a sample looks like useful patient-facing medical text."""
    blocked_terms = ["rate law", "sulfur dioxide", "sulfolene", "authors were able"]
    medical_terms = [
        "patient",
        "patients",
        "cancer",
        "infection",
        "treatment",
        "diagnosed",
        "fatal",
        "body",
        "fluid",
        "disease",
        "clinical",
        "drug",
        "blood",
    ]
    lowered_text = text.lower()
    return not any(term in lowered_text for term in blocked_terms) and any(
        term in lowered_text for term in medical_terms
    )


def reading_grade(text: str) -> float:
    """Return the Flesch-Kincaid grade level for a text string."""
    return round(float(textstat.flesch_kincaid_grade(text)), 1)


def similarity_ratio(first_text: str, second_text: str) -> float:
    """Return a similarity score between two text strings."""
    return SequenceMatcher(None, first_text.lower(), second_text.lower()).ratio()


def needs_rewrite_repair(original: str, simplified: str) -> bool:
    """Return whether the first rewrite is too close to the original or not easier to read."""
    if not simplified.strip():
        return True
    return similarity_ratio(original, simplified) > 0.82 or reading_grade(simplified) >= reading_grade(original)


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
    """Call the fine-tuned model first and fall back when the rewrite is weak."""
    first_pass = predict(text, model_id=model_id, api_key=api_key, system_prompt=PATIENT_REWRITE_PROMPT)
    if not needs_rewrite_repair(text, first_pass):
        return first_pass

    repaired_output = predict(text, model_id=model_id, api_key=api_key, system_prompt=REPAIR_PROMPT)
    if not needs_rewrite_repair(text, repaired_output):
        return repaired_output

    return predict(text, model_id=BASE_MODEL, api_key=api_key, system_prompt=REPAIR_PROMPT)


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
        st.write("The app tries the fine-tuned model first, then uses a stricter rewrite pass if the first answer is too close to the original.")
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
