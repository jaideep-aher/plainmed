"""OpenAI fine-tuning and prediction helpers for PlainMed."""

import os
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI


BASE_MODEL = "gpt-4o-mini"
FINE_TUNE_BASE_MODEL = "gpt-4o-mini-2024-07-18"
SYSTEM_PROMPT = "You rewrite medical text in plain language a patient can understand."
TRAINING_FILE = Path("data/processed/train.jsonl")
MODEL_ID_FILE = Path("models/model_id.txt")
POLL_SECONDS = 60
FAILED_STATUSES = {"failed", "cancelled"}
SUCCESS_STATUS = "succeeded"


def get_client(api_key: Optional[str] = None) -> OpenAI:
    """Create an OpenAI client from an explicit key or OPENAI_API_KEY."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=key)


def upload_training_file(client: OpenAI, training_path: Path = TRAINING_FILE) -> str:
    """Upload a JSONL training file for OpenAI fine-tuning and return its file ID."""
    if not training_path.exists():
        raise FileNotFoundError(f"Training file not found: {training_path}")

    with training_path.open("rb") as handle:
        uploaded_file = client.files.create(file=handle, purpose="fine-tune")

    print(f"Uploaded training file: {uploaded_file.id}")
    return uploaded_file.id


def launch_fine_tuning_job(
    client: OpenAI,
    training_file_id: str,
    model: str = FINE_TUNE_BASE_MODEL,
):
    """Launch a supervised fine-tuning job and return the job object."""
    job = client.fine_tuning.jobs.create(
        model=model,
        training_file=training_file_id,
        method={"type": "supervised"},
    )
    print(f"Launched fine-tuning job: {job.id}")
    return job


def get_job_error(job) -> str:
    """Format a fine-tuning job error for human-readable logging."""
    error = getattr(job, "error", None)
    if not error:
        return "No error details returned."
    return str(error)


def poll_fine_tuning_job(
    client: OpenAI,
    job_id: str,
    poll_seconds: int = POLL_SECONDS,
):
    """Poll a fine-tuning job until it succeeds or reaches a failed terminal state."""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        print(f"Fine-tuning job {job.id}: {status}")

        if status == SUCCESS_STATUS:
            print(f"Fine-tuned model: {job.fine_tuned_model}")
            return job

        if status in FAILED_STATUSES:
            print(f"Fine-tuning failed: {get_job_error(job)}")
            return job

        time.sleep(poll_seconds)


def save_model_id(model_id: str, output_path: Path = MODEL_ID_FILE) -> None:
    """Save the fine-tuned model ID to disk."""
    if not model_id:
        raise ValueError("Cannot save an empty fine-tuned model ID.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(model_id + "\n", encoding="utf-8")
    print(f"Saved model ID to {output_path}")


def fine_tune_with_retries(max_attempts: int = 2) -> str:
    """Upload training data, launch fine-tuning, retry once on failure, and save the model ID."""
    client = get_client()

    for attempt in range(1, max_attempts + 1):
        print(f"Fine-tuning attempt {attempt} of {max_attempts}")
        training_file_id = upload_training_file(client)
        job = launch_fine_tuning_job(client, training_file_id)
        completed_job = poll_fine_tuning_job(client, job.id)

        if completed_job.status == SUCCESS_STATUS and completed_job.fine_tuned_model:
            save_model_id(completed_job.fine_tuned_model)
            return completed_job.fine_tuned_model

        if attempt == max_attempts:
            raise RuntimeError(f"Fine-tuning failed after {max_attempts} attempts: {get_job_error(completed_job)}")

        print("Relaunching fine-tuning job after failure.")

    raise RuntimeError("Fine-tuning did not complete.")


def predict(
    text: str,
    model_id: str,
    api_key: Optional[str] = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Rewrite medical text using the supplied model ID and system prompt."""
    client = get_client(api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    """Run fine-tuning and persist the resulting model ID."""
    fine_tune_with_retries()


if __name__ == "__main__":
    main()
