"""Package setup for PlainMed."""

from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path: Path = Path("requirements.txt")):
    """Read pinned install requirements from requirements.txt."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def main() -> None:
    """Run package setup."""
    setup(
        name="plainmed",
        version="0.1.0",
        description="Medical text simplification with a fine-tuned OpenAI model.",
        py_modules=["main"],
        packages=find_packages(include=["scripts", "scripts.*"]),
        install_requires=read_requirements(),
        python_requires=">=3.9",
    )


if __name__ == "__main__":
    main()
