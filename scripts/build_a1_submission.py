import re
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
NOTES_DIRECTORY = REPOSITORY_ROOT / "notes" / "a1"
MARKDOWN_OUTPUT = NOTES_DIRECTORY / "submission.md"
PDF_OUTPUT = NOTES_DIRECTORY / "submission.pdf"

REPORTS = (
    NOTES_DIRECTORY / "p1" / "p1_hyperparameters.md",
    NOTES_DIRECTORY / "p2_scaling_law_reliability.md",
    NOTES_DIRECTORY / "p3_measuring_variation.md",
    NOTES_DIRECTORY / "p4_amplification.md",
    NOTES_DIRECTORY / "p5_loss_curve_augury.md",
    NOTES_DIRECTORY / "p6_activation_gradient_norms.md",
    NOTES_DIRECTORY / "run_index.md",
)


def read_report(path):
    text = path.read_text().strip()
    if path.parent.name == "p1":
        text = re.sub(r"]\(([^/)]+\.png)\)", r"](p1/\1)", text)
    return text


def build_markdown():
    preamble = """---
title: "Assignment 1: Basics — Empirical Machine Learning"
author: "Kesavan Ramakrishnan"
geometry: margin=0.75in
fontsize: 10pt
---

Unless stated otherwise, the standard run uses d8 (8 layers, width 512), 614.4M fresh training tokens, AdamW, learning rate 0.003, 1% warmup, linear decay, batch size 64, weight decay 0.1, and no dropout.
"""
    sections = [preamble.strip(), *(read_report(path) for path in REPORTS)]
    MARKDOWN_OUTPUT.write_text("\n\n\\newpage\n\n".join(sections) + "\n")


def build_pdf():
    subprocess.run(
        [
            "pandoc",
            MARKDOWN_OUTPUT.name,
            "--from=gfm+tex_math_dollars+raw_tex",
            "--pdf-engine=xelatex",
            "--toc",
            "--resource-path=.",
            "--output",
            PDF_OUTPUT.name,
        ],
        cwd=NOTES_DIRECTORY,
        check=True,
    )


def main():
    build_markdown()
    build_pdf()
    print(f"Wrote {MARKDOWN_OUTPUT}")
    print(f"Wrote {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
