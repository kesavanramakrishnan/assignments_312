"""Combine the four Assignment 2 problem notes into one compact PDF (20-page limit)."""
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).parents[1]
NOTES = ROOT / "notes" / "a2"
BUILD = ROOT / "tmp" / "a2_build"
OUTPUT = NOTES / "a2_submission.pdf"
REPORTS = (("p1", "p1_learning_rate_scaling.md"), ("p2", "p2_lr_wd_scaling.md"),
           ("p3", "p3_batch_size_scaling.md"), ("p4", "p4_mup_transfer.md"))
TEXT_WIDTH_IN = 7.3
MAX_HEIGHT_IN = 2.55
GRID_MAX_HEIGHT_IN = 2.85  # multi-row grids (wide canvases with aspect below 2) get more height to stay legible
SINGLE_MAX_HEIGHT_IN = 2.5
MAX_DPI = 200
# Per-figure physical width overrides (inches); everything else fits the text width within MAX_HEIGHT_IN.
WIDTHS = {"p42a_omega.png": 3.3}
# STIXGeneral lacks Unicode subscripts and a few superscripts; write them as plain text.
SUBSTITUTIONS = (("ᵀ", "^T"), ("ᵢⱼ", "_ij"), ("ᵢ", "_i"), ("ⱼ", "_j"), ("⁻⁸", "^-8"), ("⁻", "^-"), ("⁴", "^4"), ("⁸", "^8"),
                 *((chr(0x2080 + d), str(d)) for d in range(10)))
HEADER = r"""\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\footnotesize}
\usepackage{enumitem}\setlist{nosep,leftmargin=1.2em}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{}{0pt}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0pt}{}
\titleformat{\subsubsection}{\small\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{8pt}{3pt}
\titlespacing*{\subsection}{0pt}{6pt}{2pt}
\titlespacing*{\subsubsection}{0pt}{4pt}{1pt}
\setlength{\parskip}{2.5pt}
\setlength{\LTpre}{2pt}\setlength{\LTpost}{2pt}
\setlength{\tabcolsep}{3pt}
\pagenumbering{arabic}
"""


def resized_image(path, name):
    """Copy a figure with its dpi set so its natural size is the physical width we want on the page."""
    image = Image.open(path)
    aspect = image.width / image.height
    max_height = MAX_HEIGHT_IN if aspect >= 2 else GRID_MAX_HEIGHT_IN if image.width >= 1100 else SINGLE_MAX_HEIGHT_IN
    width = WIDTHS.get(path.name, min(TEXT_WIDTH_IN, max_height * aspect))
    dpi = image.width / width
    if dpi > MAX_DPI:
        image = image.resize((round(width * MAX_DPI), round(width * MAX_DPI / aspect)), Image.LANCZOS)
        dpi = MAX_DPI
    out = BUILD / name
    image.save(out, dpi=(dpi, dpi))
    return out


def prepare(problem, filename):
    text = (NOTES / problem / filename).read_text()
    for old, new in SUBSTITUTIONS:
        text = text.replace(old, new)

    def replace(match):
        alt, target = match.group(1), match.group(2)
        return f"![{alt}]({resized_image(NOTES / problem / target, f'{problem}_{target}')})"

    return re.sub(r"!\[([^\]]*)\]\(([^)/]+\.png)\)", replace, text)


def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    sections = ["**Assignment 2: Hyperparameter Scaling** (Kesavan Ramakrishnan)"]
    sections += [prepare(*report) for report in REPORTS]
    markdown = BUILD / "a2.md"
    markdown.write_text("\n\n".join(sections) + "\n")
    (BUILD / "header.tex").write_text(HEADER)
    subprocess.run(
        ["pandoc", markdown.name, "--from=gfm", "--pdf-engine=lualatex",
         "-V", "documentclass=extarticle", "-V", "fontsize=9pt", "-V", "geometry:margin=0.6in",
         "-V", "mainfont=STIXGeneral",
         "-H", "header.tex", "--output", str(OUTPUT)],
        cwd=BUILD, check=True)
    print(f"Wrote {OUTPUT}")
    pages = subprocess.run(["pdfinfo", str(OUTPUT)], capture_output=True, text=True).stdout
    print(next((line for line in pages.splitlines() if line.startswith("Pages")), "pages unknown"))


if __name__ == "__main__":
    sys.exit(build())
