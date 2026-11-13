"""
Helpers shared by the retrieval and generation evaluation tooling.

- text matching: normalize_for_match
- the question set: validate_question, load_questions
- run naming: parse_label, output_path, run_label, resolve_run_path
- provenance: git_revision

Results files are named <YYYY-MM-DD>_<label><suffix>: ".json" for
retrieval runs, ".jsonl" for generation runs and ".judged.jsonl" for the
judgments of a generation run. A label is looked up by exact match on that
name, so "gen_k3" never matches "<date>_gen_k3.judged.jsonl" when looking
for ".jsonl" files (its label there would be "gen_k3.judged").

No services or models are imported here.
"""

import argparse
import json
import re
import subprocess
import unicodedata
from datetime import date
from pathlib import Path
from typing import Dict, List


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = EVALUATION_DIR / "retrieval" / "questions.jsonl"

VALID_TYPES = {"direct", "paraphrased", "unanswerable"}
VALID_SUBTYPES = {"near_miss", "off_topic"}


# Text matching
# Whitespace plus every dash-like character PDF extraction produces:
# hyphen-minus, soft hyphen, U+2010..U+2015 (hyphens, en/em dash, bar),
# and the minus sign.
_IGNORED_CHARS = re.compile(r"[\s\-­‐-―−]+")


def normalize_for_match(text: str) -> str:
    """
    Normalize text for substring matching only (never for display).

    - NFKD turns ligatures into plain letters ("ﬁnite" -> "finite") and
      splits accented letters into base letter + combining mark
    - dropping combining marks folds accents, because OCR'd tables lose
      them ("Växjö" in the PDF is stored as "Vaxjo")
    - casefold makes matching case-insensitive
    - dropping whitespace and dashes makes line-break hyphenation
      ("right- skewed", "member- ship") and spacing differences match
    """
    decomposed = unicodedata.normalize("NFKD", text)
    unaccented = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _IGNORED_CHARS.sub("", unaccented.casefold())


# Questions file
def validate_question(q: Dict) -> List[str]:
    """Return a list of problems with one question row (empty if valid)."""
    errors = []

    for key in ("id", "type", "question"):
        if not q.get(key):
            errors.append(f"missing '{key}'")

    q_type = q.get("type")
    if q_type and q_type not in VALID_TYPES:
        errors.append(f"unknown type '{q_type}' (expected one of {sorted(VALID_TYPES)})")

    subtype = q.get("subtype")
    if subtype is not None:
        if q_type != "unanswerable":
            errors.append("'subtype' is only for unanswerable questions")
        elif subtype not in VALID_SUBTYPES:
            errors.append(f"unknown subtype '{subtype}' (expected one of {sorted(VALID_SUBTYPES)})")

    if q_type == "unanswerable":
        if q.get("doc") is not None or q.get("must_contain") is not None:
            errors.append("unanswerable questions must have doc and must_contain set to null")
    elif q_type in VALID_TYPES:
        if not q.get("doc"):
            errors.append("answerable questions need a 'doc'")
        must_contain = q.get("must_contain")
        if not must_contain or not normalize_for_match(must_contain):
            # An empty normalized string would match every chunk.
            errors.append("answerable questions need a non-empty 'must_contain'")

    return errors


def load_questions(path: Path) -> List[Dict]:
    """Load and validate questions.jsonl, raising ValueError listing every problem."""
    questions, errors, seen_ids = [], [], set()

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            q = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: invalid JSON ({exc.msg})")
            continue

        label = f"line {lineno} ({q.get('id', '?')})"
        errors.extend(f"{label}: {e}" for e in validate_question(q))

        if q.get("id") in seen_ids:
            errors.append(f"{label}: duplicate id")
        seen_ids.add(q.get("id"))
        questions.append(q)

    if errors:
        raise ValueError("Invalid questions file:\n  " + "\n  ".join(errors))
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


# Run naming
def parse_label(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("--label may only contain letters, digits, '_' and '-'")
    return value


def output_path(out_dir: Path, label: str, overwrite: bool, suffix: str = ".json") -> Path:
    """results/<today>_<label><suffix>, refusing to replace a non-empty file unless overwrite."""
    path = out_dir / f"{date.today().isoformat()}_{label}{suffix}"
    # Zero-byte files are placeholders and safe to fill in.
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite or use a different --label")
    return path


# Matches <YYYY-MM-DD>_<label> once the suffix is removed.
_DATED_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_(?P<label>.+)$")


def run_label(path: Path, suffix: str = ".json") -> str:
    """
    Label from the filename, not meta: files get renamed after the fact
    (e.g. a run saved as 'baseline' later kept as 'pre_label_fix'), and
    the filename is what tells runs apart.

    `suffix` is removed whole, so "<date>_x.judged.jsonl" has label
    "x.judged" for suffix ".jsonl" and "x" for suffix ".judged.jsonl".
    """
    name = path.name
    base = name[: -len(suffix)] if suffix and name.endswith(suffix) else path.stem
    match = _DATED_NAME.match(base)
    return match.group("label") if match else base


def resolve_run_path(arg: str, results_dir: Path, suffix: str = ".json") -> Path:
    """Accept a file path, or a label matching results/<date>_<label><suffix> exactly."""
    path = Path(arg)
    if path.is_file():
        return path

    matches = sorted(
        p for p in results_dir.glob(f"*{suffix}") if run_label(p, suffix) == arg
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No results file or label '{arg}' in {results_dir}")
    names = ", ".join(p.name for p in matches)
    raise ValueError(f"Label '{arg}' matches several files ({names}); pass a file path instead")


# Provenance
def git_revision() -> str:
    """Short commit SHA, with '-dirty' if anything is uncommitted (untracked files included)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=EVALUATION_DIR,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, cwd=EVALUATION_DIR,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
