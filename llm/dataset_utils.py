from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_CANDIDATES = (
    ROOT_DIR / "data" / "raw" / "sign_language_conflict.xlsx",
    ROOT_DIR / "data" / "raw" / "sign_language_conflict.csv",
    ROOT_DIR / "data" / "raw" / "FinalSheet2.xlsx",
    ROOT_DIR / "data" / "raw" / "FinalSheet2.csv",
)


def _normalize_column_name(value: object) -> str:
    text = str(value or "").strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def resolve_dataset_path(input_file: str | Path | None = None) -> Path:
    if input_file is not None:
        path = Path(input_file)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.exists():
            return path
        if (ROOT_DIR / path).exists():
            return ROOT_DIR / path
        raise FileNotFoundError(f"Dataset file not found: {path}")

    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No dataset file found. Expected one of: "
        + ", ".join(str(path) for path in DEFAULT_DATASET_CANDIDATES)
    )


def load_dataset_frame(input_file: str | Path | None = None, sheet_name: str | None = None):
    path = resolve_dataset_path(input_file)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=(suffix == ".jsonl"))
    if suffix in {".xls", ".xlsx"}:
        if sheet_name:
            return pd.read_excel(path, sheet_name=sheet_name)
        excel = pd.ExcelFile(path)
        if len(excel.sheet_names) == 1:
            return pd.read_excel(path, sheet_name=0)
        raise ValueError(f"Excel file contains multiple sheets: {excel.sheet_names}. Specify --sheet_name.")

    raise ValueError("Unsupported input file format. Use CSV, JSON, JSONL, or Excel.")


def resolve_column_name(
    df: pd.DataFrame,
    preferred: str | None = None,
    aliases: Iterable[str] | None = None,
    disallowed: Iterable[str] | None = None,
) -> str:
    if preferred and preferred in df.columns:
        return preferred

    if aliases:
        alias_lookup = {_normalize_column_name(alias): alias for alias in aliases if alias}
        for column in df.columns:
            normalized = _normalize_column_name(column)
            if normalized in alias_lookup:
                return column

    disallowed_set = {_normalize_column_name(name) for name in (disallowed or [])}
    for column in df.columns:
        normalized = _normalize_column_name(column)
        if normalized in disallowed_set:
            continue
        return column

    if len(df.columns) == 1:
        return df.columns[0]

    raise ValueError(f"Unable to infer a suitable column from: {list(df.columns)}")


def resolve_text_column(df: pd.DataFrame, preferred: str | None = None) -> str:
    aliases = [
        "sentences",
        "sentence",
        "text",
        "content",
        "description",
        "transcript",
        "input",
        "statement",
        "statementtext",
        "names",
        "prompt",
    ]
    return resolve_column_name(df, preferred=preferred, aliases=aliases, disallowed=["category", "class", "label", "tag"])


def resolve_complaint_column(df: pd.DataFrame, preferred: str | None = None) -> str | None:
    aliases = [
        "complaint",
        "output",
        "legalcomplaint",
        "fir",
        "firstinformationreport",
        "summary",
        "response",
    ]
    if preferred and preferred in df.columns:
        return preferred
    for column in df.columns:
        normalized = _normalize_column_name(column)
        if normalized in {_normalize_column_name(alias) for alias in aliases}:
            return column
    return None


def build_synthetic_complaint(text: str, category: str | None = None) -> str:
    text = str(text or "").strip().rstrip(".")
    if not text:
        return ""

    category_line = f"ঘটনার ধরন: {category}." if category else "ঘটনার ধরন: __________"
    return f"""ফৌজদারি অভিযোগ / FIR

মামলা নং: __________
ঘটনার তারিখ: __________
ঘটনার সময়: __________
ঘটনার স্থান: __________

বাদীর নাম: __________
বাদীর ঠিকানা: __________

ঘটনার বিবরণ:
দায়েরকারী অভিযোগ করেন যে {text}। {category_line}

অভিযুক্ত ব্যক্তি বা প্রতিষ্ঠান: __________
প্রমাণের বর্ণনা: __________

বাদী এই ঘটনার বিষয়ে দ্রুত আইনানুগ ব্যবস্থা গ্রহণের জন্য বিনীত অনুরোধ করেন।

স্বাক্ষর: __________
তারিখ: __________"""
