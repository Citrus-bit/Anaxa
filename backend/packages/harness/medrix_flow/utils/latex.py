"""Utilities for preparing and compiling LaTeX artifacts."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlretrieve

_REMOTE_GRAPHICS_RE = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)(https?://[^}]+)(\})")
_CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff]")
_DOCUMENTCLASS_RE = re.compile(r"\\documentclass(?:\[(?P<options>[^\]]*)\])?\{(?P<class>[^{}]+)\}")
_CTEX_PACKAGE_RE = re.compile(r"\\usepackage(?:\[(?P<options>[^\]]*)\])?\{ctex\}")
_HYPERREF_PACKAGE_RE = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{hyperref\}")
_BEGIN_DOCUMENT_RE = re.compile(r"\\begin\s*\{\s*document\s*\}")
_CTEX_CLASSES = {"ctexart", "ctexrep", "ctexbook", "ctexbeamer"}
_TOUNICODE_SPECIAL = r"\AtBeginDvi{\special{pdf:tounicode UTF8-UCS2}}"
_HIDDEN_HYPERREF_SETUP = r"\hypersetup{hidelinks,pdfborder={0 0 0}}"
_SUBSCRIPT_CHARS = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "₊": "+",
    "₋": "-",
    "₌": "=",
    "₍": "(",
    "₎": ")",
    "ₐ": "a",
    "ₑ": "e",
    "ₕ": "h",
    "ᵢ": "i",
    "ⱼ": "j",
    "ₖ": "k",
    "ₗ": "l",
    "ₘ": "m",
    "ₙ": "n",
    "ₒ": "o",
    "ₚ": "p",
    "ᵣ": "r",
    "ₛ": "s",
    "ₜ": "t",
    "ᵤ": "u",
    "ᵥ": "v",
    "ₓ": "x",
}
_SUPERSCRIPT_CHARS = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁼": "=",
    "⁽": "(",
    "⁾": ")",
    "ᵃ": "a",
    "ᵇ": "b",
    "ᶜ": "c",
    "ᵈ": "d",
    "ᵉ": "e",
    "ᶠ": "f",
    "ᵍ": "g",
    "ʰ": "h",
    "ⁱ": "i",
    "ʲ": "j",
    "ᵏ": "k",
    "ˡ": "l",
    "ᵐ": "m",
    "ⁿ": "n",
    "ᵒ": "o",
    "ᵖ": "p",
    "ʳ": "r",
    "ˢ": "s",
    "ᵗ": "t",
    "ᵘ": "u",
    "ᵛ": "v",
    "ʷ": "w",
    "ˣ": "x",
    "ʸ": "y",
    "ᶻ": "z",
}
_SUBSCRIPT_CLASS = "".join(map(re.escape, _SUBSCRIPT_CHARS))
_SUPERSCRIPT_CLASS = "".join(map(re.escape, _SUPERSCRIPT_CHARS))
_UNICODE_MATH_TOKEN_RE = re.compile(
    rf"(?<![$\\])([A-Za-z0-9]+)([{_SUBSCRIPT_CLASS}{_SUPERSCRIPT_CLASS}]+)"
)
_ADJACENT_INLINE_MATH_RATIO_RE = re.compile(r"\$([^$]+)\$/\$([^$]+)\$")


@dataclass(frozen=True)
class PdfTextAuditResult:
    """Result for checking whether compiled PDF text is extractable."""

    status: str
    pdf_path: str
    cid_marker_count: int
    text_length: int
    cid_marker_ratio: float
    pages_checked: int
    violations: list[str]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "pdf_path": self.pdf_path,
            "cid_marker_count": self.cid_marker_count,
            "text_length": self.text_length,
            "cid_marker_ratio": self.cid_marker_ratio,
            "pages_checked": self.pages_checked,
            "violations": self.violations,
            "error": self.error,
        }


def _translate_script_chars(chars: str, mapping: dict[str, str]) -> str:
    return "".join(mapping[ch] for ch in chars if ch in mapping)


def _replace_unicode_math_tokens(source: str) -> str:
    def repl(match: re.Match[str]) -> str:
        base = match.group(1)
        scripts = match.group(2)
        pieces: list[str] = [base]
        idx = 0
        while idx < len(scripts):
            char = scripts[idx]
            if char in _SUBSCRIPT_CHARS:
                start = idx
                while idx < len(scripts) and scripts[idx] in _SUBSCRIPT_CHARS:
                    idx += 1
                translated = _translate_script_chars(scripts[start:idx], _SUBSCRIPT_CHARS)
                pieces.append(f"_{{{translated}}}")
                continue
            if char in _SUPERSCRIPT_CHARS:
                start = idx
                while idx < len(scripts) and scripts[idx] in _SUPERSCRIPT_CHARS:
                    idx += 1
                translated = _translate_script_chars(scripts[start:idx], _SUPERSCRIPT_CHARS)
                pieces.append(f"^{{{translated}}}")
                continue
            idx += 1
        return f"${''.join(pieces)}$"

    source = _UNICODE_MATH_TOKEN_RE.sub(repl, source)
    previous = None
    while previous != source:
        previous = source
        source = _ADJACENT_INLINE_MATH_RATIO_RE.sub(r"$\1/\2$", source)
    return source


def _sanitize_latex_source(source: str) -> str:
    source = _replace_unicode_math_tokens(source)
    source = _ensure_cjk_pdf_text_support(source)
    if "\\usepackage{subfig}" not in source and "\\subfloat" in source:
        source = source.replace("\\usepackage{graphicx}", "\\usepackage{graphicx}\n\\usepackage{subfig}", 1)
    if "\\usepackage{amsmath}" not in source and ("$" in source or "\\[" in source or "\\begin{equation" in source):
        source = source.replace("\\usepackage{graphicx}", "\\usepackage{graphicx}\n\\usepackage{amsmath}", 1)
    source = _ensure_hidden_hyperref_borders(source)
    return source


def _ensure_hidden_hyperref_borders(source: str) -> str:
    if not _HYPERREF_PACKAGE_RE.search(source) or _HIDDEN_HYPERREF_SETUP in source:
        return source

    begin_document_match = _BEGIN_DOCUMENT_RE.search(source)
    insert_at = begin_document_match.start() if begin_document_match is not None else len(source)
    prefix = "" if insert_at == 0 or source[insert_at - 1] == "\n" else "\n"
    return source[:insert_at] + f"{prefix}{_HIDDEN_HYPERREF_SETUP}\n" + source[insert_at:]


def _ensure_cjk_pdf_text_support(source: str) -> str:
    if not _CJK_CHAR_RE.search(source):
        return source

    source = _ensure_ctex_fontset(source)
    if "pdf:tounicode UTF8-UCS2" in source:
        return source

    documentclass_match = _DOCUMENTCLASS_RE.search(source)
    if documentclass_match is not None:
        insert_at = documentclass_match.end()
        return source[:insert_at] + "\n" + _TOUNICODE_SPECIAL + source[insert_at:]
    return _TOUNICODE_SPECIAL + "\n" + source


def _ensure_ctex_fontset(source: str) -> str:
    documentclass_match = _DOCUMENTCLASS_RE.search(source)
    if documentclass_match is not None and documentclass_match.group("class") in _CTEX_CLASSES:
        options = documentclass_match.group("options") or ""
        if "fontset=" not in options:
            updated_options = _append_latex_option(options, "fontset=fandol")
            source = (
                source[: documentclass_match.start()]
                + rf"\documentclass[{updated_options}]{{{documentclass_match.group('class')}}}"
                + source[documentclass_match.end() :]
            )
        return source

    package_match = _CTEX_PACKAGE_RE.search(source)
    if package_match is not None:
        options = package_match.group("options") or ""
        if "fontset=" in options:
            return source
        updated_options = _append_latex_option(options, "fontset=fandol")
        return source[: package_match.start()] + rf"\usepackage[{updated_options}]{{ctex}}" + source[package_match.end() :]

    if "\\usepackage{xeCJK}" in source or "\\usepackage[AutoFakeBold]{xeCJK}" in source:
        return source

    documentclass_match = _DOCUMENTCLASS_RE.search(source)
    if documentclass_match is not None:
        insert_at = documentclass_match.end()
        return source[:insert_at] + "\n" + r"\usepackage[UTF8,fontset=fandol]{ctex}" + source[insert_at:]
    return source


def _append_latex_option(options: str, option: str) -> str:
    cleaned = options.strip()
    if not cleaned:
        return option
    return f"{cleaned},{option}"


def _download_remote_graphics(source: str, workdir: Path) -> str:
    workdir.mkdir(parents=True, exist_ok=True)

    def repl(match: re.Match[str]) -> str:
        prefix, url, suffix = match.groups()
        parsed = urlparse(url)
        filename = Path(parsed.path).name or "figure"
        local_path = workdir / filename
        if not local_path.suffix:
            local_path = local_path.with_suffix(".png")
        try:
            urlretrieve(url, local_path)
            return f"{prefix}{local_path.name}{suffix}"
        except Exception:
            return match.group(0)

    return _REMOTE_GRAPHICS_RE.sub(repl, source)


def prepare_latex_preview(tex_path: Path, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or tex_path.parent / ".latex-preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    source = tex_path.read_text(encoding="utf-8")
    source = _sanitize_latex_source(source)
    source = _download_remote_graphics(source, output_dir)
    prepared_path = output_dir / tex_path.name
    prepared_path.write_text(source, encoding="utf-8")
    return prepared_path


def compile_latex_to_pdf(tex_path: Path, output_dir: Path | None = None) -> Path:
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise FileNotFoundError("tectonic not found in PATH")

    output_dir = output_dir or tex_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [tectonic, "-k", "--outdir", str(output_dir), str(tex_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "tectonic compilation failed").strip())

    pdf_path = output_dir / tex_path.with_suffix(".pdf").name
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not produced: {pdf_path}")
    return pdf_path


def audit_pdf_text_extractability(
    pdf_path: Path,
    *,
    max_cid_markers: int = 20,
    max_cid_marker_ratio: float = 0.01,
) -> PdfTextAuditResult:
    """Fail when extracted PDF text is dominated by PDF CID placeholders."""

    text, pages_checked, error = _extract_pdf_text(pdf_path)
    cid_marker_count = text.count("(cid:")
    text_length = len(text)
    cid_marker_ratio = cid_marker_count / max(text_length, 1)
    violations: list[str] = []
    if cid_marker_count > max_cid_markers and cid_marker_ratio > max_cid_marker_ratio:
        violations.append(
            f"PDF text extraction contains too many CID placeholders: "
            f"{cid_marker_count} markers, ratio {cid_marker_ratio:.3f}."
        )

    return PdfTextAuditResult(
        status="fail" if violations else "pass",
        pdf_path=str(pdf_path),
        cid_marker_count=cid_marker_count,
        text_length=text_length,
        cid_marker_ratio=cid_marker_ratio,
        pages_checked=pages_checked,
        violations=violations,
        error=error,
    )


def write_pdf_text_audit(result: PdfTextAuditResult, output_path: Path) -> Path:
    output_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _extract_pdf_text(pdf_path: Path) -> tuple[str, int, str | None]:
    if not pdf_path.exists():
        return "", 0, f"{pdf_path} does not exist"
    try:
        import pdfplumber
    except Exception as exc:
        return "", 0, f"pdfplumber unavailable: {exc}"

    try:
        texts: list[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
            return "\n".join(texts), len(pdf.pages), None
    except Exception as exc:
        return "", 0, f"pdf text extraction failed: {exc}"
