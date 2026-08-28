"""Tests for one-shot manuscript export."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

manuscript_export_tool_module = importlib.import_module("medrix_flow.tools.builtins.manuscript_export_tool")
manuscript_quality_module = importlib.import_module("medrix_flow.utils.manuscript_quality")
latex_module = importlib.import_module("medrix_flow.utils.latex")


def _make_runtime(outputs_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        state={"thread_data": {"outputs_path": outputs_path}},
        context={"thread_id": "thread-1"},
    )


def _tex(citation: str = r"\cite{smith2024}") -> str:
    return rf"""
\documentclass{{article}}
\begin{{document}}
Supported claim {citation}.
\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}
"""


def _bib() -> str:
    return """
@article{smith2024,
  title = {A paper},
  author = {Smith, Jane},
  journal = {Journal},
  year = {2024}
}
"""


def _many_bib(total: int = 15, recent: int = 8) -> str:
    entries = []
    for index in range(1, total + 1):
        year = 2024 if index <= recent else 2005
        entries.append(
            f"""
@article{{ref{index},
  title = {{Verified modeling reference {index}}},
  author = {{Author, Example}},
  journal = {{Journal of Modeling}},
  year = {{{year}}}
}}
"""
        )
    return "\n".join(entries)


def _academic_bib(total: int = 30) -> str:
    entries = []
    for index in range(1, total + 1):
        year = 2024 if index <= 20 else 2010
        entries.append(
            f"""
@article{{acad{index},
  title = {{Academic reference {index}}},
  author = {{Researcher, Example and Author, Second}},
  journal = {{Journal of Virtual Cell}},
  year = {{{year}}},
  url = {{https://example.test/acad{index}}}
}}
"""
        )
    return "\n".join(entries)


def _formal_chinese_review_tex(keys: list[str]) -> str:
    citations = ",".join(keys)
    paragraphs = "\n\n".join(rf"本文综述虚拟细胞基础模型中的图学习、扰动预测和多组学整合问题，并结合可验证文献讨论模型局限与改进方向 \citep{{{key}}}。" for key in keys)
    return rf"""
\documentclass[UTF8,12pt]{{ctexart}}
\usepackage{{natbib}}
\title{{虚拟细胞基础模型研究综述}}
\begin{{document}}
\maketitle
\begin{{abstract}}
本文是一篇中文综述，系统讨论模型进展、局限和参考文献质量 \citep{{{citations}}}。
\end{{abstract}}
{paragraphs}
\bibliographystyle{{plainnat}}
\bibliography{{references}}
\end{{document}}
"""


def _short_math_modeling_tex() -> str:
    return r"""
\documentclass{article}
\begin{document}
\title{Mathematical Modeling Competition Manuscript}
\maketitle
This mathematical modeling competition manuscript is too short but cites one paper \cite{ref1}.
\begin{figure}\caption{Only one schematic.}\end{figure}
\begin{table}\caption{Only one table.}\begin{tabular}{cc}a&b\end{tabular}\end{table}
\bibliographystyle{plain}
\bibliography{references}
\end{document}
"""


def _long_math_modeling_tex() -> str:
    cite_cycle = [f"ref{index}" for index in range(1, 16)]
    paragraphs = []
    sentence = (
        "This formal mathematical modeling competition paper develops a complete probabilistic model, "
        "algorithmic workflow, simulation design, sensitivity analysis, robustness check, and diagnostic "
        "interpretation for a contest-style decision problem. "
    )
    for index in range(80):
        paragraphs.append((sentence * 3) + rf"The claim is supported by \cite{{{cite_cycle[index % len(cite_cycle)]}}}.")
    figures = "\n".join(rf"\begin{{figure}}\caption{{Diagnostic figure {index}.}}\end{{figure}}" for index in range(1, 6))
    tables = "\n".join(rf"\begin{{table}}\caption{{Result table {index}.}}\begin{{tabular}}{{cc}}a&b\end{{tabular}}\end{{table}}" for index in range(1, 6))
    return rf"""
\documentclass{{article}}
\begin{{document}}
\title{{Mathematical Modeling Competition Manuscript}}
\maketitle
{figures}
{tables}

{chr(10).join(paragraphs)}

\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}
"""


def _stub_latex_compile(monkeypatch, pdf_bytes: bytes = b"%PDF-1.4") -> None:
    def prepare(tex_path: Path) -> Path:
        preview_dir = tex_path.parent / ".latex-preview"
        preview_dir.mkdir()
        preview_path = preview_dir / tex_path.name
        preview_path.write_text(tex_path.read_text(encoding="utf-8"), encoding="utf-8")
        return preview_path

    def compile_pdf(tex_path: Path, output_dir: Path | None = None) -> Path:
        pdf_path = (output_dir or tex_path.parent) / tex_path.with_suffix(".pdf").name
        pdf_path.write_bytes(pdf_bytes)
        return pdf_path

    monkeypatch.setattr(manuscript_export_tool_module, "prepare_latex_preview", prepare)
    monkeypatch.setattr(manuscript_export_tool_module, "compile_latex_to_pdf", compile_pdf)


def test_manuscript_quality_counts_visible_latex_command_arguments():
    tex = r"""
\documentclass{article}
\begin{document}
\section{Visible Section}
Plain body \textbf{Bold Claim}.
\begin{figure}\caption{Diagnostic Caption}\end{figure}
This citation should not count \cite{hidden2024}.
\end{document}
"""

    count = manuscript_quality_module._latex_word_count(tex)

    assert count >= 8
    assert count < 16


def test_latex_sanitizer_adds_cjk_tounicode_special():
    source = r"""
\documentclass[UTF8,12pt]{ctexart}
\begin{document}
中文综述
\end{document}
"""

    sanitized = latex_module._sanitize_latex_source(source)

    assert r"\documentclass[UTF8,12pt,fontset=fandol]{ctexart}" in sanitized
    assert r"\AtBeginDvi{\special{pdf:tounicode UTF8-UCS2}}" in sanitized


def test_manuscript_quality_auto_detects_modeling_competition_aliases():
    assert (
        manuscript_quality_module.resolve_quality_profile(
            r"\begin{document}MCM/ICM contest manuscript\end{document}",
            "auto",
        )
        == "math_modeling_competition"
    )
    assert (
        manuscript_quality_module.resolve_quality_profile(
            r"\begin{document}数学模型竞赛论文\end{document}",
            "auto",
        )
        == "math_modeling_competition"
    )
    assert (
        manuscript_quality_module.resolve_quality_profile(
            r"\begin{document}MCMC sampling manuscript\end{document}",
            "auto",
        )
        is None
    )


def test_manuscript_quality_reports_unavailable_pdf_page_count(tmp_path, monkeypatch):
    pdf_path = tmp_path / "bad.pdf"
    pdf_path.write_bytes(b"not a valid pdf")
    monkeypatch.setattr(
        manuscript_quality_module,
        "_pdf_page_count",
        lambda _path: (_ for _ in ()).throw(RuntimeError("reader failed")),
    )
    monkeypatch.setattr(
        manuscript_quality_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("pdfinfo missing")),
    )

    result = manuscript_quality_module.audit_manuscript_quality(
        tex_content=_long_math_modeling_tex(),
        bibtex_content=_many_bib(total=15, recent=8),
        quality_profile="math_modeling_competition",
        pdf_path=pdf_path,
    )

    assert result.status == "fail"
    assert result.metrics["page_count"] is None
    assert result.metrics["page_count_status"] == "unavailable"
    assert "reader failed" in result.metrics["page_count_error"]
    assert any("page_count unavailable" in violation for violation in result.violations)


def test_manuscript_export_writes_bundle_and_pdf(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)
    _stub_latex_compile(monkeypatch)

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_tex(),
        bibtex_content=_bib(),
        filename_stem="paper",
        tool_call_id="tc-1",
    )

    assert result.update["artifacts"] == [
        "/mnt/user-data/outputs/paper.pdf",
        "/mnt/user-data/outputs/paper.tex",
        "/mnt/user-data/outputs/references.bib",
        "/mnt/user-data/outputs/citation_audit.json",
    ]
    assert (outputs_dir / "paper.pdf").read_bytes() == b"%PDF-1.4"
    assert (outputs_dir / "paper.tex").exists()
    assert (outputs_dir / "references.bib").exists()
    assert json.loads((outputs_dir / "citation_audit.json").read_text(encoding="utf-8"))["status"] == "pass"
    assert result.update["messages"][0].content.startswith("PASS:")


def test_manuscript_export_blocks_missing_citation_key(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)
    monkeypatch.setattr(
        manuscript_export_tool_module,
        "compile_latex_to_pdf",
        lambda _tex_path, _output_dir=None: (_ for _ in ()).throw(AssertionError("should not compile")),
    )

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_tex(r"\cite{missing2025}"),
        bibtex_content=_bib(),
        filename_stem="paper",
        tool_call_id="tc-1",
    )

    assert result.update["artifacts"] == [
        "/mnt/user-data/outputs/citation_audit.json",
        "/mnt/user-data/outputs/paper.tex",
        "/mnt/user-data/outputs/references.bib",
    ]
    assert not (outputs_dir / "paper.pdf").exists()
    assert "citation audit blocked PDF generation" in result.update["messages"][0].content
    audit = json.loads((outputs_dir / "citation_audit.json").read_text(encoding="utf-8"))
    assert audit["missing_keys"] == ["missing2025"]


def test_manuscript_export_blocks_nocite_all_by_default(tmp_path):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_tex(r"\nocite{*}"),
        bibtex_content=_bib(),
        filename_stem="paper",
        tool_call_id="tc-1",
    )

    assert not (outputs_dir / "paper.pdf").exists()
    assert r"\nocite{*} is not allowed" in result.update["messages"][0].content


def test_manuscript_export_blocks_unsupported_claim_map(tmp_path):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)
    claim_map = {
        "claims": [
            {
                "claim": "The method is universally superior.",
                "support_status": "unsupported",
                "evidence": [],
            }
        ]
    }

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_tex(r"The method is universally superior. \cite{smith2024}"),
        bibtex_content=_bib(),
        claim_map_json=json.dumps(claim_map),
        filename_stem="paper",
        tool_call_id="tc-1",
    )

    audit = json.loads((outputs_dir / "citation_audit.json").read_text(encoding="utf-8"))
    assert audit["unsupported_claims"] == ["The method is universally superior."]
    assert not (outputs_dir / "paper.pdf").exists()
    assert "Unsupported claims: 1" in result.update["messages"][0].content


def test_manuscript_export_allows_stale_unsupported_claim_map(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)
    _stub_latex_compile(monkeypatch)
    claim_map = {
        "claims": [
            {
                "claim": "The method is universally superior.",
                "support_status": "unsupported",
                "evidence": [],
            }
        ]
    }

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_tex(),
        bibtex_content=_bib(),
        claim_map_json=json.dumps(claim_map),
        filename_stem="paper",
        tool_call_id="tc-1",
    )

    audit = json.loads((outputs_dir / "citation_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    assert audit["unsupported_claims"] == []
    assert audit["stale_claims"] == ["The method is universally superior."]
    assert (outputs_dir / "paper.pdf").exists()
    assert "stale claim_map" in result.update["messages"][0].content


def test_manuscript_export_blocks_literature_only_experimental_claim(tmp_path):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)
    claim_map = {
        "claims": [
            {
                "claim": "The method outperforms the baseline on the benchmark.",
                "support_status": "supported_by_literature",
                "evidence": ["smith2024"],
            }
        ]
    }

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_tex(r"The method outperforms the baseline on the benchmark. \cite{smith2024}"),
        bibtex_content=_bib(),
        claim_map_json=json.dumps(claim_map),
        filename_stem="paper",
        tool_call_id="tc-1",
    )

    audit = json.loads((outputs_dir / "citation_audit.json").read_text(encoding="utf-8"))
    assert audit["unsupported_claims"] == ["The method outperforms the baseline on the benchmark."]
    assert not (outputs_dir / "paper.pdf").exists()
    assert "Unsupported claims: 1" in result.update["messages"][0].content


def test_manuscript_export_blocks_simulation_claim_without_assumptions(tmp_path):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)
    claim_map = {
        "claims": [
            {
                "claim": "The simulated experiment outperforms the baseline on F1.",
                "support_status": "supported_by_simulation",
                "evidence": ["synthetic_results.json"],
            }
        ]
    }

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_tex(r"The simulated experiment outperforms the baseline on F1. \cite{smith2024}"),
        bibtex_content=_bib(),
        claim_map_json=json.dumps(claim_map),
        filename_stem="paper",
        tool_call_id="tc-1",
    )

    audit = json.loads((outputs_dir / "citation_audit.json").read_text(encoding="utf-8"))
    assert audit["unsupported_claims"] == ["The simulated experiment outperforms the baseline on F1."]
    assert not (outputs_dir / "paper.pdf").exists()
    assert "Unsupported claims: 1" in result.update["messages"][0].content


def test_manuscript_export_allows_simulation_claim_with_assumptions(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)
    _stub_latex_compile(monkeypatch)
    claim_map = {
        "simulation_disclosure": "Synthetic personal experiment data were generated from documented assumptions.",
        "claims": [
            {
                "claim": "The simulated experiment produced F1=0.82.",
                "support_status": "supported_by_simulation",
                "evidence_type": "simulation",
                "evidence": ["synthetic_results.json", "simulation_assumptions.json"],
                "simulation_assumptions_path": "simulation_assumptions.json",
            }
        ],
    }

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_tex(),
        bibtex_content=_bib(),
        claim_map_json=json.dumps(claim_map),
        filename_stem="paper",
        tool_call_id="tc-1",
    )

    assert result.update["artifacts"][0] == "/mnt/user-data/outputs/paper.pdf"
    audit = json.loads((outputs_dir / "citation_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    assert audit["unsupported_claims"] == []


def test_manuscript_export_blocks_author_process_notes(tmp_path):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_tex(r"\cite{smith2024}") + "\n% bibliography keys are synchronized\n",
        bibtex_content=_bib(),
        filename_stem="paper",
        tool_call_id="tc-1",
    )

    audit = json.loads((outputs_dir / "citation_audit.json").read_text(encoding="utf-8"))
    assert audit["author_notes"] == ["bibliography keys are synchronized"]
    assert not (outputs_dir / "paper.pdf").exists()
    assert "Author/tool process notes remain" in result.update["messages"][0].content


def test_manuscript_export_blocks_short_formal_review_bibliography_with_academic_corpus(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    academic_dir = outputs_dir / "academic-research" / "virtual-cell"
    academic_dir.mkdir(parents=True)
    (academic_dir / "references.bib").write_text(_academic_bib(total=30), encoding="utf-8")
    (academic_dir / "references.md").write_text("# References\n", encoding="utf-8")
    (academic_dir / "evidence_map.json").write_text("{}\n", encoding="utf-8")
    (academic_dir / "retrieval_audit.json").write_text(
        json.dumps({"canonical_reference_count": 30}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        manuscript_export_tool_module,
        "compile_latex_to_pdf",
        lambda _tex_path, _output_dir=None: (_ for _ in ()).throw(AssertionError("should not compile")),
    )

    short_bib = """
@article{manual1,
  title = {Manual reference 1},
  journal = {Journal},
  year = {2024}
}
@article{manual2,
  title = {Manual reference 2},
  journal = {Journal},
  year = {2024}
}
@article{manual3,
  title = {Manual reference 3},
  journal = {Journal},
  year = {2024}
}
@article{manual4,
  title = {Manual reference 4},
  journal = {Journal},
  year = {2024}
}
@article{manual5,
  title = {Manual reference 5},
  journal = {Journal},
  year = {2024}
}
"""
    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_formal_chinese_review_tex(["manual1", "manual2", "manual3", "manual4", "manual5"]),
        bibtex_content=short_bib,
        filename_stem="review",
        tool_call_id="tc-review",
    )

    assert not (outputs_dir / "review.pdf").exists()
    assert "citation audit blocked PDF generation" in result.update["messages"][0].content
    audit = json.loads((outputs_dir / "citation_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "fail"
    assert audit["reference_count"] == 35
    assert audit["cited_reference_count"] == 5
    assert audit["metadata"]["reference_resolution"]["mode"] == "merged_academic_references"
    assert audit["metadata"]["reference_resolution"]["academic_reference_count"] == 30
    assert any("Inline cited reference count below threshold: 5/15" in item for item in audit["violations"])
    assert any("manual1 missing author" in item for item in audit["invalid_references"])
    assert any("detached from academic_research references" in item for item in audit["violations"])


def test_manuscript_export_reuses_complete_academic_bib_for_formal_review(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    academic_dir = outputs_dir / "academic-research" / "virtual-cell"
    academic_dir.mkdir(parents=True)
    academic_bib = _academic_bib(total=30)
    (academic_dir / "references.bib").write_text(academic_bib, encoding="utf-8")
    (academic_dir / "retrieval_audit.json").write_text(
        json.dumps({"canonical_reference_count": 30}) + "\n",
        encoding="utf-8",
    )
    _stub_latex_compile(monkeypatch)
    monkeypatch.setattr(
        manuscript_export_tool_module,
        "audit_pdf_text_extractability",
        lambda path: manuscript_export_tool_module.PdfTextAuditResult(
            status="pass",
            pdf_path=str(path),
            cid_marker_count=0,
            text_length=200,
            cid_marker_ratio=0.0,
            pages_checked=1,
            violations=[],
        ),
    )

    cited_keys = [f"acad{index}" for index in range(1, 16)]
    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_formal_chinese_review_tex(cited_keys),
        bibtex_content="@article{manual1, title={Manual}, author={Author, A}, year={2024}}\n",
        filename_stem="review",
        tool_call_id="tc-review",
    )

    assert result.update["artifacts"] == [
        "/mnt/user-data/outputs/review.pdf",
        "/mnt/user-data/outputs/review.tex",
        "/mnt/user-data/outputs/references.bib",
        "/mnt/user-data/outputs/citation_audit.json",
        "/mnt/user-data/outputs/pdf_text_audit.json",
    ]
    assert (outputs_dir / "references.bib").read_text(encoding="utf-8") == academic_bib
    audit = json.loads((outputs_dir / "citation_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    assert audit["reference_count"] == 30
    assert audit["cited_reference_count"] == 15
    assert audit["recent_reference_count"] >= 8
    assert audit["metadata"]["reference_resolution"]["mode"] == "academic_references"


def test_manuscript_export_blocks_pdf_with_cid_text(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    academic_dir = outputs_dir / "academic-research" / "virtual-cell"
    academic_dir.mkdir(parents=True)
    (academic_dir / "references.bib").write_text(_academic_bib(total=30), encoding="utf-8")
    (academic_dir / "retrieval_audit.json").write_text(
        json.dumps({"canonical_reference_count": 30}) + "\n",
        encoding="utf-8",
    )
    _stub_latex_compile(monkeypatch)
    monkeypatch.setattr(
        manuscript_export_tool_module,
        "audit_pdf_text_extractability",
        lambda path: manuscript_export_tool_module.PdfTextAuditResult(
            status="fail",
            pdf_path=str(path),
            cid_marker_count=50,
            text_length=500,
            cid_marker_ratio=0.1,
            pages_checked=2,
            violations=["PDF text extraction contains too many CID placeholders: 50 markers, ratio 0.100."],
        ),
    )

    cited_keys = [f"acad{index}" for index in range(1, 16)]
    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_formal_chinese_review_tex(cited_keys),
        bibtex_content="@article{manual1, title={Manual}, author={Author, A}, year={2024}}\n",
        filename_stem="review",
        tool_call_id="tc-review",
    )

    assert not (outputs_dir / "review.pdf").exists()
    assert result.update["artifacts"] == [
        "/mnt/user-data/outputs/pdf_text_audit.json",
        "/mnt/user-data/outputs/citation_audit.json",
        "/mnt/user-data/outputs/review.tex",
        "/mnt/user-data/outputs/references.bib",
    ]
    assert "PDF text audit blocked PDF generation" in result.update["messages"][0].content
    pdf_text_audit = json.loads((outputs_dir / "pdf_text_audit.json").read_text(encoding="utf-8"))
    assert pdf_text_audit["status"] == "fail"
    assert pdf_text_audit["cid_marker_count"] == 50


def test_manuscript_export_blocks_thin_math_modeling_manuscript(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)
    monkeypatch.setattr(
        manuscript_export_tool_module,
        "compile_latex_to_pdf",
        lambda _tex_path, _output_dir=None: (_ for _ in ()).throw(AssertionError("should not compile")),
    )

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_short_math_modeling_tex(),
        bibtex_content=_many_bib(total=5, recent=1),
        filename_stem="modeling_paper",
        tool_call_id="tc-quality",
    )

    assert not (outputs_dir / "modeling_paper.pdf").exists()
    assert result.update["artifacts"] == [
        "/mnt/user-data/outputs/manuscript_quality_audit.json",
        "/mnt/user-data/outputs/citation_audit.json",
        "/mnt/user-data/outputs/modeling_paper.tex",
        "/mnt/user-data/outputs/references.bib",
    ]
    assert "manuscript quality audit blocked PDF generation" in result.update["messages"][0].content
    quality = json.loads((outputs_dir / "manuscript_quality_audit.json").read_text(encoding="utf-8"))
    assert quality["status"] == "fail"
    assert quality["profile"] == "math_modeling_competition"
    assert "word_count below threshold: " in " ".join(quality["violations"])
    assert "figure_count below threshold: 1/5." in quality["violations"]
    assert "reference_count below threshold: 5/15." in quality["violations"]


def test_manuscript_export_allows_quality_checked_math_modeling_manuscript(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)
    _stub_latex_compile(monkeypatch)
    monkeypatch.setattr(manuscript_quality_module, "_pdf_page_count", lambda _path: 10)

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_long_math_modeling_tex(),
        bibtex_content=_many_bib(total=15, recent=8),
        filename_stem="modeling_paper",
        quality_profile="math_modeling_competition",
        tool_call_id="tc-quality-pass",
    )

    assert result.update["artifacts"] == [
        "/mnt/user-data/outputs/modeling_paper.pdf",
        "/mnt/user-data/outputs/modeling_paper.tex",
        "/mnt/user-data/outputs/references.bib",
        "/mnt/user-data/outputs/citation_audit.json",
        "/mnt/user-data/outputs/manuscript_quality_audit.json",
    ]
    quality = json.loads((outputs_dir / "manuscript_quality_audit.json").read_text(encoding="utf-8"))
    assert quality["status"] == "pass"
    assert quality["metrics"]["word_count"] >= 4500
    assert quality["metrics"]["page_count"] == 10
    assert quality["metrics"]["figure_count"] == 5
    assert quality["metrics"]["table_count"] == 5
    assert quality["metrics"]["reference_count"] == 15
    assert quality["metrics"]["recent_reference_count"] == 8


def test_manuscript_export_allows_nocite_all_when_explicit(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)
    _stub_latex_compile(monkeypatch)

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_tex(r"\nocite{*}"),
        bibtex_content=_bib(),
        filename_stem="paper",
        allow_nocite_all=True,
        tool_call_id="tc-1",
    )

    assert result.update["artifacts"][0] == "/mnt/user-data/outputs/paper.pdf"
    assert json.loads((outputs_dir / "citation_audit.json").read_text(encoding="utf-8"))["status"] == "pass"


def test_manuscript_export_reports_compile_failure_and_preserves_inputs(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)
    stale_pdf = outputs_dir / "paper.pdf"
    stale_pdf.write_bytes(b"stale")

    def prepare(tex_path: Path) -> Path:
        return tex_path

    def fail_compile(_tex_path: Path, _output_dir: Path | None = None) -> Path:
        raise RuntimeError("tectonic failed")

    monkeypatch.setattr(manuscript_export_tool_module, "prepare_latex_preview", prepare)
    monkeypatch.setattr(manuscript_export_tool_module, "compile_latex_to_pdf", fail_compile)

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_tex(),
        bibtex_content=_bib(),
        filename_stem="paper",
        tool_call_id="tc-1",
    )

    assert result.update["artifacts"] == [
        "/mnt/user-data/outputs/paper.tex",
        "/mnt/user-data/outputs/references.bib",
        "/mnt/user-data/outputs/citation_audit.json",
    ]
    assert not stale_pdf.exists()
    assert (outputs_dir / "paper.tex").exists()
    assert (outputs_dir / "references.bib").exists()
    assert "LaTeX compilation failed" in result.update["messages"][0].content


def test_manuscript_export_rejects_unsafe_filename_stem(tmp_path):
    outputs_dir = tmp_path / "threads" / "thread-1" / "user-data" / "outputs"
    outputs_dir.mkdir(parents=True)

    result = manuscript_export_tool_module.manuscript_export_tool.func(
        runtime=_make_runtime(str(outputs_dir)),
        tex_content=_tex(),
        bibtex_content=_bib(),
        filename_stem="../outside",
        tool_call_id="tc-1",
    )

    assert "filename_stem must be a filename stem" in result.update["messages"][0].content
    assert not any(outputs_dir.iterdir())
