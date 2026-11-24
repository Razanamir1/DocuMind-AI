"""Deterministic first-run workspace seeder creating real PyMuPDF PDFs and ChromaDB vectors."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.entities import (
    ConversationORM,
    DocumentORM,
    DocumentPageORM,
    MessageORM,
    utcnow,
)
from app.providers.llm import LLMProvider
from app.rag.chunker import chunk_pages
from app.rag.pdf_processor import PageText, create_sample_pdf_bytes
from app.rag.vector_store import ChromaVectorStore


def seed_initial_workspace(
    db: Session,
    *,
    provider: LLMProvider,
    vector_store: ChromaVectorStore,
    upload_dir: Path,
    defer_embeddings: bool = False,
) -> None:
    """Seeds 2 real PyMuPDF documents and 1 grounded starter conversation on an empty database."""
    if db.query(DocumentORM).count() > 0 or db.query(ConversationORM).count() > 0:
        return

    upload_dir.mkdir(parents=True, exist_ok=True)
    now = utcnow()
    two_hours_ago = now - timedelta(hours=2)
    one_hour_ago = now - timedelta(hours=1, minutes=5)
    thirty_mins_ago = now - timedelta(minutes=30)

    sample_docs = [
        {
            "id": "doc-meridian-annual-2026",
            "filename": "meridian_cloud_q3_financial_report.pdf",
            "title": "Meridian Cloud Systems — Q3 2026 Financial & Operating Report",
            "created_at": two_hours_ago,
            "pages": [
                {
                    "heading": "Executive Summary & Revenue Highlights",
                    "lines": [
                        "Meridian Cloud Systems delivered $148.4 million in total revenue during Q3 2026, representing 34.2% year-over-year growth compared to $110.6 million in Q3 2025.",
                        "Annual Recurring Revenue (ARR) reached $612.0 million as of September 30, 2026, driven by enterprise expansion in North America and EMEA.",
                        "Key Financial Metrics:",
                        "Gross margin expanded by 240 basis points to 79.4% due to custom inference cluster optimization and lower cold-storage transit costs.",
                        "Net dollar retention (NDR) stood at 121.8% across enterprise accounts generating more than $100,000 in ARR, with 418 total enterprise customers.",
                        "Operating cash flow was $31.6 million (21.3% margin), while free cash flow reached $26.9 million for the quarter.",
                    ],
                },
                {
                    "heading": "AI Infrastructure & R&D Investment",
                    "lines": [
                        "Research and development investment totaled $38.2 million in Q3 2026, or 25.7% of total revenue, focused on distributed vector indexing and multi-region failover.",
                        "Latency & Reliability Benchmarks:",
                        "Median vector retrieval latency across 50 million indexed enterprise passages improved from 42 milliseconds in Q2 to 18 milliseconds in Q3 following HNSW quantization upgrades.",
                        "Platform availability averaged 99.985% across all four sovereign cloud regions (US-East, US-West, EU-Central Frankfurt, and APAC-Tokyo).",
                        "Security & Compliance Milestones:",
                        "Meridian completed SOC 2 Type II re-certification with zero exceptions and achieved ISO/IEC 27001:2022 accreditation in August 2026.",
                    ],
                },
                {
                    "heading": "Q4 2026 Guidance & Risk Outlook",
                    "lines": [
                        "For Q4 2026, management projects total revenue between $158.0 million and $162.0 million, representing approximately 32% year-over-year growth at the midpoint.",
                        "Full-year FY 2026 revenue guidance is raised to $576.0 million - $580.0 million, up $14 million from prior outlook.",
                        "Capital Allocation & Headcount:",
                        "Ending cash, cash equivalents, and marketable securities totaled $412.5 million with zero long-term debt.",
                        "Engineering and AI research headcount grew to 640 full-time employees, with planned hiring of 45 additional staff in Q4 focused on regulated financial services deployments.",
                    ],
                },
            ],
        },
        {
            "id": "doc-security-policy-2026",
            "filename": "enterprise_ai_governance_policy.pdf",
            "title": "Enterprise AI Governance & Data Handling Policy v3.2",
            "created_at": one_hour_ago,
            "pages": [
                {
                    "heading": "1. Scope, Data Classification & Residency",
                    "lines": [
                        "This policy governs all production Retrieval-Augmented Generation (RAG) workflows, vector indices, and Large Language Model integrations operated by internal engineering teams.",
                        "Data Classification Tiers:",
                        "Tier 1 (Public): Marketing literature, public product documentation, and open-source specifications. Approved for standard indexing.",
                        "Tier 2 (Internal Confidential): Financial forecasts, architecture specs, and HR policies. Requires encrypted storage at rest (AES-256) and server-side API key isolation.",
                        "Tier 3 (Restricted PII / Regulated): Customer health records, payment card data, and government identifiers. Strictly prohibited from unredacted vector indexing.",
                        "All document ingestion pipelines must enforce a strict 25 MB maximum file size per PDF and validate binary PDF headers prior to text extraction.",
                    ],
                },
                {
                    "heading": "2. Citation Verification & Hallucination Controls",
                    "lines": [
                        "Every document-grounded AI response delivered to internal or external stakeholders must include structured, verifiable source citations listing the exact filename and page number.",
                        "Mandatory Grounding Thresholds:",
                        "Retrieval pipelines must apply a minimum semantic relevance threshold of 0.34 before passing passages to a generative model.",
                        "When retrieved passages fall below the confidence threshold, the assistant must explicitly decline to answer rather than relying on parametric model memory.",
                        "Prompt Injection Defense:",
                        "All extracted PDF passages must be encapsulated as untrusted data blocks separate from system instructions. Models must never execute embedded directives found inside uploaded files.",
                        "Document Deletion SLA: Deleting a source document must purge both its stored binary file and all derived vector embeddings from active retrieval within 5 seconds.",
                    ],
                },
            ],
        },
    ]

    for spec in sample_docs:
        pdf_bytes = create_sample_pdf_bytes(spec["title"], spec["pages"])
        sha256 = hashlib.sha256(pdf_bytes).hexdigest()
        stored_filename = f"{spec['id']}.pdf"
        (upload_dir / stored_filename).write_bytes(pdf_bytes)

        page_texts = [
            PageText(
                page_number=idx + 1,
                text=f"{p['heading']}\n\n" + "\n".join(p["lines"]),
            )
            for idx, p in enumerate(spec["pages"])
        ]
        chunks = chunk_pages(page_texts, chunk_size=850, chunk_overlap=160)

        embedding_status = "pending"
        if not defer_embeddings and provider.is_configured():
            try:
                vectors = provider.embed_texts([c.content for c in chunks])
                vector_store.upsert_document_chunks(
                    document_id=spec["id"],
                    filename=spec["filename"],
                    chunks=chunks,
                    embeddings=vectors,
                )
                embedding_status = "vector"
            except Exception:
                embedding_status = "pending"

        doc = DocumentORM(
            id=spec["id"],
            filename=spec["filename"],
            stored_filename=stored_filename,
            sha256=sha256,
            file_size_bytes=len(pdf_bytes),
            page_count=len(page_texts),
            chunk_count=len(chunks),
            status="ready",
            embedding_status=embedding_status,
            error_message=None,
            created_at=spec["created_at"],
            updated_at=spec["created_at"],
        )
        db.add(doc)
        for pt in page_texts:
            db.add(
                DocumentPageORM(
                    document_id=spec["id"],
                    page_number=pt.page_number,
                    content=pt.text,
                )
            )

    convo_id = "conv-seed-q3-briefing"
    sample_citations = [
        {
            "id": "doc-meridian-annual-2026-chunk-0",
            "documentId": "doc-meridian-annual-2026",
            "filename": "meridian_cloud_q3_financial_report.pdf",
            "pageNumber": 1,
            "excerpt": (
                "Meridian Cloud Systems delivered $148.4 million in total revenue during Q3 2026, "
                "representing 34.2% year-over-year growth compared to $110.6 million in Q3 2025. "
                "Annual Recurring Revenue (ARR) reached $612.0 million…"
            ),
            "score": 0.91,
            "chunkIndex": 0,
        },
        {
            "id": "doc-meridian-annual-2026-chunk-1",
            "documentId": "doc-meridian-annual-2026",
            "filename": "meridian_cloud_q3_financial_report.pdf",
            "pageNumber": 2,
            "excerpt": (
                "Median vector retrieval latency across 50 million indexed enterprise passages "
                "improved from 42 milliseconds in Q2 to 18 milliseconds in Q3 following HNSW "
                "quantization upgrades. Platform availability averaged 99.985%…"
            ),
            "score": 0.84,
            "chunkIndex": 1,
        },
    ]

    db.add(
        ConversationORM(
            id=convo_id,
            title="Q3 2026 Revenue & Latency Benchmarks",
            mode="documents",
            created_at=thirty_mins_ago,
            updated_at=thirty_mins_ago,
        )
    )
    db.add(
        MessageORM(
            id="msg-seed-1",
            conversation_id=convo_id,
            role="user",
            content="Summarize Meridian Cloud Q3 2026 revenue growth and how vector retrieval latency changed.",
            mode="documents",
            citations_json="[]",
            insufficient_evidence=False,
            created_at=thirty_mins_ago,
        )
    )
    db.add(
        MessageORM(
            id="msg-seed-2",
            conversation_id=convo_id,
            role="assistant",
            content="\n".join(
                [
                    "Based on the uploaded Q3 2026 financial report, Meridian Cloud Systems recorded strong top-line growth alongside infrastructure efficiency gains:",
                    "",
                    "### Financial Performance",
                    "- **Total Q3 2026 Revenue:** **$148.4 million** (+34.2% YoY vs. $110.6 million in Q3 2025) [meridian_cloud_q3_financial_report.pdf, Page 1]",
                    "- **Annual Recurring Revenue (ARR):** **$612.0 million** as of September 30, 2026, supported by 121.8% Net Dollar Retention across 418 enterprise accounts [meridian_cloud_q3_financial_report.pdf, Page 1]",
                    "- **Gross Margin & Cash Flow:** Gross margin expanded 240 bps to **79.4%**, with **$26.9 million** in free cash flow [meridian_cloud_q3_financial_report.pdf, Page 1]",
                    "",
                    "### Vector Retrieval Latency",
                    "- Median vector retrieval latency across 50 million indexed passages dropped from **42 ms in Q2 to 18 ms in Q3** after deploying HNSW quantization upgrades [meridian_cloud_q3_financial_report.pdf, Page 2].",
                    "- Platform availability averaged **99.985%** across all four sovereign cloud regions [meridian_cloud_q3_financial_report.pdf, Page 2].",
                ]
            ),
            mode="documents",
            citations_json=json.dumps(sample_citations),
            insufficient_evidence=False,
            created_at=thirty_mins_ago + timedelta(seconds=4),
        )
    )
    db.commit()
