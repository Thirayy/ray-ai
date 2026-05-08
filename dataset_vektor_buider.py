#!/usr/bin/env python3
"""
Build vector dataset from PDFs only:
- full parsing per file
- word-based chunking + overlap
- grouped processing (e.g. 3 PDFs per step)
- save: embeddings.npy + metadata.jsonl
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import PyPDF2
from PyPDF2.errors import DependencyError
from sentence_transformers import SentenceTransformer


# =========================
# CONFIG
# =========================
PDF_DIR = Path("pdfs")
OUT_DIR = Path("kb_dataset")
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_WORDS = 700
CHUNK_OVERLAP = 35
MIN_CHUNK_CHARS = 50

GROUP_SIZE = 3
EMBED_BATCH_SIZE = 8
EMBED_SLEEP_SEC = 0.3


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def chunk_words(text: str, chunk_words: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    chunks = []
    for i in range(0, len(words), step):
        part = words[i:i + chunk_words]
        if not part:
            continue
        c = " ".join(part).strip()
        if len(c) >= MIN_CHUNK_CHARS:
            chunks.append(c)
        if i + chunk_words >= len(words):
            break
    return chunks


def extract_pdf_chunks(pdf_path: Path) -> List[Dict]:
    items: List[Dict] = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            # Some PDFs are encrypted with AES. PyPDF2 needs pycryptodome to decrypt AES streams.
            # If the PDF is encrypted and we can't decrypt (even with empty password), skip it
            # so the dataset build can continue.
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")  # best-effort: try empty password first
                except Exception:
                    print(f"   ! skip encrypted PDF (need password/AES support): {pdf_path.name}")
                    return []

            total_pages = len(reader.pages)
            for page_idx, page in enumerate(reader.pages, start=1):
                text = clean_text(page.extract_text() or "")
                if not text:
                    continue
                chunks = chunk_words(text, CHUNK_WORDS, CHUNK_OVERLAP)
                for ci, ch in enumerate(chunks, start=1):
                    items.append({
                        "text": ch,
                        "source": pdf_path.name,
                        "page": page_idx,
                        "total_pages": total_pages,
                        "chunk_in_page": ci,
                    })
    except DependencyError:
        # Typical case: AES-encrypted PDF but pycryptodome isn't installed.
        print(f"   ! skip AES-encrypted PDF (install pycryptodome): {pdf_path.name}")
        return []
    except Exception as e:
        print(f"   ! error parsing {pdf_path.name}: {e}")
        return []

    return items


def build_vector_dataset():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF found in:", PDF_DIR)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(EMBED_MODEL)

    all_meta: List[Dict] = []
    all_vecs: List[np.ndarray] = []

    total_groups = (len(pdf_files) + GROUP_SIZE - 1) // GROUP_SIZE
    for g, start in enumerate(range(0, len(pdf_files), GROUP_SIZE), start=1):
        group = pdf_files[start:start + GROUP_SIZE]
        print(f"\n[Group {g}/{total_groups}] processing {len(group)} PDF(s)")

        group_items: List[Dict] = []
        for p in group:
            print(" - parsing:", p.name)
            group_items.extend(extract_pdf_chunks(p))

        if not group_items:
            print("   no chunks in this group")
            continue

        texts = [x["text"] for x in group_items]
        print(f"   embedding {len(texts)} chunks...")

        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            vec = model.encode(batch, convert_to_numpy=True, show_progress_bar=False).astype("float32")
            all_vecs.append(vec)
            time.sleep(EMBED_SLEEP_SEC)

        all_meta.extend(group_items)
        print(f"   total chunks so far: {len(all_meta)}")

    if not all_meta:
        print("No chunks extracted.")
        return

    vectors = np.vstack(all_vecs).astype("float32")
    np.save(OUT_DIR / "embeddings.npy", vectors)

    with open(OUT_DIR / "metadata.jsonl", "w", encoding="utf-8") as f:
        for m in all_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "embed_model": EMBED_MODEL,
                "dim": int(vectors.shape[1]),
                "total_vectors": int(vectors.shape[0]),
                "chunk_words": CHUNK_WORDS,
                "chunk_overlap": CHUNK_OVERLAP,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nDone.")
    print("Saved:", OUT_DIR / "embeddings.npy")
    print("Saved:", OUT_DIR / "metadata.jsonl")
    print("Saved:", OUT_DIR / "manifest.json")


if __name__ == "__main__":
    build_vector_dataset()
