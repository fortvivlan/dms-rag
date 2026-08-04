"""Build a FAISS vector store directly from codex_parts.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_MODEL = "ai-forever/sbert_large_nlu_ru"


def load_documents(csv_path: Path) -> list[Document]:
    """Load non-empty text/source rows from the reparsed codex CSV."""
    documents: list[Document] = []

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != ["text", "source"]:
            raise ValueError(
                f"Unexpected CSV header {reader.fieldnames}; "
                "expected ['text', 'source']"
            )

        for row_number, row in enumerate(reader, start=2):
            text = row["text"].strip()
            source = row["source"].strip()
            if not text or not source:
                raise ValueError(f"Empty text or source in CSV row {row_number}")
            documents.append(Document(page_content=text, metadata={"source": source}))

    if not documents:
        raise ValueError(f"No documents found in {csv_path}")
    return documents


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split long rows with the same settings used by the original indexer."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def create_embeddings(
    model_name: str, device: str | None, batch_size: int
) -> HuggingFaceEmbeddings:
    """Create the sentence-transformers embedding adapter."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    model_kwargs = {"device": device} if device else {}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"batch_size": batch_size},
        show_progress=True,
    )


def build_vectorstore(
    documents: list[Document], embeddings: HuggingFaceEmbeddings
) -> FAISS:
    """Embed documents and create a cosine-similarity FAISS index."""
    texts = [document.page_content for document in documents]
    vectors = np.asarray(embeddings.embed_documents(texts), dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(documents):
        raise ValueError(
            f"Embedding model returned shape {vectors.shape} for "
            f"{len(documents)} documents"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned non-finite values")

    # Inner product over unit vectors is cosine similarity.
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_to_docstore_id = {index: str(index) for index in range(len(documents))}
    docstore = InMemoryDocstore(
        {str(index): document for index, document in enumerate(documents)}
    )
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS vector store from codex_parts.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("codex_parts.csv"),
        help="input CSV (default: codex_parts.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("faiss_codex"),
        help="output directory (default: faiss_codex)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face embedding model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        help="sentence-transformers device, for example cpu, cuda, or cuda:0",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="embedding batch size (default: 32)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="maximum chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="chunk overlap in characters (default: 200)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading documents from {args.input} ...", flush=True)
    documents = load_documents(args.input)
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
    print(
        f"Loaded {len(documents)} rows and produced {len(chunks)} chunks.",
        flush=True,
    )

    print(f"Loading embedding model {args.model} ...", flush=True)
    embeddings = create_embeddings(args.model, args.device, args.batch_size)
    print("Embedding chunks and building the FAISS index ...", flush=True)
    vectorstore = build_vectorstore(chunks, embeddings)

    args.output.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(args.output))
    print(
        f"Saved {vectorstore.index.ntotal} vectors to {args.output} "
        f"(dimension {vectorstore.index.d}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
