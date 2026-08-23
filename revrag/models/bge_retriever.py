"""
BGE dense retrieval baseline.

Uses BAAI/bge-large-en-v1.5 via sentence-transformers with a FAISS flat index.
Documents are chunked (chunk_size tokens, chunk_overlap overlap) because BGE
has a 512-token limit. This is the standard chunked dense retrieval baseline.

The chunking-induced information loss is precisely what our Mamba-2 encoder
is designed to avoid — this baseline makes that gap measurable.
"""

import logging
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    tokenizer=None,
) -> List[str]:
    """
    Split text into overlapping token-based chunks.
    Falls back to word-based splitting if no tokenizer provided.
    """
    if tokenizer is not None:
        tokens = tokenizer.tokenize(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk = tokenizer.convert_tokens_to_string(tokens[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks
    else:
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            start += chunk_size - overlap
        return chunks


class BGERetriever:
    """
    Sentence-Transformer (BGE) bi-encoder with FAISS index.

    Each document is split into chunks; retrieval returns the highest-scoring
    chunk and its parent doc_id. Chunk metadata is stored for cross-page
    analysis in ablation (b): consistency verifier bolted onto BGE.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading BGE model {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

        # Index storage
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunk_meta: List[Dict] = []  # {doc_id, chunk_idx, text}

    def index_documents(self, documents: List[Tuple[str, str]]) -> None:
        """
        Chunk documents, encode, and build FAISS index.

        Args:
            documents: [(doc_id, doc_text), ...]
        """
        logger.info(f"Chunking and indexing {len(documents)} documents...")
        all_chunks, self.chunk_meta = [], []

        for doc_id, text in tqdm(documents, desc="Chunking"):
            chunks = _chunk_text(text, self.chunk_size, self.chunk_overlap)
            for c_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                self.chunk_meta.append({
                    "doc_id": doc_id,
                    "chunk_idx": c_idx,
                    "total_chunks": len(chunks),
                    "text": chunk,
                })

        logger.info(f"Encoding {len(all_chunks)} chunks...")
        embeddings = self.model.encode(
            all_chunks,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def retrieve(
        self, query: str, top_k: int = 20
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve top-k chunks for a single query.

        Returns:
            [(doc_id, score, chunk_meta), ...] sorted descending
        """
        q_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.chunk_meta[idx]
            results.append((meta["doc_id"], float(score), meta))
        return results

    def batch_retrieve(
        self, queries: List[Tuple[str, str]], top_k: int = 20
    ) -> Dict[str, List[Tuple[str, float, Dict]]]:
        query_texts = [q[1] for q in queries]
        q_embs = self.model.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        results = {}
        scores_batch, indices_batch = self.index.search(q_embs, top_k)
        for (qid, _), scores, indices in zip(queries, scores_batch, indices_batch):
            hits = []
            for score, idx in zip(scores, indices):
                if idx == -1:
                    continue
                meta = self.chunk_meta[idx]
                hits.append((meta["doc_id"], float(score), meta))
            results[qid] = hits
        return results
