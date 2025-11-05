"""
Knowledge Base Manager - Infrastructure layer.
Manages KB building, chunking, and vector store operations.
"""

import json
import hashlib
import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core import settings
from app.core.models import KnowledgeBaseSnapshot
from app.infrastructure.services.document_fetcher import DocumentFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgenticChunker:
    """Uses an LLM to intelligently chunk documents based on semantic boundaries."""

    def __init__(self, chunk_size: int = 2000):
        self.chunk_size = chunk_size
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
        )

    def chunk_document(self, text: str) -> List[str]:
        """
        Use LLM to intelligently chunk document based on semantic boundaries.
        The LLM identifies meaningful sections and creates semantically coherent chunks.
        """
        logger.info(f"Starting agentic chunking for {len(text)} characters")
        # For very large texts, split into manageable chunks first
        # to avoid token limits
        if len(text) > 50000:
            logger.info(f"Text is large ({len(text)} chars), splitting into parts for processing")
            # Process in parts
            parts = []
            current_pos = 0
            chunk_target = 50000

            while current_pos < len(text):
                end_pos = min(current_pos + chunk_target, len(text))
                # Try to split at a paragraph boundary
                if end_pos < len(text):
                    last_newline = text.rfind("\n\n", current_pos, end_pos)
                    if last_newline > current_pos:
                        end_pos = last_newline

                parts.append(text[current_pos:end_pos])
                current_pos = end_pos

            logger.info(f"Split into {len(parts)} parts for processing")
            all_chunks = []
            for i, part in enumerate(parts):
                logger.info(f"Processing part {i+1}/{len(parts)} ({len(part)} chars)")
                chunks = self._chunk_with_llm(part)
                all_chunks.extend(chunks)
                logger.info(f"Part {i+1} produced {len(chunks)} chunks")
            return all_chunks
        else:
            logger.info("Text is within single-part processing limit")
            return self._chunk_with_llm(text)

    def _chunk_with_llm(self, text: str) -> List[str]:
        """
        Call LLM to chunk a section of text based on semantic boundaries.
        """
        logger.info(f"Calling LLM to chunk {len(text)} characters")
        system_message = SystemMessage(
            content=(
                "You are an AI assistant helping to split text into meaningful chunks based on topics and concepts. "
                "Identify natural boundaries in the text (such as section breaks, topic changes, or logical groupings) "
                "and split the text accordingly. Each chunk should be self-contained and semantically coherent. "
                "Return chunks separated by '\\n---CHUNK_BOUNDARY---\\n' markers."
            )
        )

        human_message = HumanMessage(
            content=(
                f"Please divide the following text into semantically different, separate and meaningful chunks. "
                f"Maintain the original text content without modification. "
                f"Use '\\n---CHUNK_BOUNDARY---\\n' to separate chunks:\n\n{text}"
            )
        )

        try:
            logger.info("Invoking LLM for chunking...")
            response = self.llm.invoke([system_message, human_message])
            logger.info("LLM response received")
            response_text = response.content if hasattr(response, "content") else str(response)

            # Split by the boundary marker
            chunks = response_text.split("\n---CHUNK_BOUNDARY---\n")
            logger.info(f"Response split into {len(chunks)} chunks")

            # Clean up chunks
            cleaned_chunks = [
                chunk.strip() for chunk in chunks if chunk.strip()
            ]

            logger.info(f"After cleaning: {len(cleaned_chunks)} valid chunks")
            return cleaned_chunks if cleaned_chunks else [text]

        except Exception as e:
            logger.error(f"Error during agentic chunking: {e}. Falling back to simple chunking.")
            return self._fallback_chunking(text)

    def _fallback_chunking(self, text: str) -> List[str]:
        """
        Fallback simple chunking strategy: split by sections, paragraphs, and size limits.
        Used if LLM chunking fails.
        """
        chunks = []
        current_chunk = ""

        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


class VectorStoreManager:
    """Manages FAISS vector store for embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.vector_store: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict[str, str]] = []  # Metadata for each chunk (source file, etc.)

    def create_index(self, chunks: List[str], metadata: Optional[List[Dict[str, str]]] = None) -> None:
        """
        Create FAISS index from document chunks.

        Args:
            chunks: List of text chunks to embed
            metadata: Optional list of metadata dicts for each chunk (e.g., {"source": "filename.md"})
        """
        logger.info(f"Creating index from {len(chunks)} chunks")
        # Generate embeddings
        logger.info("Generating embeddings for all chunks...")
        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
        logger.info(f"Embeddings generated: shape {embeddings.shape}")

        # Create FAISS index
        logger.info("Creating FAISS index...")
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatL2(dimension)
        self.vector_store.add(embeddings.astype(np.float32))
        logger.info(f"FAISS index created with dimension {dimension}")

        self.chunks = chunks
        self.chunk_metadata = metadata or [{"source": "unknown"} for _ in chunks]
        logger.info("Index creation complete")

    def search(self, query: str, k: int = 5) -> List[tuple]:
        """
        Search for top-k similar chunks.
        Returns list of tuples: (chunk_text, distance, metadata_dict)
        """
        if self.vector_store is None:
            return []

        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.vector_store.search(
            query_embedding.astype(np.float32), k
        )

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                metadata = self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {"source": "unknown"}
                results.append((self.chunks[idx], float(distance), metadata))

        return results

    def save(self, path: str) -> None:
        """Save vector store to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.vector_store, os.path.join(path, "index.faiss"))

        # Save chunks and metadata separately
        with open(os.path.join(path, "chunks.json"), "w") as f:
            json.dump(self.chunks, f)

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(self.chunk_metadata, f)

    def load(self, path: str) -> None:
        """Load vector store from disk."""
        self.vector_store = faiss.read_index(os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "chunks.json"), "r") as f:
            self.chunks = json.load(f)

        # Load metadata if it exists (backward compatible)
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.chunk_metadata = json.load(f)
        else:
            self.chunk_metadata = [{"source": "unknown"} for _ in self.chunks]


class KnowledgeBaseManager:
    """Main KB manager - orchestrates chunking, embedding, and storage."""

    def __init__(self):
        self.chunker = AgenticChunker(
            chunk_size=settings.chunk_size_for_agentic_chunking
        )
        self.vector_store = VectorStoreManager(model_name=settings.embedding_model)
        self.snapshot: Optional[KnowledgeBaseSnapshot] = None

        # Automatically load vector store if it exists
        self._load_existing_vector_store()

    def _load_existing_vector_store(self) -> None:
        """Load existing vector store from disk if it exists."""
        logger.info(f"Attempting to load vector store from {settings.vector_store_path}")
        try:
            if os.path.exists(settings.vector_store_path):
                self.vector_store.load(settings.vector_store_path)
                self.snapshot = self._load_snapshot()
                logger.info("Successfully loaded existing vector store on initialization")
            else:
                logger.info(f"Vector store path does not exist: {settings.vector_store_path}")
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {e}")

    def _download_docs(self, url: str) -> str:
        """Download documentation from URL."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text

    def _compute_hash(self, content: str) -> str:
        """Compute hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _load_local_docs(self) -> Optional[str]:
        """Load local documentation file if it exists."""
        if os.path.exists(settings.local_docs_path):
            with open(settings.local_docs_path, "r") as f:
                return f.read()
        return None

    def _save_local_docs(self, content: str) -> None:
        """Save documentation to local file."""
        os.makedirs(os.path.dirname(settings.local_docs_path), exist_ok=True)
        with open(settings.local_docs_path, "w") as f:
            f.write(content)

    def _load_snapshot(self) -> Optional[KnowledgeBaseSnapshot]:
        """Load KB snapshot metadata."""
        snapshot_path = os.path.join(settings.vector_store_path, "snapshot.json")
        if os.path.exists(snapshot_path):
            with open(snapshot_path, "r") as f:
                data = json.load(f)
                return KnowledgeBaseSnapshot(**data)
        return None

    def _save_snapshot(self, snapshot: KnowledgeBaseSnapshot) -> None:
        """Save KB snapshot metadata."""
        os.makedirs(settings.vector_store_path, exist_ok=True)
        snapshot_path = os.path.join(settings.vector_store_path, "snapshot.json")
        with open(snapshot_path, "w") as f:
            json.dump(snapshot.model_dump(mode="json"), f, indent=2, default=str)

    def build_kb(self, force_rebuild: bool = False) -> bool:
        """
        Build or rebuild the knowledge base from multiple sources (LangGraph + LangChain).
        Implements smart change detection:
        - Downloads to temp directory and compares with working directory
        - Only rebuilds embeddings for changed sources (or all if force_rebuild=True)
        - Cleans up temporary downloads after completion
        """
        try:
            logger.info(f"Starting KB build (force_rebuild={force_rebuild})")

            # Initialize document fetcher
            fetcher = DocumentFetcher(data_dir="data")

            # Check for updates (downloads to temp dir and compares)
            logger.info("Fetching and checking all document sources...")
            changed_sources, all_new_metadata = fetcher.check_for_updates()

            # Load all documents from disk
            logger.info("Loading all documents from disk...")
            all_documents = fetcher.load_all_documents()

            if not all_documents:
                logger.error("No documents found")
                print("Error: No documents found")
                return False

            logger.info(f"Loaded {len(all_documents)} documents for embedding")

            # Check if rebuild is needed
            existing_snapshot = self._load_snapshot()

            # Compute hash of all documents combined
            combined_content = "\n\n".join(all_documents.values())
            content_hash = self._compute_hash(combined_content)

            # Determine if we need to rebuild
            needs_full_rebuild = (
                force_rebuild
                or existing_snapshot is None
                or len(changed_sources) > 0
            )

            if not needs_full_rebuild:
                logger.info("KB is up to date. Loading existing vector store...")
                print("KB is up to date. Loading existing vector store...")
                self.vector_store.load(settings.vector_store_path)
                self.snapshot = existing_snapshot
                logger.info("Successfully loaded existing vector store")
                return True

            if changed_sources and not force_rebuild:
                logger.info(f"Changed sources detected: {changed_sources}. Rebuilding embeddings...")
                print(f"Changed sources detected: {changed_sources}. Rebuilding embeddings...")

            print("Building knowledge base from multiple sources (LangGraph + LangChain)...")
            logger.info(f"Starting agentic chunking for {len(all_documents)} documents...")

            # Create chunks using intelligent agentic chunking with parallelization
            chunks, chunk_metadata = self._chunk_documents(all_documents)

            logger.info(f"Created {len(chunks)} embedding chunks from {len(all_documents)} documents")
            print(f"Created {len(chunks)} embedding chunks from {len(all_documents)} documents")

            # Create vector index with metadata
            logger.info("Creating vector index from chunks with metadata...")
            self.vector_store.create_index(chunks, metadata=chunk_metadata)
            logger.info("Vector index created successfully")

            logger.info(f"Saving vector store to {settings.vector_store_path}")
            self.vector_store.save(settings.vector_store_path)
            logger.info("Vector store saved")

            # Save snapshot
            logger.info("Creating and saving snapshot...")
            snapshot = KnowledgeBaseSnapshot(
                source_url="multi-source (LangGraph + LangChain)",
                local_file_path="data/documents/",
                content_hash=content_hash,
                last_updated_at=datetime.now(),
                embedding_index_version="3.0",
                is_fresh=True,
            )
            self._save_snapshot(snapshot)
            self.snapshot = snapshot
            logger.info("Snapshot saved")

            print("Knowledge base built successfully from multiple sources!")
            logger.info("KB build completed successfully")
            return True

        except Exception as e:
            logger.exception(f"Error building KB: {e}")
            print(f"Error building KB: {e}")
            return False

    def _chunk_documents(self, documents: Dict[str, str]) -> Tuple[List[str], List[Dict[str, str]]]:
        """
        Chunk documents intelligently with parallelization:
        - Files below threshold: embed as-is
        - Files above threshold: use agentic chunking in parallel

        Returns tuple of (chunks, metadata) where metadata tracks source file for each chunk.
        Threshold is configurable in .env via agentic_chunking_threshold_kb (in KB).
        """
        threshold_bytes = settings.agentic_chunking_threshold_kb * 1024
        logger.info(f"Chunking threshold: {settings.agentic_chunking_threshold_kb}KB ({threshold_bytes} bytes)")

        # Separate documents into two groups
        small_docs = []
        large_docs = []

        for file_path, content in documents.items():
            file_name = Path(file_path).name
            content_size = len(content.encode('utf-8'))

            if content_size < threshold_bytes:
                small_docs.append((file_name, content))
            else:
                large_docs.append((file_name, content))

        chunks = []
        chunk_metadata = []

        # Add small documents as-is
        logger.info(f"Embedding {len(small_docs)} small documents as-is")
        for file_name, content in small_docs:
            logger.info(f"  {file_name}: {len(content.encode('utf-8'))} bytes (embedding as-is)")
            chunks.append(content)
            chunk_metadata.append({"source": file_name})

        # Process large documents in parallel using agentic chunking
        if large_docs:
            logger.info(f"Agentic chunking {len(large_docs)} large documents in parallel...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                future_to_filename = {
                    executor.submit(self.chunker.chunk_document, content): file_name
                    for file_name, content in large_docs
                }

                # Collect results as they complete
                for future in as_completed(future_to_filename):
                    file_name = future_to_filename[future]
                    try:
                        file_chunks = future.result()
                        logger.info(f"  {file_name}: produced {len(file_chunks)} chunks")
                        chunks.extend(file_chunks)
                        # Add metadata for each chunk from this file
                        for _ in file_chunks:
                            chunk_metadata.append({"source": file_name})
                    except Exception as e:
                        logger.error(f"  {file_name}: chunking failed - {e}")

        return chunks, chunk_metadata

    def search_offline(self, query: str, top_k: Optional[int] = None) -> List[tuple]:
        """
        Search the local knowledge base.
        Returns list of tuples: (chunk_text, distance, metadata_dict)
        """
        logger.info(f"search_offline() called with query: '{query}'")
        if self.vector_store.vector_store is None:
            logger.error("Vector store is None - KB may not be loaded")
            return []

        k = top_k or settings.top_k_chunks
        logger.info(f"Searching with k={k}")
        results = self.vector_store.search(query, k=k)
        logger.info(f"Raw search returned {len(results)} results")

        if results:
            logger.info(f"Top result distance: {results[0][1]}, similarity_threshold: {settings.similarity_threshold}")
            for i, (chunk, distance, metadata) in enumerate(results[:3]):
                logger.info(f"Result {i}: distance={distance}, source={metadata.get('source', 'unknown')}, chunk_preview='{chunk[:100]}...'")

        # Filter by similarity threshold
        filtered_results = [
            (chunk, distance, metadata)
            for chunk, distance, metadata in results
            if distance <= settings.similarity_threshold
        ]

        logger.info(f"After filtering by threshold {settings.similarity_threshold}: {len(filtered_results)} results")
        return filtered_results