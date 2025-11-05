"""
Document Fetcher - Infrastructure layer.
Handles downloading and versioning of multiple documentation sources.
Implements change detection with temporary directories and partial rebuilds.
"""

import hashlib
import json
import logging
import shutil
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import re

import requests

from app.core import settings
from app.core.models import DocumentMetadata

# Configure logging
logger = logging.getLogger(__name__)


class DocumentFetcher:
    """Fetches, manages, and versions documentation from multiple sources."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.docs_dir = self.data_dir / "documents"
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.data_dir / "documents_metadata.json"
        logger.info(f"DocumentFetcher initialized with data_dir: {self.data_dir}")

    @staticmethod
    def _calculate_hash(content: str) -> str:
        """Calculate SHA256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _download_file(self, url: str, timeout: int = 30) -> Optional[str]:
        """Download content from URL. Returns content or None if failed."""
        try:
            logger.info(f"Downloading from {url}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            logger.info(f"Successfully downloaded {url} ({len(response.text)} bytes)")
            return response.text
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None

    def _parse_contents_file(self, content: str) -> List[Tuple[str, str]]:
        """
        Parse a contents file with format:
        - [Title](url)
        Returns list of (title, url) tuples.
        """
        pattern = r'-\s*\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(pattern, content)
        logger.info(f"Found {len(matches)} links in contents file")
        return matches

    def _resolve_url(self, base_url: str, relative_url: str) -> str:
        """Resolve relative URL to absolute URL."""
        if relative_url.startswith('http'):
            return relative_url

        # Extract base domain from base_url
        base_parts = base_url.split('/')
        base_domain = '/'.join(base_parts[:3])
        base_dir = '/'.join(base_parts[:-1])

        if relative_url.startswith('/'):
            return base_domain + relative_url
        else:
            return base_dir + '/' + relative_url

    def fetch_langchain_docs(self) -> List[DocumentMetadata]:
        """
        Fetch LangChain documentation from contents file.
        Returns list of DocumentMetadata for downloaded documents.
        """
        contents_url = settings.langchain_docs_url
        logger.info(f"Fetching LangChain contents from {contents_url}")

        contents = self._download_file(contents_url)
        if not contents:
            logger.error("Failed to download LangChain contents file")
            return []

        # Parse contents to get links
        links = self._parse_contents_file(contents)
        logger.info(f"Parsed {len(links)} links from LangChain contents")

        downloaded_docs = []

        for title, url in links:
            # Resolve relative URLs
            absolute_url = self._resolve_url(contents_url, url)

            # Download the document
            doc_content = self._download_file(absolute_url)
            if not doc_content:
                logger.warning(f"Skipped {title} - download failed")
                continue

            # Save locally
            # Create filename from URL (remove special chars)
            filename = re.sub(r'[^a-zA-Z0-9._-]', '_', title.lower()) + '.md'
            local_path = self.docs_dir / f"langchain_{filename}"

            try:
                local_path.write_text(doc_content)
                file_hash = self._calculate_hash(doc_content)

                metadata = DocumentMetadata(
                    source="langchain",
                    url=absolute_url,
                    local_path=str(local_path),
                    file_hash=file_hash,
                )
                downloaded_docs.append(metadata)
                logger.info(f"Saved LangChain doc: {filename}")

            except Exception as e:
                logger.error(f"Failed to save {filename}: {e}")

        logger.info(f"Successfully downloaded {len(downloaded_docs)} LangChain documents")
        return downloaded_docs

    def fetch_langgraph_docs(self) -> List[DocumentMetadata]:
        """
        Fetch LangGraph documentation (existing single file).
        First tries to use existing local copy for backward compatibility,
        then falls back to downloading from configured URL.
        Returns list of DocumentMetadata.
        """
        # Check if we already have a local LangGraph file (for backward compatibility)
        legacy_path = Path(settings.local_docs_path)
        if legacy_path.exists():
            try:
                content = legacy_path.read_text()
                file_hash = self._calculate_hash(content)

                # Copy to new location in documents directory
                local_path = self.docs_dir / "langgraph_main.txt"
                local_path.write_text(content)

                metadata = DocumentMetadata(
                    source="langgraph",
                    url=settings.langgraph_docs_url,  # Track the configured URL
                    local_path=str(local_path),
                    file_hash=file_hash,
                )
                logger.info(f"Using existing LangGraph docs from {legacy_path}")
                return [metadata]
            except Exception as e:
                logger.warning(f"Could not use legacy LangGraph file: {e}")

        # If no legacy file, try to download from configured URL
        url = settings.langgraph_docs_url
        logger.info(f"Fetching LangGraph docs from {url}")

        content = self._download_file(url)
        if not content:
            logger.error("Failed to download LangGraph docs")
            return []

        # Save locally
        local_path = self.docs_dir / "langgraph_main.txt"
        try:
            local_path.write_text(content)
            file_hash = self._calculate_hash(content)

            metadata = DocumentMetadata(
                source="langgraph",
                url=url,
                local_path=str(local_path),
                file_hash=file_hash,
            )
            logger.info(f"Saved LangGraph docs to {local_path}")
            return [metadata]

        except Exception as e:
            logger.error(f"Failed to save LangGraph docs: {e}")
            return []

    def load_all_documents(self) -> Dict[str, str]:
        """
        Load all documents from disk.
        Returns dict with {local_path: content}.
        """
        documents = {}

        for doc_file in self.docs_dir.glob("*.txt"):
            try:
                content = doc_file.read_text()
                documents[str(doc_file)] = content
                logger.info(f"Loaded {doc_file.name} ({len(content)} bytes)")
            except Exception as e:
                logger.error(f"Failed to load {doc_file}: {e}")

        for doc_file in self.docs_dir.glob("*.md"):
            try:
                content = doc_file.read_text()
                documents[str(doc_file)] = content
                logger.info(f"Loaded {doc_file.name} ({len(content)} bytes)")
            except Exception as e:
                logger.error(f"Failed to load {doc_file}: {e}")

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def check_for_updates(self) -> Tuple[Set[str], Dict[str, List[DocumentMetadata]]]:
        """
        Check if any documents have changed by downloading to temp directory first.
        Compares with working directory to detect changes.
        If changes detected, moves new files to working directory and updates metadata.

        Returns:
            Tuple of (changed_sources, all_new_metadata):
            - changed_sources: Set of source names that changed ("langgraph", "langchain", etc)
            - all_new_metadata: Dict mapping source to list of DocumentMetadata with updated paths
        """
        logger.info("Checking for document updates...")

        # Create temp directory for downloads
        temp_dir = self.data_dir / "temp_downloads"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Temporarily use temp directory for downloads
            original_docs_dir = self.docs_dir
            self.docs_dir = temp_dir

            # Fetch new documents to temp directory
            new_langgraph = self.fetch_langgraph_docs()
            new_langchain = self.fetch_langchain_docs()

            # Restore original docs directory
            self.docs_dir = original_docs_dir

            # Now compare temp downloads with working directory
            changed_sources: Set[str] = set()

            # Load previous metadata
            old_metadata = self._load_metadata()
            old_docs_by_source = {}
            for meta in old_metadata:
                source = meta.source
                if source not in old_docs_by_source:
                    old_docs_by_source[source] = {}
                old_docs_by_source[source][meta.local_path] = meta

            logger.info(f"Loaded {len(old_metadata)} documents from metadata file")
            for source, docs in old_docs_by_source.items():
                logger.info(f"  {source}: {len(docs)} documents")
                for path, meta in list(docs.items())[:3]:  # Show first 3
                    logger.debug(f"    - {Path(path).name}: {meta.file_hash[:8]}...")

            # Check LangGraph changes
            logger.info(f"Checking LangGraph changes: {len(new_langgraph)} new vs {len(old_docs_by_source.get('langgraph', {}))} old")
            langgraph_changed = self._check_source_changes("langgraph", new_langgraph, old_docs_by_source.get("langgraph", {}))
            if langgraph_changed:
                changed_sources.add("langgraph")
                logger.info("LangGraph documents have changed")
            else:
                logger.info("LangGraph documents have NOT changed")

            # Check LangChain changes
            logger.info(f"Checking LangChain changes: {len(new_langchain)} new vs {len(old_docs_by_source.get('langchain', {}))} old")
            langchain_changed = self._check_source_changes("langchain", new_langchain, old_docs_by_source.get("langchain", {}))
            if langchain_changed:
                changed_sources.add("langchain")
                logger.info("LangChain documents have changed")
            else:
                logger.info("LangChain documents have NOT changed")

            # Move temp files to working directory
            logger.info("Moving downloaded files from temp to working directory...")
            self._move_downloaded_files(temp_dir, original_docs_dir)

            # Update metadata paths to point to working directory
            updated_langgraph = self._update_metadata_paths(new_langgraph, temp_dir, original_docs_dir)
            updated_langchain = self._update_metadata_paths(new_langchain, temp_dir, original_docs_dir)

            all_new_metadata = {
                "langgraph": updated_langgraph,
                "langchain": updated_langchain,
            }

            # Only save metadata if there were changes OR if no metadata exists yet
            if changed_sources or not old_metadata:
                logger.info("Saving updated metadata due to changes or first run")
                all_metadata_to_save = updated_langgraph + updated_langchain
                self._save_metadata(all_metadata_to_save)
            else:
                logger.info("No changes detected, keeping existing metadata")

            logger.info(f"Changed sources: {changed_sources}")
            return changed_sources, all_new_metadata

        finally:
            # Clean up temp directory
            if temp_dir.exists():
                logger.info(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)

    def _move_downloaded_files(self, temp_dir: Path, target_dir: Path) -> None:
        """Move all files from temp directory to target directory."""
        try:
            if not temp_dir.exists():
                return

            target_dir.mkdir(parents=True, exist_ok=True)

            for file_path in temp_dir.glob("*"):
                if file_path.is_file():
                    target_file = target_dir / file_path.name
                    logger.debug(f"Moving {file_path.name} to {target_dir}")
                    shutil.move(str(file_path), str(target_file))

            logger.info(f"Successfully moved downloaded files to {target_dir}")
        except Exception as e:
            logger.error(f"Error moving downloaded files: {e}")

    def _update_metadata_paths(self, metadata_list: List[DocumentMetadata], temp_dir: Path, target_dir: Path) -> List[DocumentMetadata]:
        """Update metadata paths from temp directory to target directory."""
        updated = []
        for meta in metadata_list:
            new_meta = DocumentMetadata(
                source=meta.source,
                url=meta.url,
                local_path=str(target_dir / Path(meta.local_path).name),
                file_hash=meta.file_hash,
            )
            new_meta.last_updated = meta.last_updated
            updated.append(new_meta)
        return updated

    def _check_source_changes(self, source: str, new_metadata: List[DocumentMetadata], old_metadata_dict: Dict[str, DocumentMetadata]) -> bool:
        """
        Check if a source has any changes by comparing content hashes.
        Returns True if any document's CONTENT actually changed, False otherwise.

        Compares by filename, not by count, to avoid false positives when the
        number of documents changes but content is identical.
        """
        if not new_metadata and not old_metadata_dict:
            logger.info(f"No {source} documents in either old or new")
            return False

        # Build hash map from old metadata for faster comparison
        old_hashes_by_filename = {}
        for old_path, old_meta in old_metadata_dict.items():
            old_name = Path(old_path).name
            old_hashes_by_filename[old_name] = old_meta.file_hash
            logger.debug(f"Old {source} document: {old_name} -> hash {old_meta.file_hash[:8]}...")

        # Build hash map for new documents
        new_hashes_by_filename = {}
        for meta in new_metadata:
            new_name = Path(meta.local_path).name
            new_hashes_by_filename[new_name] = meta.file_hash
            logger.debug(f"New {source} document: {new_name} -> hash {meta.file_hash[:8]}...")

        # Strategy: Compare actual content by hash, not by count
        # Check 1: Are there any files in new that don't exist in old with the same hash?
        for new_name, new_hash in new_hashes_by_filename.items():
            if new_name not in old_hashes_by_filename:
                logger.info(f"New document found: {source}/{new_name}")
                return True

            old_hash = old_hashes_by_filename[new_name]
            if old_hash != new_hash:
                logger.info(f"Hash mismatch for {source}/{new_name}: {old_hash[:8]}... != {new_hash[:8]}...")
                return True

        # Check 2: Are there files in old that don't exist in new? (documents were removed)
        for old_name in old_hashes_by_filename.keys():
            if old_name not in new_hashes_by_filename:
                logger.info(f"Document removed: {source}/{old_name}")
                return True

        logger.info(f"No changes detected in {source} documents (content hashes match)")
        return False

    def _load_metadata(self) -> List[DocumentMetadata]:
        """Load document metadata from file."""
        if not self.metadata_file.exists():
            return []

        try:
            data = json.loads(self.metadata_file.read_text())
            return [DocumentMetadata(**m) for m in data]
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return []

    def _save_metadata(self, metadata: List[DocumentMetadata]):
        """Save document metadata to file."""
        try:
            data = [m.model_dump() for m in metadata]
            self.metadata_file.write_text(json.dumps(data, indent=2))
            logger.info(f"Saved metadata for {len(metadata)} documents")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def fetch_all_sources(self) -> List[DocumentMetadata]:
        """
        Fetch all documentation sources.
        Returns list of all DocumentMetadata.
        """
        logger.info("Starting to fetch all documentation sources...")

        langgraph_docs = self.fetch_langgraph_docs()
        langchain_docs = self.fetch_langchain_docs()

        all_docs = langgraph_docs + langchain_docs

        # Save metadata
        self._save_metadata(all_docs)

        logger.info(f"Total documents fetched: {len(all_docs)}")
        return all_docs