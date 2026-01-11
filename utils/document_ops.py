from __future__ import annotations

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from pathlib import Path
from typing import Iterable, List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import CustomException
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """Load docs using appropriate loader based on extension."""
    docs: List[Document] = []
    try:
        for p in paths:
            path = project_root / p  # Resolve relative to project root
            if not path.exists():
                log.error(f"File does not exist: {path}")
                continue
            ext = path.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(path))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(path))
            elif ext == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
            else:
                log.warning("Unsupported extension skipped", path=str(path))
                continue
            docs.extend(loader.load())
        log.info("Documents loaded", count=len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise CustomException("Error loading documents", e) from e


def load_web(urls: Iterable[str]) -> List[Document]:
    """Load web documents using WebBaseLoader."""
    docs: List[Document] = []
    try:
        for url in urls:
            if not url.startswith("http"):
                log.warning("Invalid URL skipped", url=url)
                continue
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        log.info("Web documents loaded", count=len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading web documents", error=str(e))
        raise CustomException("Error loading web documents", e) from e


if __name__ == "__main__":
    # Example usage for testing
    test_paths = [
        Path("data/sample_docs.txt")
    ]

    test_urls = [
        "https://langchain-ai.github.io/langgraph/tutorials/introduction/"
    ]

    try:
        documents = load_documents(test_paths)
        for doc in documents:
            print(f"Loaded document with {len(doc.page_content)} characters.")
    except CustomException as e:
        print("Error loading documents:", e)

    try:
        web_documents = load_web(test_urls)
        for doc in web_documents:
            print(f"Loaded web document with {len(doc.page_content)} characters.")
    except CustomException as e:
        print("Error loading web documents:", e)