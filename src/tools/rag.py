"""RAG (Retrieval-Augmented Generation) System.

Provides a custom knowledge base with vector storage for information retrieval.
Uses ChromaDB for vector storage and sentence-transformers for embeddings.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from claude_agent_sdk import tool

logger = logging.getLogger(__name__)


@dataclass
class RAGDocument:
    """A document in the RAG system."""

    id: str
    content: str
    metadata: dict
    score: float = 0.0


class RAGSystem:
    """Vector-based knowledge retrieval system using ChromaDB."""

    def __init__(
        self,
        persist_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "knowledge_base",
    ):
        """Initialize RAG system.

        Args:
            persist_dir: Directory for vector store persistence
            embedding_model: Sentence transformer model name
            collection_name: Name of the ChromaDB collection
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self._client = None
        self._collection = None
        self._embedder = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of ChromaDB and embeddings."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                self._client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=Settings(anonymized_telemetry=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                logger.error("ChromaDB not installed. Run: pip install chromadb")
                raise

        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.embedding_model)
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Add a document to the knowledge base.

        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata dictionary

        Returns:
            True if successful
        """
        self._ensure_initialized()

        try:
            embedding = self._embedder.encode(content).tolist()
            self._collection.upsert(
                ids=[doc_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata or {}],
            )
            logger.info(f"Added document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False

    def add_documents_batch(
        self,
        documents: list[tuple[str, str, dict]],
    ) -> int:
        """Add multiple documents to the knowledge base.

        Args:
            documents: List of (doc_id, content, metadata) tuples

        Returns:
            Number of documents successfully added
        """
        self._ensure_initialized()

        if not documents:
            return 0

        ids = [d[0] for d in documents]
        contents = [d[1] for d in documents]
        metadatas = [d[2] for d in documents]

        try:
            embeddings = self._embedder.encode(contents).tolist()
            self._collection.upsert(
                ids=ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(f"Added {len(documents)} documents")
            return len(documents)
        except Exception as e:
            logger.error(f"Error adding documents batch: {e}")
            return 0

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[RAGDocument]:
        """Query the knowledge base for relevant documents.

        Args:
            query_text: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of RAGDocument objects sorted by relevance
        """
        self._ensure_initialized()

        try:
            query_embedding = self._embedder.encode(query_text).tolist()

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata,
            )

            documents = []
            if results and results["ids"]:
                for i, doc_id in enumerate(results["ids"][0]):
                    documents.append(
                        RAGDocument(
                            id=doc_id,
                            content=results["documents"][0][i] if results["documents"] else "",
                            metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                            score=1 - results["distances"][0][i] if results["distances"] else 0.0,
                        )
                    )

            return documents

        except Exception as e:
            logger.error(f"Query error: {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the knowledge base.

        Args:
            doc_id: Document identifier

        Returns:
            True if successful
        """
        self._ensure_initialized()

        try:
            self._collection.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False

    def clear(self) -> bool:
        """Clear all documents from the knowledge base.

        Returns:
            True if successful
        """
        self._ensure_initialized()

        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Cleared knowledge base")
            return True
        except Exception as e:
            logger.error(f"Clear error: {e}")
            return False

    def count(self) -> int:
        """Get the number of documents in the knowledge base.

        Returns:
            Document count
        """
        self._ensure_initialized()

        try:
            return self._collection.count()
        except Exception as e:
            logger.error(f"Count error: {e}")
            return 0


# Global RAG instance (lazy initialized)
_rag_system: Optional[RAGSystem] = None


def get_rag_system() -> RAGSystem:
    """Get or create the global RAG system instance."""
    global _rag_system
    if _rag_system is None:
        from ..config import get_config

        config = get_config()
        _rag_system = RAGSystem(
            persist_dir=config.rag.persist_dir,
            embedding_model=config.rag.embedding_model,
        )
    return _rag_system


# Tool function for Claude Agent SDK
@tool(
    "query_knowledge_base",
    "Search the knowledge base for relevant information. Use this to retrieve facts, documentation, or other stored knowledge before answering questions.",
    {
        "query": str,
        "top_k": int,  # Optional, number of results (default 5)
    },
)
async def query_knowledge_base(args: dict) -> dict:
    """Query the RAG knowledge base.

    Args:
        args: Dictionary with 'query' (required) and 'top_k' (optional)

    Returns:
        Dictionary with retrieved documents
    """
    query = args.get("query", "")
    top_k = args.get("top_k", 5)

    if not query.strip():
        return {"content": [{"type": "text", "text": "Error: No query provided."}]}

    try:
        rag = get_rag_system()
        results = rag.query(query, top_k=top_k)

        if not results:
            return {
                "content": [
                    {"type": "text", "text": "No relevant documents found in the knowledge base."}
                ]
            }

        response_parts = [f"Found {len(results)} relevant documents:\n"]

        for i, doc in enumerate(results, 1):
            response_parts.append(
                f"**Document {i}** (relevance: {doc.score:.2f}):\n{doc.content}\n"
            )

        return {"content": [{"type": "text", "text": "\n".join(response_parts)}]}

    except Exception as e:
        logger.error(f"RAG query error: {e}")
        return {"content": [{"type": "text", "text": f"Knowledge base query error: {str(e)}"}]}
