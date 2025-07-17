import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Dict, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp.server.fastmcp import FastMCP

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

db_client = chromadb.PersistentClient(
    path="./cuda_docs_db",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

class RAGDatabase:
    """RAG server for CUDA docs"""    
    def __init__(self):
        self.collection_name = "cuda_docs"
        self.embedding_model = "text-embedding-3-small" # TODO: try text-embedding-3-large
        self.collection = None
        self.initialize_collection()
    
    def initialize_collection(self):
        try:
            self.collection = db_client.get_collection(
                name=self.collection_name,
                embedding_function=self._get_embedding_function()
            )
        except Exception:
            self.collection = db_client.create_collection(
                name=self.collection_name,
                embedding_function=self._get_embedding_function(),
                metadata={"description": "CUDA documentation embeddings"}
            )
    
    def _get_embedding_function(self):        
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=self.embedding_model
        )
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        try:
            texts = [doc["text"] for doc in documents]
            metadatas = [doc.get("metadata", {}) for doc in documents]
            ids = [doc["id"] for doc in documents]
            
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            return True
            
        except Exception as e:
            return False
    
    async def semantic_search(self, query: str, top_k: int = 5, where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Perform search on the CUDA documentation.
        Args:
            query: Search query
            top_k: Number of top results to return
            where: Optional metadata filters using ChromaDB query operators
        Returns list of search results with text, metadata, and similarity scores
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        "text": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'][0] else {},
                        "similarity_score": 1 - results['distances'][0][i],
                        "rank": i + 1
                    }
                    formatted_results.append(result)            
            return formatted_results
            
        except Exception as e:
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model
            }
        except Exception as e:
            return {"error": str(e)}

rag_server = RAGDatabase()
mcp = FastMCP("CUDA Docs Server")

@mcp.tool()
async def semantic_search_cuda(
    query: str, 
    top_k: int = 5,
    where: Optional[Dict] = None,
    include_raw_results: bool = False
) -> str:
    """
    Semantically search CUDA documentation for relevant information.
    
    Args:
        query: Search query about CUDA concepts
        top_k: Number of top results to return (default: 5), 1 < top_k < 20
        where: Optional metadata filters using ChromaDB query operators
               Examples:
               - Simple equality: {"source": "cuda_guide"}
               - Comparison: {"page_number": {"$gt": 5}}
               - Logical AND: {"$and": [{"source": "cuda_guide"}, {"difficulty": "beginner"}]}
        include_raw_results: If True, includes raw ChromaDB results in response
    """
    if not query:
        return "Query was empty"
    
    results = await rag_server.semantic_search(query, top_k, where)
    
    if not results:
        return "No results found for your query. The collection might be empty or there was an error."
    
    formatted_output = f"## CUDA Documentation Search Results\n\n"
    #formatted_output += f"**Query:** {query}\n"
    #if where:
    #    formatted_output += f"**Filters:** {json.dumps(where, indent=2)}\n"
    formatted_output += f"**Found {len(results)} results:**\n\n"
    
    for result in results:
        formatted_output += f"### Result {result['rank']}\n"
        formatted_output += f"**Similarity Score:** {result['similarity_score']:.3f}\n"
        
        if result['metadata']:
            formatted_output += f"**Metadata:** {json.dumps(result['metadata'], indent=2)}\n"
        
        formatted_output += f"**Content:**\n{result['text']}\n\n"
        formatted_output += "---\n\n"
    
    if include_raw_results:
        formatted_output += f"## Raw ChromaDB Results\n\n"
        formatted_output += f"```json\n{json.dumps(results, indent=2, default=str)}\n```\n"
    
    return formatted_output

@mcp.tool()
async def add_cuda_documents(documents: List[Dict[str, Any]]) -> str:
    """
    Add new CUDA documentation to the vector store.
    
    Args:
        documents: List of documents to add. Each document must have 'id' and 'text' fields,
                  and optionally 'metadata' field with additional information.
    """
    if not documents:
        return "Error: No documents provided"
    
    for doc in documents:
        if not isinstance(doc, dict):
            return "Error: Each document must be a dictionary"
        if not doc.get("id") or not doc.get("text"):
            return "Error: Each document must have 'id' and 'text' fields"
    
    success = await rag_server.add_documents(documents)
    
    if success:
        return f"Successfully added {len(documents)} documents to the CUDA documentation collection."
    else:
        return "Error: Failed to add documents to the collection."

@mcp.tool()
async def get_collection_info() -> str:
    """
    Get information about the CUDA documentation collection.
    
    Returns statistics about the current state of the vector database.
    """
    stats = await rag_server.get_collection_stats()
    
    info_text = "## CUDA Documentation Collection Info\n\n"
    for key, value in stats.items():
        info_text += f"**{key.replace('_', ' ').title()}:** {value}\n"
    
    return info_text

if __name__ == "__main__":
    mcp.run(transport='stdio')