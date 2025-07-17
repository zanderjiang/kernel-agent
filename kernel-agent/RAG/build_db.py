"""Build RAG Database"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(os.path.dirname(__file__))
from chroma_db import rag_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cuda-db-builder")

class CUDADocsBuilder:    
    def __init__(self):
        self.rag_server = rag_server
        
    def load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                raise ValueError("JSON must contain either an object or array of objects")
            
            logger.info(f"Loaded {len(data)} sections from {file_path}")
            return data
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    def process_cuda_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        documents = []
        
        for section in sections:
            try:
                section_id = section.get("section_id", "")
                title = section.get("title", "")
                text = section.get("text", "")                
                if not section_id or not text:
                    logger.warning(f"Skipping section with missing ID or text: {section}")
                    continue
                doc_id = section_id
                full_text = f"{title}\n\n{text}" if title else text
                
                metadata = {
                    "section_id": section_id,
                    "title": title,
                    "section_number": section.get("section_number", ""),
                    "text_length": len(text),
                    "has_title": bool(title),
                    "source": "cuda_documentation"
                }
                
                for key, value in section.items():
                    if key not in ["section_id", "title", "text", "section_number"]:
                        metadata[f"extra_{key}"] = value
                
                document = {
                    "id": doc_id,
                    "text": full_text,
                    "metadata": metadata
                }
                
                documents.append(document)
                
            except Exception as e:
                logger.error(f"Error processing section {section.get('section_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Processed {len(documents)} documents from {len(sections)} sections")
        return documents
    
    async def build_database(self, json_files: List[str], batch_size: int = 50) -> bool:
        all_documents = []
        
        for json_file in json_files:
            logger.info(f"Processing file: {json_file}")
            sections = self.load_json_data(json_file)
            
            if not sections:
                logger.warning(f"No data loaded from {json_file}")
                continue
            
            documents = self.process_cuda_sections(sections)
            all_documents.extend(documents)
        
        if not all_documents:
            logger.error("No documents to add to database")
            return False
        
        logger.info(f"Total documents to add: {len(all_documents)}")
        
        success_count = 0
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(all_documents) + batch_size - 1) // batch_size
            
            logger.info(f"Adding batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            try:
                success = await self.rag_server.add_documents(batch)
                if success:
                    success_count += len(batch)
                    logger.info(f"Successfully added batch {batch_num}")
                else:
                    logger.error(f"Failed to add batch {batch_num}")
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error adding batch {batch_num}: {e}")
        
        logger.info(f"added {success_count}/{len(all_documents)} documents")
        return success_count > 0
    
    async def verify_database(self) -> None:
        stats = await self.rag_server.get_collection_stats()
        logger.info("Database Stats:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        test_query = "GPU parallel computing"
        logger.info(f"Testing with query: '{test_query}'")
        results = await self.rag_server.semantic_search(test_query, top_k=3)
        
        if results:
            logger.info(f"found {len(results)} results")
            for i, result in enumerate(results[:2], 1):  # Show first 2 results
                logger.info(f"  Result {i}: {result['metadata'].get('title', 'No title')} "
                          f"(scre: {result['similarity_score']:.3f})")
        else:
            logger.warning("no results found")

async def main():    
    json_files = ["cuda_docs_parsed.json"]    
    logger.info(f"Processing files: {json_files}")
    
    missing_files = [f for f in json_files if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return
    
    builder = CUDADocsBuilder()
    
    try:
        success = await builder.build_database(json_files, batch_size=50)
        if success:
            logger.info("Database build success")
            await builder.verify_database()
        else:
            logger.error("Database build failed")
            return
            
    except Exception as e:
        logger.error(f"error during build: {e}")
        return
    
if __name__ == "__main__":
    asyncio.run(main())