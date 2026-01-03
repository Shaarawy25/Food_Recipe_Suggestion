import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import uuid
from tqdm import tqdm
import numpy as np

class RecipeVectorDatabase:
    """Vector database for recipe data using Chroma and SentenceTransformers."""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "recipe_collection"):
        self.db_path = db_path
        self.collection_name = collection_name
        
        print("Loading embedding model: all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Recipe chunks with embeddings"}
        )
        
        print("Vector database initialized. Collection:", collection_name)
        print("Database path:", db_path)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks."""
        embeddings = []
        print("Generating embeddings for", len(texts), "text chunks...")
        
        for i in tqdm(range(0, len(texts), 32)):
            batch = texts[i:i + 32]
            batch_embeddings = self.embedding_model.encode(batch, convert_to_tensor=False)
            embeddings.extend(batch_embeddings.tolist())
        
        return embeddings
    
    def add_documents(self, chunks_data: List[Dict]) -> None:
        """Add documents to vector database."""
        if not chunks_data:
            print("No data to add to vector database.")
            return
        
        texts = [chunk["text"] for chunk in chunks_data]
        embeddings = self.generate_embeddings(texts)
        ids = [str(uuid.uuid4()) for _ in chunks_data]
        metadatas = []
        
        for chunk in chunks_data:
            metadata = chunk["metadata"].copy()
            metadata = {k: str(v) if v is not None else "" for k, v in metadata.items()}
            metadatas.append(metadata)
        
        print("Adding", len(chunks_data), "documents to vector database...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print("Successfully added", len(chunks_data), "documents to vector database.")
    
    def search_similar(self, query: str, n_results: int = 5) -> Dict:
        """Search for similar documents."""
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        formatted_results = {"query": query, "results": []}
        
        for i in range(len(results['ids'][0])):
            result = {
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": float(results['distances'][0][i]) if 'distances' in results else None
            }
            formatted_results["results"].append(result)
        
        return formatted_results
    
   

def create_vector_database_from_processed_data(processed_data_file: str) -> RecipeVectorDatabase:
    """Create and populate vector database from processed data."""
    print("Loading processed data from:", processed_data_file)
    with open(processed_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks_data = data.get("all_chunks", [])
    if not chunks_data:
        raise ValueError("No chunks found in processed data file.")
    
    vector_db = RecipeVectorDatabase()
    if vector_db.collection.count() > 0:
        print("Clearing existing data in collection...")
        try:
            vector_db.client.delete_collection(vector_db.collection_name)
            vector_db.collection = vector_db.client.get_or_create_collection(
                name=vector_db.collection_name,
                metadata={"description": "Recipe chunks with embeddings"}
            )
        except Exception:
            pass
    
    vector_db.add_documents(chunks_data)
    return vector_db

def test_vector_database(vector_db: RecipeVectorDatabase) -> Dict:
    """Test vector database with sample queries."""
    test_queries = [
        "chocolate cake recipe",
        "healthy chicken dinner", 
        "vegetarian pasta",
        "quick breakfast ideas",
        "dessert with berries"
    ]
    
    test_results = {
        "database_stats": vector_db.get_database_stats(),
        "query_tests": []
    }
    
    print("\nTesting vector database with sample queries...")
    
    for query in test_queries:
        print("\nTesting query:", "'" + query + "'")
        results = vector_db.search_similar(query, n_results=3)
        
        test_result = {"query": query, "num_results": len(results["results"]), "results": []}
        
        for i, result in enumerate(results["results"][:2]):
            test_result["results"].append({
                "rank": i + 1,
                "source": result["metadata"].get("source", "Unknown"),
                "recipe_title": result["metadata"].get("recipe_title", ""),
                "text_preview": result["text"][:150] + "..." if len(result["text"]) > 150 else result["text"],
                "distance": float(result["distance"]) if result["distance"] else None
            })
            
            print("  " + str(i+1) + ". Source:", result['metadata'].get('source', 'Unknown'))
            print("     Preview:", result['text'][:100] + "...")
            print("     Distance:", round(result['distance'], 3))
        
        test_results["query_tests"].append(test_result)
    
    return test_results

if __name__ == "__main__":
    processed_data_file = "processed_recipes.json"
    
    if os.path.exists(processed_data_file):
        print("Creating vector database from processed data...")
        
        try:
            vector_db = create_vector_database_from_processed_data(processed_data_file)
            test_results = test_vector_database(vector_db)
            
            with open("vector_db_test_results.json", "w", encoding="utf-8") as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            
            print("\nVector database created and tested successfully!")
            print("Test results saved to 'vector_db_test_results.json'")
            
        except Exception as e:
            print("Error creating vector database:", str(e))
    else:
        print("Processed data file not found:", processed_data_file)