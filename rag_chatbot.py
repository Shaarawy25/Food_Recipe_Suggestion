import ollama
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json

class RecipeRAGChatbot:
    def __init__(self, db_path="./chroma_db", model_name="llama3.2:3b"):
        self.model_name = model_name
        self.db_path = db_path
        
        print("Initializing Recipe RAG Chatbot...")
        
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(allow_reset=True)
        )
        self.collection = self.client.get_collection(name="recipe_collection")
        print("Loaded existing recipe collection with", self.collection.count(), "documents")

    
    
    def retrieve_context(self, query, n_results=5):
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        contexts = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            contexts.append({
                'text': doc,
                'source': metadata.get('source', 'Unknown'),
                'chunk_id': metadata.get('chunk_id', i),
                'distance': distance
            })
        
        return contexts
    
    def build_prompt(self, query, contexts):
        system_prompt = """You are a helpful recipe assistant. Use the provided recipe context to answer the user's question.

IMPORTANT RULES:
1. Only use information from the provided context
2. If the context doesn't contain the answer, say "I don't have that information in the recipe database"
3. Always cite the source cookbook when providing information
4. Be concise and helpful
5. If asked for a recipe, provide clear steps and ingredients from the context"""
        
        context_text = "\n\n---\n\n".join([
            "Source: {source}\n{text}".format(
                source=ctx['source'],
                text=ctx['text'][:500]
            ) for ctx in contexts[:3]
        ])
        
        user_prompt = """Context from recipe books:
{context}

User Question: {query}

Answer:""".format(context=context_text, query=query)
        
        return system_prompt, user_prompt
    
    def chat(self, query, n_results=5, stream=True):
        print("Query:", query)
        
        print("\nRetrieving relevant recipes...")
        contexts = self.retrieve_context(query, n_results)
        
        print("\nTop", len(contexts), "relevant chunks:")
        for i, ctx in enumerate(contexts[:5], 1):
            print("  ", i, "-", ctx['source'])
        
        system_prompt, user_prompt = self.build_prompt(query, contexts)
        
        print("\nGenerating response...")
        
        try:
            if stream:
                response_text = ""
                for chunk in ollama.chat(
                    model=self.model_name,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    stream=True
                ):
                    content = chunk['message']['content']
                    print(content, end='', flush=True)
                    response_text += content
                print()
                return response_text
            else:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ]
                )
                response_text = response['message']['content']
                print(response_text)
                return response_text
                
        except Exception as e:
            print("\nError generating response:", e)
            print("\nMake sure:")
            print("1. Ollama is running")
            print("2. Model", self.model_name, "is downloaded")
            print("3. Run: ollama pull", self.model_name)
            return None

  

def test_rag_pipeline():
    print("Testing RAG Pipeline with Sample Queries")
    
    chatbot = RecipeRAGChatbot(model_name="llama3.2:3b")
    
    test_queries = [
        "How do I make chocolate cake?",
        "What are some healthy chicken recipes?",
        "Give me a quick breakfast idea",
        "What desserts can I make with berries?",
        "How to make pasta?"
    ]
    
    results = []
    for query in test_queries:
        response = chatbot.chat(query, n_results=5, stream=False)
        results.append({
            'query': query,
            'response': response
        })
    
    with open('rag_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Test results saved to rag_test_results.json")


if __name__ == "__main__":
    test_rag_pipeline()
    
