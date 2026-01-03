import streamlit as st
import ollama
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
import time
from datetime import datetime

st.set_page_config(
    page_title="Recipe Assistant Chatbot",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_chatbot():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(allow_reset=True)
    )
    collection = client.get_collection(name="recipe_collection")
    return embedding_model, collection

@st.cache_data
def get_db_stats(_collection):
    return _collection.count()

def retrieve_context(query, embedding_model, collection, n_results=5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    results = collection.query(
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

def build_prompt(query, contexts):
    system_prompt = """You are a helpful recipe assistant. Use the provided recipe context to answer the user's question.

IMPORTANT RULES:
1. Only use information from the provided context
2. If the context doesn't contain the answer, say "I don't have that information in the recipe database"
3. Always cite the source cookbook when providing information
4. When providing a recipe, give FULL details with ingredients list and complete step-by-step instructions from the context
5. Do not just list recipe names - provide the actual recipe content"""
    
    context_text = "\n\n---\n\n".join([
        "Source: {source}\n{text}".format(
            source=ctx['source'],
            text=ctx['text']
        ) for ctx in contexts[:3]
    ])
    
    user_prompt = """Context from recipe books:
{context}

User Question: {query}

Answer:""".format(context=context_text, query=query)
    
    return system_prompt, user_prompt

def generate_response(query, embedding_model, collection, model_name="llama3.2:3b", n_results=5):
    print("\n[{}] Query: {}".format(datetime.now().strftime("%H:%M:%S"), query))
    
    contexts = retrieve_context(query, embedding_model, collection, n_results)
    print("[{}] Retrieved {} contexts from: {}".format(
        datetime.now().strftime("%H:%M:%S"),
        len(contexts),
        ", ".join(set([ctx['source'] for ctx in contexts[:3]]))
    ))
    
    system_prompt, user_prompt = build_prompt(query, contexts)
    
    print("[{}] Generating response with {}...".format(
        datetime.now().strftime("%H:%M:%S"),
        model_name
    ))
    
    response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )
    print("[{}] Response generated successfully".format(datetime.now().strftime("%H:%M:%S")))
    return response['message']['content'], contexts
st.title("Recipe Assistant Chatbot")
st.markdown("Ask me anything about recipes")

with st.sidebar:
    
    st.header("Statistics")
    embedding_model, collection = load_chatbot()
    total_chunks = get_db_stats(collection)
    st.metric("Total Recipe Chunks", total_chunks)
    st.metric("Embedding Model", "all-MiniLM-L6-v2")
    st.metric("LLM Model", "Llama 3.2 3B")
    
    st.header("Settings")
    n_results = st.slider("Number of context chunks", 1, 10, 5)
    model_name = "llama3.2:3b"
    
    st.header("Sample Questions")
    sample_questions = [
        "How do I make chocolate cake?",
        "What are some healthy chicken recipes?",
        "Give me a quick breakfast idea",
        "What desserts can I make with berries?",
        "How to make pasta from scratch?"
    ]
    for q in sample_questions:
        if st.button(q, key=q):
            st.session_state.current_question = q

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_question" in st.session_state:
    user_input = st.session_state.current_question
    del st.session_state.current_question
else:
    user_input = st.chat_input("Ask me a recipe question...")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Source Chunks"):
                for i, ctx in enumerate(message["sources"], 1):
                    st.markdown("**Source {}: {}**".format(i, ctx['source']))
                    st.text(ctx['text'][:300] + "..." if len(ctx['text']) > 300 else ctx['text'])
                    st.divider()

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching recipes..."):
            response, contexts = generate_response(
                user_input, 
                embedding_model, 
                collection, 
                model_name, 
                n_results
            )
        
        st.markdown(response)
        
        if contexts:
            with st.expander("View Source Chunks"):
                for i, ctx in enumerate(contexts[:3], 1):
                    st.markdown("**Source {}: {}**".format(i, ctx['source']))
                    st.text(ctx['text'][:300] + "..." if len(ctx['text']) > 300 else ctx['text'])
                    st.divider()
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "sources": contexts
    })

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()


