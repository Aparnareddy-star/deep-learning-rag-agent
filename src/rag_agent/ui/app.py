"""Part 1 Markdown RAG chatbot using Streamlit, Groq and ChromaDB."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

load_dotenv()
ROOT = Path(__file__).resolve().parents[3]
DB_DIR = ROOT / 'data' / 'chroma_part1'
MODEL_NAME = 'all-MiniLM-L6-v2'

st.set_page_config(page_title='Deep Learning RAG Chatbot', page_icon='🧠', layout='wide')
st.title('🧠 Deep Learning RAG Chatbot')
st.caption('Part 1 • Upload Markdown notes, ingest them, and ask grounded questions.')

@st.cache_resource
def embedding_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(name='workshop_notes')

@st.cache_resource
def llm():
    key = os.getenv('GROQ_API_KEY', '').strip()
    if not key:
        raise RuntimeError('GROQ_API_KEY is missing from your .env file.')
    return ChatGroq(api_key=key, model=os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b'), temperature=0)

def split_markdown(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """Split on paragraphs, with a sliding window for long passages."""
    paragraphs = [p.strip() for p in text.replace('\r\n', '\n').split('\n\n') if p.strip()]
    chunks, current = [], ''
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= chunk_size:
            current = f'{current}\n\n{paragraph}'.strip()
        else:
            if current:
                chunks.append(current)
            current = paragraph
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap:]
    if current:
        chunks.append(current)
    return chunks

def ingest(uploaded_files) -> tuple[int, int]:
    db = collection()
    added = skipped = 0
    for upload in uploaded_files:
        raw = upload.getvalue()
        try:
            content = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            st.sidebar.error(f'{upload.name} is not UTF-8 text.')
            continue
        for index, chunk in enumerate(split_markdown(content)):
            digest = hashlib.sha256((upload.name + '\n' + chunk).encode('utf-8')).hexdigest()
            if db.get(ids=[digest])['ids']:
                skipped += 1
                continue
            vector = embedding_model().encode(chunk).tolist()
            db.add(ids=[digest], embeddings=[vector], documents=[chunk],
                   metadatas=[{'source': upload.name, 'chunk': index + 1}])
            added += 1
    return added, skipped

def retrieve(question: str, count: int = 4) -> list[dict]:
    db = collection()
    if db.count() == 0:
        return []
    vector = embedding_model().encode(question).tolist()
    results = db.query(query_embeddings=[vector], n_results=min(count, db.count()),
                       include=['documents', 'metadatas', 'distances'])
    return [{'text': text, 'source': meta.get('source', 'Unknown'),
             'chunk': meta.get('chunk', '?'), 'distance': distance}
            for text, meta, distance in zip(results['documents'][0],
                                             results['metadatas'][0],
                                             results['distances'][0])]

with st.sidebar:
    st.header('📂 Document ingestion')
    uploaded_files = st.file_uploader('Upload Markdown study notes', type=['md'],
                                     accept_multiple_files=True)
    if st.button('Ingest Documents', disabled=not uploaded_files, use_container_width=True):
        try:
            with st.spinner('Embedding and saving notes...'):
                added, skipped = ingest(uploaded_files)
            st.success(f'{added} chunks added; {skipped} duplicates skipped.')
        except Exception as error:
            st.error(f'Ingestion failed: {error}')
    try:
        db = collection()
        st.metric('Stored chunks', db.count())
        data = db.get(include=['metadatas'])
        sources = sorted({meta.get('source', 'Unknown') for meta in data['metadatas']})
        if sources:
            st.subheader('Documents')
            for source in sources:
                st.write('📄', source)
    except Exception as error:
        st.error(f'Database unavailable: {error}')

if 'messages' not in st.session_state:
    st.session_state.messages = []

st.subheader('💬 Ask questions about your notes')
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('sources'):
            with st.expander('📎 Sources'):
                for source in message['sources']:
                    st.caption(source)

question = st.chat_input('Ask a question about your uploaded Markdown notes...')
if question:
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        try:
            passages = retrieve(question)
            if not passages:
                answer = 'Please upload and ingest a Markdown note before asking questions.'
                source_labels = []
            else:
                context = '\n\n'.join(
                    f"[Source: {p['source']}, chunk {p['chunk']}]\n{p['text']}"
                    for p in passages
                )
                prompt = (
                    'Answer the question using ONLY the supplied document context. '
                    'If the context does not contain the answer, say you cannot find it '
                    'in the uploaded notes. Be clear and concise, and cite filenames.\n\n'
                    f'Context:\n{context}\n\nQuestion: {question}'
                )
                with st.spinner('Generating an answer...'):
                    result = llm().invoke(prompt)
                answer = result.content if isinstance(result.content, str) else str(result.content)
                source_labels = list(dict.fromkeys(
                    f"{p['source']} (chunk {p['chunk']})" for p in passages
                ))
            st.markdown(answer)
            if source_labels:
                with st.expander('📎 Sources', expanded=True):
                    for label in source_labels:
                        st.caption(label)
            st.session_state.messages.append({'role': 'assistant',
                                              'content': answer, 'sources': source_labels})
        except Exception as error:
            st.error(f'Unable to answer: {error}')
