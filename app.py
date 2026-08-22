import streamlit as st
import os
from google import genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Page config
st.set_page_config(
    page_title='AI Assistant',
    page_icon='🤖',
    layout='wide'
)

# Title
st.title('🤖 Company Knowledge Assistant')
st.markdown('Ask me anything about company policies!')

# Gemini client
api_key = st.secrets.get('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key)

def get_embedding(text):
    response = client.models.embed_content(
        model='gemini-embedding-001',
        contents=text
    )
    return response.embeddings[0].values

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass
    def __call__(self, input: Documents) -> Embeddings:
        return [get_embedding(text) for text in input]

@st.cache_resource
def init_chromadb():
    """
    Initialize ChromaDB (cached - runs once)
    """
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chroma_client = chromadb.PersistentClient(path='./chroma_db')
    gemini_ef = GeminiEmbeddingFunction()
    collection = chroma_client.get_or_create_collection(
        name='company_docs',
        embedding_function=gemini_ef
    )

    if collection.count() == 0:
        loader = DirectoryLoader('company_docs/', glob='*.txt', loader_cls=TextLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        collection.add(
            documents=[chunk.page_content for chunk in chunks],
            ids=[f'doc_{i}' for i in range(len(chunks))],
            metadatas=[{'source': chunk.metadata.get('source', 'unknown')} for chunk in chunks]
        )

    return collection

collection = init_chromadb()

def get_rag_response(query, n_results=3):
    """
    Get answer using RAG
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        if not results['documents'][0]:
            return 'No relevant information found in documents.'
        
        context = '\n\n---\n\n'.join(results['documents'][0])
        
        prompt = f'''You are a helpful HR assistant. Answer using ONLY the context below. If not in context, say so. Be concise and friendly.

Context:
{context}

Question: {query}

Answer:'''
        
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f'Error: {str(e)}. Please try again.'

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header('About')
    st.markdown('''
    This AI assistant can answer questions about:
    - Vacation policies
    - Remote work guidelines
    - Parental leave
    - Benefits information

    Powered by:
    - Google Gemini
    - ChromaDB vector search
    - Semantic RAG
    ''')
    st.divider()
    st.metric('Documents Indexed', collection.count())
    st.metric('Messages in Chat', len(st.session_state.messages))
    st.divider()
    if st.button('Clear Chat History'):
        st.session_state.messages = []
        st.rerun()

# Welcome message
if len(st.session_state.messages) == 0:
    welcome = '''
    👋 Hi! I'm your company knowledge assistant. I can help you find information about:
    - Vacation and time off policies
    - Remote work guidelines
    - Parental leave benefits
    - And more!

    Just ask me a question to get started.
    '''
    with st.chat_message('assistant'):
        st.write(welcome)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

# Chat input
if prompt := st.chat_input('Ask a question...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.write(prompt)
    
    with st.chat_message('assistant'):
        with st.spinner('Searching documents...'):
            response = get_rag_response(prompt)
            st.write(response)
    
    st.session_state.messages.append({'role': 'assistant', 'content': response})