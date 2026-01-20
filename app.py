import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page configuration
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdfs_processed' not in st.session_state:
    st.session_state.pdfs_processed = False
if 'uploaded_pdf_names' not in st.session_state:
    st.session_state.uploaded_pdf_names = []

# Directory to store vector databases
VECTORSTORE_DIR = "vectorstore_data"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

def format_docs(docs):
    """Format documents for the prompt"""
    return "\n\n".join(doc.page_content for doc in docs)

def process_pdfs(uploaded_files, api_key):
    """Process uploaded PDFs and create vector store"""
    os.environ["OPENAI_API_KEY"] = api_key
    
    all_documents = []
    
    # Save uploaded files temporarily and load them
    with st.spinner("Loading PDFs..."):
        for uploaded_file in uploaded_files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            all_documents.extend(documents)
            
            # Clean up temp file
            os.unlink(tmp_path)
        
        st.success(f"Loaded {len(uploaded_files)} PDFs with {len(all_documents)} pages")
    
    # Split into chunks
    with st.spinner("Creating text chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(all_documents)
        st.success(f"Created {len(chunks)} chunks")
    
    # Create embeddings and vector store
    with st.spinner("Creating embeddings... This may take a minute."):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.vectorstore = vectorstore
        
        # Save to disk
        vectorstore_path = os.path.join(VECTORSTORE_DIR, "faiss_index")
        vectorstore.save_local(vectorstore_path)
        
        # Save PDF names
        st.session_state.uploaded_pdf_names = [f.name for f in uploaded_files]
        
        st.success("Vector store created and saved to disk!")
    
    # Setup QA chain
    with st.spinner("Setting up QA system..."):
        # Create prompt template
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Create chain
        qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        st.session_state.qa_chain = qa_chain
        st.session_state.pdfs_processed = True
        st.success("✅ Ready to answer questions!")

def load_existing_vectorstore(api_key):
    """Load previously saved vector store"""
    os.environ["OPENAI_API_KEY"] = api_key
    vectorstore_path = os.path.join(VECTORSTORE_DIR, "faiss_index")
    
    if os.path.exists(vectorstore_path):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            vectorstore_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        st.session_state.vectorstore = vectorstore
        
        # Setup QA chain
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        st.session_state.qa_chain = qa_chain
        st.session_state.pdfs_processed = True
        return True
    return False

def ask_question(question):
    """Get answer for a question"""
    if not st.session_state.qa_chain:
        return "Please upload and process PDFs first.", []
    
    # Get answer
    answer = st.session_state.qa_chain.invoke(question)
    
    # Get source documents separately
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    sources = retriever.get_relevant_documents(question)
    
    return answer, sources

# Main UI
st.title("📚 PDF Chatbot with RAG")
st.markdown("Upload your PDFs and ask questions about their content!")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    
    # Try to load existing vectorstore
    if api_key and not st.session_state.pdfs_processed:
        if load_existing_vectorstore(api_key):
            st.success("📂 Loaded previously processed PDFs!")
    
    st.markdown("---")
    st.header("📄 Upload PDFs")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files"
    )
    
    if uploaded_files and api_key:
        if st.button("Process PDFs", type="primary"):
            process_pdfs(uploaded_files, api_key)
    elif uploaded_files and not api_key:
        st.warning("Please enter your OpenAI API key first")
    
    st.markdown("---")
    
    if st.session_state.pdfs_processed:
        st.success("✅ PDFs Processed")
        if st.session_state.uploaded_pdf_names:
            st.info(f"📚 Loaded: {', '.join(st.session_state.uploaded_pdf_names)}")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("Delete Saved Data", type="secondary"):
            vectorstore_path = os.path.join(VECTORSTORE_DIR, "faiss_index")
            if os.path.exists(vectorstore_path):
                import shutil
                shutil.rmtree(vectorstore_path)
            st.session_state.vectorstore = None
            st.session_state.qa_chain = None
            st.session_state.pdfs_processed = False
            st.session_state.chat_history = []
            st.session_state.uploaded_pdf_names = []
            st.success("Data deleted!")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This chatbot uses:
    - **RAG** (Retrieval Augmented Generation)
    - **LangChain** for orchestration
    - **FAISS** for vector storage
    - **OpenAI** for embeddings & chat
    """)

# Main chat interface
if not st.session_state.pdfs_processed:
    st.info("👈 Please upload PDFs and enter your API key in the sidebar to get started")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📖 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}.** {source}")
    
    # Chat input
    if question := st.chat_input("Ask a question about your PDFs..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = ask_question(question)
                st.markdown(answer)
                
                # Format sources
                source_list = []
                for i, doc in enumerate(sources, 1):
                    source_file = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'Unknown')
                    source_list.append(f"Page {page + 1} from uploaded PDF")
                
                with st.expander("📖 Sources"):
                    for i, source in enumerate(source_list, 1):
                        st.markdown(f"**{i}.** {source}")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_list
                })

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit, LangChain, and OpenAI</p>
    </div>
    """,
    unsafe_allow_html=True
)
