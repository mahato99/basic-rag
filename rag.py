import os
import warnings
from typing import List
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings('ignore')

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and return its documents.
    
    Args:
        file_path (str): Path to the PDF file
    
    Returns:
        List[Document]: List of extracted documents
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int =100) -> List[Document]:
    """
    Split documents into smaller chunks.
    
    Args:
        documents (List[Document]): Input documents
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Number of characters to overlap between chunks
    
    Returns:
        List[Document]: Split documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents: List[Document], embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    """
    Create a vector store from documents.
    
    Args:
        documents (List[Document]): Input documents
        embedding_model (str): Hugging Face embedding model to use
    
    Returns:
        Chroma: Vector store
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        return Chroma.from_documents(documents, embeddings)
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def create_qa_chain(vector_store, model: str = 'gpt-3.5-turbo'):
    """
    Create a question-answering chain.
    
    Args:
        vector_store: Vector store to use as retriever
        model (str): OpenAI model to use
    
    Returns:
        RetrievalQA: Question-answering chain
    """
    try:
        # Use ChatOpenAI instead of the deprecated OpenAI
        llm = ChatOpenAI(model_name=model, temperature=0.1)
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}  # Retrieve top 3 most similar chunks
        )
        return RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return None

def main():
    # PDF file path
    pdf_path = "ril.pdf"
    
    # Load PDF
    documents = load_pdf(pdf_path)
    if not documents:
        return "No Document was found."
    
    # Split documents
    split_docs = split_documents(documents)
    
    # Create vector store
    vector_store = create_vector_store(split_docs)
    if not vector_store:
        return "No vector store"
    
    # Create QA chain
    qa_chain = create_qa_chain(vector_store)
    if not qa_chain:
        return
    
    # Example query
    query = "Who are the auditors for the financial year of Reliance?"
    
    try:
        # Run query
        response = qa_chain.invoke({"query": query})
        
        # Print response and source documents
        print("Answer:", response['result'])
        print("\nSource Documents:")
        for doc in response['source_documents']:
            print(f"- Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}...")
    
    except Exception as e:
        print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()