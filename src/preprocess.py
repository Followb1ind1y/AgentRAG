import os
import getpass
import argparse
import time
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pprint import pprint
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4

def init_pinecone(index_name, pinecone_api_key):
    """Initialize Pinecone connection and ensure the index exists"""
    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    # Create the index if it doesn't exist
    if index_name not in existing_indexes:
        print(f"Index {index_name} does not exist, creating...")
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI text-embedding-3-small dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    
    print(f"✅ Index {index_name} is ready")

def fetch_webpage(urls):
    """Fetch webpages and extract text content"""
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, unwanted_tags=["script", "style", "header", "footer", "nav", "aside"], tags_to_extract=["p"]
    )
    return docs_transformed

def process_websites(urls, index_name, pinecone_api_key, openai_api_key):
    """Scrape webpages, extract text, vectorize, and store into Pinecone"""
    print("Starting to fetch webpages...")
    docs_transformed = fetch_webpage(urls)

    print("Text chunking...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks_with_metadata = []

    for doc in docs_transformed:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunks_with_metadata.append({
                "text": chunk,
                "metadata": {
                    "source": doc.metadata.get("source", ""),
                    "title": doc.metadata.get("title", ""),
                }
            })

    print("Vectorizing and storing in Pinecone...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    documents = [
        Document(page_content=item["text"], metadata=item["metadata"])
        for item in chunks_with_metadata
    ]

    uuids = [str(uuid4()) for _ in range(len(documents))]

    init_pinecone(index_name, pinecone_api_key)

    vector_store =PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
        ids=uuids
    )

    print(f"Successfully stored {len(documents)} chunks to index {index_name}")
    return vector_store

def main():
    parser = argparse.ArgumentParser(description="Fetch webpages and store them into Pinecone vector database")
    parser.add_argument("--urls", nargs="+", required=True, help="List of URLs to scrape")
    parser.add_argument("--index_name", type=str, default="web-rag", help="Pinecone index name")
    parser.add_argument("--pinecone_api_key", type=str, help="Pinecone API key (optional, will read from environment variable if not provided)")
    parser.add_argument("--openai_api_key", type=str, help="Openai API key (optional, will read from environment variable if not provided)")

    args = parser.parse_args()

    # Handle Pinecone API Key
    pinecone_api_key = args.pinecone_api_key or os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        pinecone_api_key = getpass.getpass("Please enter Pinecone API Key: ")
        os.environ["PINECONE_API_KEY"] = pinecone_api_key  # Store for later use
    
    # Handle Openai API Key
    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = getpass.getpass("Please enter Openai API Key: ")
        os.environ["OPENAI_API_KEY"] = openai_api_key  # Store for later use

    # Run the main process
    vector_store = process_websites(args.urls, args.index_name, pinecone_api_key, openai_api_key)

if __name__ == "__main__":
    main()
