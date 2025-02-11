import os
import time
import argparse
from typing import Optional, List
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

def init_pinecone(index_name: str, 
                  pinecone_api_key: str, 
                  openai_api_key: str, 
                  namespace: Optional[str] = None, 
                  search_type: str = "similarity", 
                  search_kwargs: Optional[dict] = None):
    """
    Initialize the Pinecone vector database and return a retriever.
    
    Args:
        index_name (str): Pinecone index name
        pinecone_api_key (str): Pinecone API key
        openai_api_key (str): OpenAI API key
        namespace (Optional[str]): Namespace for queries (optional)
        search_type (str): Search type ("similarity" or "mmr")
        search_kwargs (Optional[dict]): Additional search parameters (e.g., k value)
    
    Returns:
        vector_retriever: A retriever for vector search
    """
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        raise ValueError(f"❌ Error: Index '{index_name}' does not exist. Please create it first.")

    print(f"✅ Index '{index_name}' found. Checking status...")

    while not pc.describe_index(index_name).status["ready"]:
        print(f"⏳ Waiting for Index '{index_name}' to become ready...")
        time.sleep(1)

    print(f"✅ Index '{index_name}' is ready for use.")

    # Set up embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings, text_key="text", namespace=namespace)

    # Set default search parameters
    search_kwargs = search_kwargs or {"k": 2}
    
    # Choose search mode
    if search_type == "similarity":
        vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    elif search_type == "mmr":
        vector_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    else:
        raise ValueError(f"❌ Error: Unsupported search_type '{search_type}'. Use 'similarity' or 'mmr'.")

    return vectorstore, vector_retriever

def retrieve_data(query: str, retriever, return_metadata: bool = False) -> List[dict]:
    """
    Retrieve relevant data from Pinecone.
    
    Args:
        query (str): Query string
        retriever: Previously initialized retriever
        return_metadata (bool): Whether to return metadata information

    Returns:
        List[dict]: Retrieved search results
    """
    docs = retriever.invoke(query)
    
    if return_metadata:
        return [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]
    return [doc.page_content for doc in docs]

def main():
    parser = argparse.ArgumentParser(description="Retrieve information from Pinecone vector database")
    parser.add_argument("--index", type=str, required=True, help="Pinecone index name")
    parser.add_argument("--search_type", type=str, choices=["similarity", "mmr"], default="similarity", help="Search type")
    parser.add_argument("--query", type=str, required=True, help="Query string to search for")
    
    args = parser.parse_args()

    # Read API keys from environment variables
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not pinecone_api_key or not openai_api_key:
        raise ValueError("❌ Error: Please set PINECONE_API_KEY and OPENAI_API_KEY environment variables.")

    # Initialize retriever
    _, vector_retriever = init_pinecone(args.index, pinecone_api_key, openai_api_key, search_type=args.search_type)

    # Perform query
    results = retrieve_data(args.query, vector_retriever, return_metadata=True)

    print("\n🔍 Search Results:")
    for i, res in enumerate(results):
        print(f"{i+1}. {res['text']}")
        print(f"   📌 Metadata: {res['metadata']}\n")

if __name__ == "__main__":
    main()
