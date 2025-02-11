from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uuid
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from src.retriever import init_pinecone, retrieve_data
from dotenv import load_dotenv
import os
import datetime

# Load environment variables
load_dotenv()

# Initialize Pinecone and OpenAI API keys
INDEX_NAME = "wiki-mini"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize Retriever
vector_store, vector_retriever = init_pinecone(INDEX_NAME, PINECONE_API_KEY, OPENAI_API_KEY, search_type="similarity")

app = FastAPI()

class QueryRequest(BaseModel):
    user_input: str
    thread_id: str = None

class SummaryRequest(BaseModel):
    thread_id: str

# Define tools
def retrieve(query: str):
    """Retrieve information related to a query from the vector store."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
    )
    return serialized, retrieved_docs

retrieve_tool = Tool(
    name="retrieve",
    func=retrieve,
    description="Retrieve relevant documents from Pinecone's vector store based on query similarity."
)

search_tool = Tool(
    name="search",
    func=TavilySearchResults(max_results=2).invoke,
    description="Fetch additional relevant data from the web if the retrieve tool is insufficient."
)

# Initialize AI model
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=300,
    max_retries=2,
)

memory = MemorySaver()

# Define system prompt
SYSTEM_PROMPT = """You are an advanced AI assistant capable of retrieving and synthesizing information from multiple sources. 
Follow these steps strictly:

1. **First, use `retrieve_tool`** to check for relevant information in the internal knowledge base.
2. **Evaluate the retrieved information**:
   - If it is **complete and sufficient**, generate the response using only this data.
   - If it is **incomplete, outdated, or missing**, proceed to the next step.
3. **Use `search_tool`** to gather additional or updated information.
4. **Merge the results** ensuring accuracy and clarity.

⚠️ Important:
- Prioritize `retrieve_tool` first.
- If additional information is used, indicate it.
- If contradictions exist, highlight and explain them.
- Ensure responses are well-structured and clear.
"""

# Create agent executor
agent_executor = create_react_agent(model, [retrieve_tool, search_tool], checkpointer=memory, prompt=SYSTEM_PROMPT)

@app.post("/chat/")
async def chat(request: QueryRequest):
    """Handle chat requests and return AI response."""
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Get the full response (non-streaming version)
        response_events = agent_executor.invoke(
            {"messages": [{"role": "user", "content": request.user_input}]},
            config=config
        )
        
        # Extract the response text from the latest message
        response_text = response_events["messages"][-1].content
        
        return {"thread_id": thread_id, "response": response_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API Route for Summarizing the Conversation
@app.post("/summary/")
async def summary(request: SummaryRequest):
    """Generate a summary of the conversation history."""
    try:
        summary_text = generate_summary(request.thread_id)
        return {"thread_id": request.thread_id, "summary": summary_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_summary(thread_id: str):
    """Extract conversation history and generate a summary using GPT."""
    config = {"configurable": {"thread_id": thread_id}}
    messages = list(memory.list(config, limit=2))

    # Extract all conversation messages
    all_messages = []
    for msg in messages:
        all_messages.extend(msg.checkpoint["channel_values"]["messages"])

    conversation_text = "\n\n".join([f"{m.__class__.__name__}: {m.content}" for m in all_messages])

    summary_prompt = f"""
    Summarize the following conversation history into a cohesive paragraph that integrates key points from all answers 
    given to the user's questions. The summary should be structured as a continuous explanation, consolidating the main 
    insights and conclusions from the discussion in a clear and concise manner for future reference and learning.

    {conversation_text}
    """

    # Generate summary using GPT
    summary_response = model.invoke([{"role": "user", "content": summary_prompt}])
    return summary_response.content

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Chat API"}

# To run the app with uvicorn, use this command in the terminal:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload