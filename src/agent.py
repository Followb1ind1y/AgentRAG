import os
import datetime
import uuid
import argparse
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from retriever import init_pinecone, retrieve_data
from dotenv import load_dotenv
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Initialize Pinecone and OpenAI API keys
INDEX_NAME = "wiki-mini"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize Retriever
vector_store, vector_retriever = init_pinecone(INDEX_NAME, PINECONE_API_KEY, OPENAI_API_KEY, search_type="similarity")


def retrieve(query: str):
    """Retrieve information related to a query from the vector store."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Define tools
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
    insights and conclusions from the discussion in a clear and concise manner for future reference and learning. Summarize in 200 Words.

    {conversation_text}
    """

    # Generate summary using GPT
    summary_response = model.invoke([{"role": "user", "content": summary_prompt}])
    return summary_response.content


def replace_unicode_chars(text: str) -> str:
    """Replace problematic Unicode characters with ASCII equivalents."""
    replacements = {
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def save_summary_to_pdf(summary: str):
    """Save the summary to a PDF file."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Chatbot Conversation Summary", ln=True, align="C")
    pdf.ln(10)

    summary = replace_unicode_chars(summary)

    for line in summary.split("\n"):
        pdf.multi_cell(0, 10, line)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"chat_summary_{timestamp}.pdf"
    pdf.output(pdf_filename)
    print(f"✅ Conversation summary saved as {pdf_filename}")


def chatbot_interaction():
    """Run interactive chatbot session."""
    thread_id = str(uuid.uuid4())
    print("\n🤖 AI Chatbot (Type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("\n📌 Generating Summary...\n")
            summary = generate_summary(thread_id)
            print(summary)
            save_summary_to_pdf(summary)
            print("Goodbye! 👋")
            break

        config = {"configurable": {"thread_id": thread_id}}
        # for event in agent_executor.stream(
        #     {"messages": [{"role": "user", "content": user_input}]},
        #     stream_mode="values",
        #     config=config,
        # ):
        #     event["messages"][-1].pretty_print()
        response = agent_executor.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        response["messages"][-1].pretty_print()


def main():
    """Parse arguments and run the chatbot."""
    parser = argparse.ArgumentParser(description="AI Chatbot with Retrieval & Web Search")
    parser.add_argument("--mode", type=str, choices=["chat"], default="chat", help="Run chatbot in interactive mode")

    args = parser.parse_args()

    if args.mode == "chat":
        chatbot_interaction()


if __name__ == "__main__":
    main()