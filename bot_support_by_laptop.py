import os, json
from dotenv import load_dotenv
from typing import TypedDict, Optional
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.tools import tool
from langchain_tavily import TavilySearch
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import TavilySearchResults
from pinecone import Pinecone
from langgraph.graph import StateGraph, START, END
from langchain.messages import AIMessage

# ===============================
# Load ENV
load_dotenv()
# ===============================

# ===============================
# TypedDict for Messages State
class MessagesState(TypedDict):
 messages: list



# ===============================
# 3️⃣ Setup Embedding + Pinecone + Tavily Search
# ===============================
embedding_model = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
    api_version="2023-05-15"
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("bot-laptop-index")

vectorstore = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")
retriever = vectorstore.as_retriever(search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 3})


# ===============================
# Function tool demo
# ===============================
@tool
def search_laptop_specs(model: str) -> str:
    """Search for detailed specifications of a laptop model on Pinecone DB."""
    query_result = retriever.invoke(model)
    if not query_result:
        return f"No specifications found for {model}."
    specs = query_result[0].page_content
    return f"Specifications for {model}:\n{specs}"

@tool
def check_availability(model: str) -> str:
    """Check the availability of a laptop model using Tavily Search."""
    tavily_search_tool = TavilySearchResults(
        max_results=1,
        topic="general",
    )
    results = tavily_search_tool.invoke({"query": f"Check availability of {model} laptop in Shopee VN"})
    return f"Availability for {model}:\n{results}"

@tool
def check_price(model: str) -> str:
    """Check the price of a laptop model using Tavily Search."""
    tavily_search_tool = TavilySearchResults(
        max_results=1,
        topic="news",
    )
    results = tavily_search_tool.invoke({"query": f"Check price of {model} laptop"})
    return f"Price for {model}:\n{results}"


# ===============================
# 4️⃣ Memory + Chain setup
# ===============================
chat_llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version="2023-05-15",
    temperature=1,
)

tools = [search_laptop_specs, check_availability, check_price]

llm_with_tools = chat_llm.bind_tools(tools=tools)

def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def tool_node(state: MessagesState):
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return state
    
    for tool_call in last_message.tool_calls:
        tool = tool_call["name"]
        for t in tools:
            if t.name == tool:
                response = t.invoke(tool_call)
                state["messages"].append(response)
                break

    return state

# --- BUILD THE GRAPH ---
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("call_model", call_model)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "call_model")
graph_builder.add_conditional_edges("call_model", should_continue, ["tools",
END])
graph_builder.add_edge("tools", "call_model")

def process_user_message(user_input: str, chat_history: list):
    graph = graph_builder.compile()
    messages = [{"role": "system", "content": "You are a helpful assistant for laptops."}]
    for q, a in chat_history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_input})

    response = graph.invoke({"messages": messages})

    # Log response for debugging
    print("Response:", response)

        # Access the content of the AIMessage object correctly
    last_message = response["messages"][-1]
    return last_message.content if isinstance(last_message, AIMessage) else last_message["content"]
