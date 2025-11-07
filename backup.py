from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone, ServerlessSpec
from openai import AzureOpenAI

import os
import json
from dotenv import load_dotenv
import openai as openai_sdk  # used specifically for function calling

# ===============================
# 0️⃣ Load environment variables
# ===============================
load_dotenv()

# ===============================
# 1️⃣ System status check function (demo function call)
# ===============================
def check_system_status(device_id: str) -> str:
    status_map = {
        "printer01": "Online and functioning normally.",
        "router23": "Offline - requires restart.",
        "server07": "Online but high CPU usage.",
    }
    return status_map.get(device_id, "Device not found.")

# ===============================
# 2️⃣ Laptop data
# ===============================
laptops = [
    {
        "id": "1",
        "name": "Gaming Beast Pro",
        "description": "A high-end gaming laptop with RTX 4080, 32GB RAM, and 1TB SSD. Perfect for hardcore gaming.",
        "tags": "gaming, high-performance, windows"
    },
    {
        "id": "2",
        "name": "Business Ultrabook X1",
        "description": "A lightweight business laptop with Intel i7, 16GB RAM, and long battery life. Great for productivity.",
        "tags": "business, ultrabook, lightweight"
    },
    {
        "id": "3",
        "name": "Student Basic",
        "description": "Affordable laptop with 8GB RAM, 256GB SSD, and a reliable battery. Ideal for students and general use.",
        "tags": "student, budget, general"
    },
]

# ===============================
# 3️⃣ Azure OpenAI Embeddings (standard)
# ===============================
embedding_model = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),  # ví dụ: text-embedding-3-small
    api_version="2023-05-15"
)

# ===============================
# 4️⃣ Pinecone setup
# ===============================
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "bot-laptop-index"

# Only create index if it doesn't exist
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# ===============================
# 5️⃣ Azure Chat Model (chuẩn LangChain)
# ===============================
chat = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
    api_version="2023-05-15",
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),  # ví dụ: gpt-4o-mini
    temperature=1,
)

# ===============================
# 6️⃣ Tạo embeddings và upsert vào Pinecone
# ===============================
def get_embedding(text: str):
    """Sinh embedding từ text."""
    return embedding_model.embed_query(text)

vectors = []
for p in laptops:
    embedding = get_embedding(p["name"] + " " + p["description"])
    vectors.append((p["id"], embedding, {"text": p["description"]}))

# upsert vào Pinecone
index.upsert(vectors)

# ===============================
# 7️⃣ LangChain retriever
# ===============================
vectorstore = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")
retriever = vectorstore.as_retriever()

retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=chat,
    retriever=retriever,
    return_source_documents=True,
)

# ===============================
# 8️⃣ Function metadata cho function calling
# ===============================
functions = [
    {
        "name": "recommend_laptop",
        "description": "Recommends the most suitable laptop based on user needs, budget, and preferences.",
        "parameters": {
            "type": "object",
            "properties": {
                "usage_purpose": {"type": "string"},
                "budget_range": {"type": "string"},
                "preferred_tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["usage_purpose"],
        },
    },
    {
        "name": "get_laptop_details",
        "description": "Get laptop details by ID.",
        "parameters": {
            "type": "object",
            "properties": {"laptop_id": {"type": "string"}},
            "required": ["laptop_id"],
        },
    },
    {
        "name": "check_system_status",
        "description": "Check IT device status.",
        "parameters": {
            "type": "object",
            "properties": {"device_id": {"type": "string"}},
            "required": ["device_id"],
        },
    },
]

system_prompt = (
    "You are a helpful assistant specializing in laptop recommendations. "
    "Use the context to suggest the best laptop for each query."
)

# ===============================
# 9️⃣ Function chat_with_functions
# ===============================
from openai import OpenAI

from openai import AzureOpenAI

def chat_with_functions(user_input, chat_history):
    """Function calling qua Azure OpenAI SDK v1."""
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
        api_version="2023-05-15",
    )

    messages = [{"role": "system", "content": system_prompt}]
    for q, a in chat_history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),  # ví dụ gpt-4o-mini
        messages=messages,
        functions=functions,
        function_call="auto"
    )

    message = response.choices[0].message
    if message.function_call:
        func_name = message.function_call.name
        args = json.loads(message.function_call.arguments)
        if func_name == "check_system_status":
            result = check_system_status(args["device_id"])
            chat_history.append((user_input, result))
            return result, chat_history

    reply = message.content or ""
    chat_history.append((user_input, reply))
    return reply, chat_history



# ===============================
# 🔟 Chạy batch queries
# ===============================
user_queries = [
    "I want a lightweight laptop with long battery life for business trips.",
    "I need a laptop for gaming with the best graphics card available.",
    "Looking for a budget laptop suitable for student tasks and general browsing.",
]

if __name__ == "__main__":
    chat_history = []
    print("=== Laptop Recommendation Chatbot ===\n")

    for query in user_queries:
        print(f"🧑 User: {query}")
        rag_result = retrieval_chain({"question": query, "chat_history": chat_history})
        print(f"🤖 RAG Answer: {rag_result['answer']}")
        func_answer, chat_history = chat_with_functions(query, chat_history)
        # print(f"⚙️ Function Call Answer: {func_answer}\n")
