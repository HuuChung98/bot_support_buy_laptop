import os, json
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone, ServerlessSpec
from openai import AzureOpenAI

load_dotenv()

# ===============================
# 1️⃣ Function tools
# ===============================
def check_system_status(device_id: str) -> str:
    status_map = {
        "printer01": "Online and functioning normally.",
        "router23": "Offline - requires restart.",
        "server07": "Online but high CPU usage.",
    }
    return status_map.get(device_id, "Device not found.")

# ===============================
# 2️⃣ Embedding + Pinecone setup
# ===============================
embedding_model = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
    api_version="2023-05-15"
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "bot-laptop-index"

if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# ===============================
# 3️⃣ LangChain setup
# ===============================
chat_llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version="2023-05-15",
    temperature=1,
)

vectorstore = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")
retriever = vectorstore.as_retriever()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",   # ✅ chỉ đặt ở đây
    output_key="answer",    # ✅ chỉ đặt ở đây 
    return_messages=True
)

retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer",
)

# ===============================
# 4️⃣ Function calling metadata
# ===============================
functions = [
    {
        "name": "check_system_status",
        "description": "Check IT device status.",
        "parameters": {
            "type": "object",
            "properties": {"device_id": {"type": "string"}},
            "required": ["device_id"],
        },
    }
]

def call_function_if_needed(message, user_input, chat_history):
    if not getattr(message, "function_call", None):
        return None, None
    func_name = message.function_call.name
    args = json.loads(message.function_call.arguments)
    if func_name == "check_system_status":
        return check_system_status(args["device_id"]), chat_history
    return None, None


def process_user_message(user_input: str, chat_history: list):
    """Main logic: RAG + function calling."""
    # 1️⃣ RAG retrieval
    rag_result = retrieval_chain({"question": user_input})
    answer = rag_result["answer"]

    # 2️⃣ Function calling check
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_LLM_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_LLM_ENDPOINT"),
        api_version="2023-05-15",
    )

    messages = [{"role": "system", "content": "You are a helpful assistant for laptops."}]
    for q, a in chat_history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    message = response.choices[0].message

    func_result, _ = call_function_if_needed(message, user_input, chat_history)
    final_answer = func_result if func_result else answer
    return final_answer, rag_result
