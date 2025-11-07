import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import AzureOpenAIEmbeddings

# ===============================
# 1️⃣ Load ENV
# ===============================
load_dotenv()

# ===============================
# 2️⃣ Init Embedding + Pinecone
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
    print(f"🆕 Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# ===============================
# 3️⃣ Load JSON data
# ===============================
with open("data/laptops.json", "r", encoding="utf-8") as f:
    laptops = json.load(f)

# ===============================
# 4️⃣ Create embeddings + upsert
# ===============================
def get_embedding(text: str):
    return embedding_model.embed_query(text)

stats = index.describe_index_stats()
if stats["total_vector_count"] == 0:
    vectors = []
    for p in laptops:
        content = p["name"] + " " + p["description"]
        emb = get_embedding(content)
        vectors.append((p["id"], emb, {"text": content, "tags": p["tags"]}))
    index.upsert(vectors)
    print(f"✅ Upserted {len(vectors)} vectors into Pinecone index '{index_name}'.")
else:
    print("⚡ Index already contains data. Skipping upsert.")
