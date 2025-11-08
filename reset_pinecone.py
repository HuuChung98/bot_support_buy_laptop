from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "bot-laptop-index")

# Xóa index cũ
if index_name in [i["name"] for i in pc.list_indexes()]:
    pc.delete_index(index_name)
    print(f"🗑️ Deleted old index: {index_name}")

# Tạo lại index mới
pc.create_index(
    name=index_name,
    dimension=1536,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
print(f"🆕 Recreated index: {index_name}")
