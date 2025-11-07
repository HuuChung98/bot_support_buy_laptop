import streamlit as st
from bot_support_by_laptop import process_user_message

st.set_page_config(page_title="💻 Copilot RAG", page_icon="🤖", layout="wide")

st.markdown("<h1 style='text-align:center;'>💬 Laptop Assistant</h1>", unsafe_allow_html=True)
# st.write("A Conversational RAG + Function Calling assistant powered by Azure OpenAI & Pinecone")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Type your message...")

# If user sends a message
if user_input:
    # 1️⃣ Hiển thị ngay câu hỏi
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2️⃣ Gọi logic xử lý (RAG + Function)
    with st.spinner("Thinking..."):
        final_answer, rag_result = process_user_message(user_input, st.session_state.chat_history)

    # 3️⃣ Hiển thị phản hồi assistant
    with st.chat_message("assistant"):
        st.markdown(final_answer)

    # 4️⃣ Lưu vào session
    st.session_state.chat_history.append((user_input, final_answer))

# 5️⃣ Hiển thị lịch sử hội thoại
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# 6️⃣ Hiển thị tài liệu nguồn nếu có
if "rag_result" in locals() and rag_result.get("source_documents"):
    with st.expander("📚 Retrieved Sources"):
        for doc in rag_result["source_documents"]:
            st.write("- ", doc.page_content[:200], "...")
