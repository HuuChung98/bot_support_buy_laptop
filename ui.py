import streamlit as st
from bot_support_by_laptop import process_user_message

st.set_page_config(page_title="💻 Copilot RAG", page_icon="🤖", layout="wide")

st.markdown("<h1 style='text-align:center;'>💬 Laptop Copilot Assistant</h1>", unsafe_allow_html=True)
st.write("A Conversational RAG + Function Calling assistant powered by Azure OpenAI & Pinecone")

# =====================================
# State lưu lịch sử chat
# =====================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================
# Render lịch sử cũ (trước khi người dùng gửi mới)
# =====================================
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# =====================================
# Xử lý tin nhắn mới
# =====================================
user_input = st.chat_input("Type your message...")

if user_input:
    # 1️⃣ Hiển thị ngay câu hỏi người dùng
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2️⃣ Xử lý RAG + function calling
    with st.spinner("Thinking..."):
        final_answer, rag_result = process_user_message(user_input, st.session_state.chat_history)

    # 3️⃣ Hiển thị câu trả lời assistant
    with st.chat_message("assistant"):
        st.markdown(final_answer)

    # 4️⃣ Lưu vào session
    st.session_state.chat_history.append((user_input, final_answer))

    # 5️⃣ Hiển thị nguồn (nếu có)
    if rag_result.get("source_documents"):
        with st.expander("📚 Retrieved Sources"):
            for doc in rag_result["source_documents"]:
                st.write("- ", doc.page_content[:200], "...")
