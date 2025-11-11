import streamlit as st
from bot_support_by_laptop import process_user_message

# Set Streamlit page and header
st.set_page_config(page_title="💻 Copilot RAG", page_icon="🤖", layout="wide")

st.markdown("<h1 style='text-align:center;'>💬 Laptop Copilot Assistant</h1>", unsafe_allow_html=True)
st.write("A Conversational RAG + Function Calling assistant powered by Azure OpenAI & Pinecone")

# =====================================
# State: store chat history
# =====================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================
# Render previous chat history (before the user sends a new message)
# =====================================
for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

# =====================================
# Handle new incoming message
# =====================================
user_input = st.chat_input("Type your message...")

if user_input:
    # 1️⃣ Immediately display the user's question
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2️⃣ Process RAG (retrieval) + function calling
    with st.spinner("Thinking..."):
        final_answer, rag_result = process_user_message(user_input, st.session_state.chat_history)

    # 3️⃣ Display the assistant's answer
    with st.chat_message("assistant"):
        st.markdown(final_answer)

    # 4️⃣ Save the exchange to session state
    st.session_state.chat_history.append((user_input, final_answer))

    # 5️⃣ Show retrieved sources (if any)
    # if rag_result.get("source_documents"):
    #     with st.expander("📚 Retrieved Sources"):
    #         for doc in rag_result["source_documents"]:
    #             st.write("- ", doc.page_content[:200], "...")
