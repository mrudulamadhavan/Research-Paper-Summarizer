import streamlit as st
import os
import tempfile
from modules.pdf_parser import extract_text_from_pdf
from modules.chunker import chunk_text
from modules.summarizer import generate_summary
from modules.embeddings import create_embeddings
from modules.vectorstore import create_vectorstore, retrieve_relevant_chunks
from modules.chatbot import ask_question_about_paper

# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(page_title="AI Research Paper Summarizer", layout="wide")
st.title("🧠 AI Research Paper Summarizer")
st.markdown("Upload an academic paper PDF to generate summaries and chat with its content.")

# -------------------------------
# File Upload Section
# -------------------------------
uploaded_file = st.file_uploader("📄 Upload Research Paper (PDF)", type=["pdf"])

if uploaded_file:
    # Save the uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_path = temp_pdf.name

    st.success("✅ PDF uploaded successfully!")

    # -------------------------------
    # Step 1: Extract Text
    # -------------------------------
    with st.spinner("📖 Extracting text from PDF..."):
        text = extract_text_from_pdf(temp_path)

    if not text.strip():
        st.error("Could not extract text from the PDF. Please check the file.")
        st.stop()

    st.success("✅ Text extracted successfully!")

    # -------------------------------
    # Step 2: Chunk Text
    # -------------------------------
    chunk_size = st.slider("Select chunk size", 500, 2000, 1000)
    chunks = chunk_text(text, chunk_size=chunk_size)
    st.info(f"Text divided into {len(chunks)} chunks for processing.")

    # -------------------------------
    # Step 3: Generate Summary
    # -------------------------------
    if st.button("🧾 Generate Summary"):
        with st.spinner("Generating structured summary using LLM..."):
            summary = generate_summary(chunks)

        st.subheader("📑 Paper Summary")
        st.markdown(f"**Abstract:** {summary['abstract']}")
        st.markdown(f"**Methods:** {summary['methods']}")
        st.markdown(f"**Results:** {summary['results']}")
        st.markdown(f"**Insights:** {summary['insights']}")

        # Save summary
        os.makedirs("data/summaries", exist_ok=True)
        with open("data/summaries/summary.txt", "w", encoding="utf-8") as f:
            for key, value in summary.items():
                f.write(f"{key.upper()}:\n{value}\n\n")

    # -------------------------------
    # Step 4: Create Embeddings (for Q&A)
    # -------------------------------
    if st.button("🔍 Prepare Paper for Q&A"):
        with st.spinner("Creating embeddings..."):
            embeddings = create_embeddings(chunks)
            vectorstore = create_vectorstore(chunks, embeddings)
        st.success("✅ Paper indexed successfully! You can now chat with it.")

    # -------------------------------
    # Step 5: Q&A Chatbot Interface
    # -------------------------------
    st.header("💬 Chat with the Paper")

    user_query = st.text_input("Ask a question about the paper:")
    if st.button("Ask"):
        try:
            if "vectorstore" not in locals():
                st.warning("Please click 'Prepare Paper for Q&A' first.")
            elif not user_query.strip():
                st.info("Enter a question to start chatting.")
            else:
                with st.spinner("Thinking..."):
                    context_chunks = retrieve_relevant_chunks(user_query, vectorstore)
                    answer = ask_question_about_paper(user_query, context_chunks)
                st.success("✅ Answer generated!")
                st.markdown(f"**Answer:** {answer}")

        except Exception as e:
            st.error(f"Error during chat: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, LangChain & OpenAI")
