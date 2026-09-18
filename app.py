import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "University Academic Knowledge Assistant"
DATABASE_DIR = Path("rag_database")

GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_TOP_K = 5
MIN_SIMILARITY = 0.20


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-header {
        padding: 0.5rem 0 1rem 0;
    }

    .main-header h1 {
        margin-bottom: 0.15rem;
        font-size: 2.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
    }

    .source-card {
        padding: 0.9rem 1rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        margin: 0.5rem 0;
        background: rgba(128, 128, 128, 0.05);
    }

    .metric-card {
        padding: 0.8rem;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 12px;
        text-align: center;
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE VALIDATION
# ============================================================

INDEX_FILE = DATABASE_DIR / "index.faiss"
CHUNKS_FILE = DATABASE_DIR / "chunks.json"
CONFIG_FILE = DATABASE_DIR / "embedding_config.json"
METADATA_FILE = DATABASE_DIR / "metadata.json"


def validate_database():
    required_files = [
        INDEX_FILE,
        CHUNKS_FILE,
        CONFIG_FILE,
        METADATA_FILE,
    ]

    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "The RAG database is incomplete. Missing files:\n"
            + "\n".join(missing)
        )


# ============================================================
# LOAD RAG DATABASE
# ============================================================

@st.cache_resource(show_spinner="Loading academic knowledge base...")
def load_database():
    validate_database()

    index = faiss.read_index(str(INDEX_FILE))

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        embedding_config = json.load(file)

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        database_metadata = json.load(file)

    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index/vector count does not match chunks.json. "
            "Rebuild the RAG database."
        )

    model_name = embedding_config.get(
        "embedding_model",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    model = SentenceTransformer(model_name)

    return (
        index,
        chunks,
        embedding_config,
        database_metadata,
        model,
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    question,
    index,
    chunks,
    embedding_model,
    top_k=DEFAULT_TOP_K,
):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal),
    )

    results = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue

        item = chunks[int(index_position)]

        results.append(
            {
                "score": float(score),
                "text": item["text"],
                "metadata": item["metadata"],
            }
        )

    return results


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(results):
    context_parts = []

    for number, result in enumerate(results, start=1):
        metadata = result["metadata"]

        source_file = metadata.get(
            "source_file",
            "Unknown document",
        )

        page_number = metadata.get(
            "page_number",
            "Unknown",
        )

        context_parts.append(
            f"""SOURCE [{number}]
Document: {source_file}
Page: {page_number}
Similarity: {result["score"]:.4f}

Content:
{result["text"]}
"""
        )

    return "\n\n".join(context_parts)


# ============================================================
# GEMINI RAG GENERATION
# ============================================================

def generate_answer(
    question,
    context,
    client,
):
    system_instruction = """
You are a University Student and Academic Knowledge Assistant.

Your job is to answer questions using ONLY the supplied retrieved
academic document context.

Rules:
1. Do not invent facts that are not supported by the context.
2. Do not use outside knowledge to fill missing information.
3. If the context does not contain enough information, say:
   "I could not find this information in the provided academic documents."
4. Give a clear, useful answer suitable for a university student.
5. When making a factual claim, cite the relevant source using
   [1], [2], etc., matching the SOURCE numbers in the context.
6. Do not create fake page numbers or fake document names.
7. If multiple sources support the answer, cite all relevant sources.
8. Keep the answer concise but sufficiently explanatory.
"""

    prompt = f"""
Student question:
{question}

Retrieved academic context:
{context}

Answer the student's question using only the retrieved context.
Include source citations such as [1] or [2] where appropriate.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=1200,
        ),
    )

    answer = response.text

    if not answer:
        return (
            "I could not generate an answer from the provided "
            "academic documents."
        )

    return answer


# ============================================================
# SOURCE DISPLAY
# ============================================================

def display_sources(results):
    st.markdown("### 📚 Sources")

    seen = set()

    for number, result in enumerate(results, start=1):
        metadata = result["metadata"]

        source_file = metadata.get(
            "source_file",
            "Unknown document",
        )

        page_number = metadata.get(
            "page_number",
            "Unknown",
        )

        key = (source_file, page_number)

        if key in seen:
            continue

        seen.add(key)

        st.markdown(
            f"""
            <div class="source-card">
                <strong>[{number}] 📄 {source_file}</strong><br>
                <span class="small-muted">
                    Page {page_number}
                    &nbsp;•&nbsp;
                    Similarity {result["score"]:.3f}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# LOAD APPLICATION RESOURCES
# ============================================================

try:
    (
        faiss_index,
        chunks,
        embedding_config,
        database_metadata,
        embedding_model,
    ) = load_database()

except Exception as error:
    st.error("Unable to load the RAG database.")
    st.code(str(error))
    st.info(
        "Make sure the complete rag_database folder is inside "
        "the GitHub repository."
    )
    st.stop()


# ============================================================
# GEMINI API KEY
# ============================================================

try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error(
        "GEMINI_API_KEY is not configured."
    )
    st.markdown(
        """
        Add the key to Streamlit Cloud Secrets:

        ```toml
        GEMINI_API_KEY = "your-gemini-api-key"
        ```
        """
    )
    st.stop()


gemini_client = get_gemini_client(gemini_api_key)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 Academic Assistant")

    st.caption(
        "RAG-powered university knowledge assistant"
    )

    st.divider()

    top_k = st.slider(
        "Retrieved sources",
        min_value=3,
        max_value=8,
        value=DEFAULT_TOP_K,
        help="Number of document chunks retrieved from FAISS.",
    )

    st.divider()

    st.markdown("### Knowledge Base")

    total_documents = database_metadata.get(
        "documents",
        database_metadata.get("total_documents", "—"),
    )

    total_chunks = database_metadata.get(
        "chunks",
        database_metadata.get("total_chunks", len(chunks)),
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Documents", total_documents)

    with col2:
        st.metric("Chunks", total_chunks)

    st.divider()

    st.markdown("### AI Model")

    st.caption(f"Gemini: `{GEMINI_MODEL}`")

    st.caption(
        f"Embeddings: `{embedding_config.get('embedding_model', 'Unknown')}`"
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🎓 University Academic Knowledge Assistant</h1>
        <div class="subtitle">
            Ask questions about the university academic documents
            and get grounded answers with page-level sources.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Answers are generated from the provided academic document "
    "knowledge base. Sources and page numbers are shown below."
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            display_sources(message["sources"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your academic documents..."
)


if question:

    question = question.strip()

    if not question:
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching the academic knowledge base..."):

            retrieved = retrieve_documents(
                question=question,
                index=faiss_index,
                chunks=chunks,
                embedding_model=embedding_model,
                top_k=top_k,
            )

        if not retrieved:

            answer = (
                "I could not find this information in the "
                "provided academic documents."
            )

            st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": [],
                }
            )

        elif retrieved[0]["score"] < MIN_SIMILARITY:

            answer = (
                "I could not find this information in the "
                "provided academic documents."
            )

            st.markdown(answer)

            display_sources(retrieved[:3])

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved[:3],
                }
            )

        else:

            context = build_context(retrieved)

            with st.spinner("Generating grounded answer..."):

                try:
                    answer = generate_answer(
                        question=question,
                        context=context,
                        client=gemini_client,
                    )

                except Exception as error:

                    answer = (
                        "I encountered an error while generating "
                        "the answer. Please try again."
                    )

                    st.error(str(error))

            st.markdown(answer)

            display_sources(retrieved)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved,
                }
            )
