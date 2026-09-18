# 🎓 University Student & Academic Knowledge Assistant

A RAG-based university academic knowledge assistant built with:

- **Streamlit** — web application
- **FAISS** — local vector database
- **Sentence Transformers** — document/query embeddings
- **Google Gemini 3.6 Flash** — answer generation
- **JSON metadata** — source and page traceability

The application is designed around a precomputed RAG database. The six academic PDFs are processed once in Google Colab. The resulting FAISS index and metadata are stored in the repository, so the application does **not** recreate document embeddings every time it starts.

## Architecture

```text
Student Question
       │
       ▼
Sentence Transformer
       │
       ▼
Query Embedding
       │
       ▼
FAISS Similarity Search
       │
       ▼
Relevant Chunks
       │
       ├── source filename
       ├── page number
       ├── chunk number
       └── similarity score
       │
       ▼
Retrieved Context
       │
       ▼
Gemini 3.6 Flash
       │
       ▼
Grounded Answer
       │
       ▼
Sources + Page Numbers
```

## Project Structure

```text
university-academic-assistant/
│
├── app.py
├── requirements.txt
├── README.md
│
└── rag_database/
    ├── index.faiss
    ├── chunks.json
    ├── embedding_config.json
    ├── document_summary.json
    ├── metadata.json
    └── README.md
```

## Important

The `rag_database` folder must be copied from the Google Colab preprocessing step into the root of this GitHub repository.

Do not rebuild the six document embeddings inside `app.py`.

Only the student's query is embedded at runtime.

## Embedding Model

The preprocessing pipeline uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The application reads the embedding model name from:

```text
rag_database/embedding_config.json
```

This helps keep query embeddings compatible with the stored FAISS vectors.

## Gemini Model

This application uses:

```text
gemini-3.6-flash
```

The model is configured in `app.py`.

Google's current Gemini documentation lists `gemini-3.6-flash` as a stable model and documents use with the Google GenAI Python SDK. The application uses the modern `google-genai` package and `client.models.generate_content(...)`.

## API Key

Never put your Gemini API key directly into `app.py`.

For local Streamlit development, create:

```text
.streamlit/secrets.toml
```

with:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Do not upload `.streamlit/secrets.toml` to GitHub.

Add this to `.gitignore`:

```text
.streamlit/secrets.toml
__pycache__/
*.pyc
.env
```

## Run Locally

Create a Python 3.10+ environment.

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

### 1. Push the repository to GitHub

The repository should contain:

```text
app.py
requirements.txt
README.md
rag_database/
```

### 2. Open Streamlit Community Cloud

Create a new application and select your GitHub repository.

Set the main file to:

```text
app.py
```

### 3. Add Gemini Secret

In the application's Streamlit Cloud settings, open **Secrets** and add:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Save the secret and redeploy/restart the application.

## RAG Database

The database contains:

### `index.faiss`

The FAISS vector index containing the precomputed document embeddings.

### `chunks.json`

The text chunks plus metadata.

Example:

```json
{
  "text": "Academic policy information...",
  "metadata": {
    "chunk_id": 15,
    "document_number": 2,
    "source_file": "Academic_Policy.pdf",
    "page_number": 4,
    "chunk_number": 2,
    "document_type": "PDF"
  }
}
```

### `embedding_config.json`

Contains the embedding model, dimension, chunk configuration, and FAISS configuration.

### `document_summary.json`

Contains processing information for each academic PDF.

### `metadata.json`

Contains overall RAG database information.

## Source Traceability

The assistant does not only return an answer.

It also displays the retrieved:

```text
📄 Source filename
📃 Page number
🔎 Similarity score
```

Gemini is instructed to cite retrieved context using:

```text
[1]
[2]
[3]
```

The source cards underneath the answer provide the corresponding document and page information.

## Grounding Behavior

The assistant is instructed to:

1. Use only retrieved academic context.
2. Avoid inventing information.
3. Avoid using outside knowledge to fill missing information.
4. State when information cannot be found in the provided documents.
5. Cite supporting retrieved sources.

If the highest retrieval similarity is below the configured minimum threshold, the application returns:

> I could not find this information in the provided academic documents.

## Performance

The application uses Streamlit resource caching for:

- FAISS index
- metadata
- chunks
- Sentence Transformer model
- Gemini client

Therefore these resources are not repeatedly initialized during normal Streamlit reruns.

## Updating the Documents

If any academic PDF changes:

1. Update the Google Drive folder.
2. Run the Colab preprocessing pipeline again.
3. Replace the `rag_database` folder in GitHub.
4. Commit and push.
5. Streamlit Cloud will redeploy.

Do not manually modify `index.faiss` or `chunks.json`.

## Security

Never commit:

```text
GEMINI_API_KEY
```

to GitHub.

Use Streamlit Secrets instead.

## License

This project is intended as an academic/software development project. Check the licensing and usage permissions of the university documents before publicly redistributing their contents.
