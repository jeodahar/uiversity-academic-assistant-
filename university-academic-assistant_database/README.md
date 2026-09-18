# RAG Database

This directory contains the precomputed vector database for the
University Academic Knowledge Assistant.

## Files

- `index.faiss` - FAISS vector index
- `chunks.json` - Text chunks and source metadata
- `embedding_config.json` - Embedding configuration
- `document_summary.json` - Document processing summary
- `metadata.json` - Database metadata

## Embedding Model

sentence-transformers/all-MiniLM-L6-v2

## Vector Dimension

384

## Similarity

Cosine similarity using normalized embeddings and FAISS Inner Product.

## Chunking

Chunk size: 700 characters

Chunk overlap: 120 characters

## Documents

6

## Total Vectors

1056

Source metadata includes:

- source filename
- page number
- chunk number
- chunk ID
- document type
- chunk text length
