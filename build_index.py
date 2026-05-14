import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rag_engine import chunk_by_article_and_table, sub_chunk_large_articles, normalize_arabic_text

with open("arabic_text_and_tables.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

chunks = chunk_by_article_and_table(raw_text, normalize_numerals=True)
for c in chunks:
    c.content = normalize_arabic_text(c.content)
chunks = sub_chunk_large_articles(chunks, 5000)

print(f"Total chunks: {len(chunks)}")

embedder = SentenceTransformer("intfloat/multilingual-e5-large")
texts = [f"passage: {c.content}" for c in chunks]
embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(np.array(embeddings))

faiss.write_index(index, "faiss_index.bin")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Index saved!")