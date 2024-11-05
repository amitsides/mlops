# 1. prepare a small corpus to search
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
query = "Where was Turing born?"
bm25_ef = BM25EmbeddingFunction()

# 2. fit the corpus to get BM25 model parameters on your documents.
bm25_ef.fit(docs)

# 3. store the fitted parameters to disk to expedite future processing.
bm25_ef.save("bm25_params.json")

# 4. load the saved params
new_bm25_ef = BM25EmbeddingFunction()
new_bm25_ef.load("bm25_params.json")

docs_embeddings = new_bm25_ef.encode_documents(docs)
query_embeddings = new_bm25_ef.encode_queries([query])
print("Dim:", new_bm25_ef.dim, list(docs_embeddings)[0].shape)