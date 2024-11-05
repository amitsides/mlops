from pymilvus import model

# This will download "all-MiniLM-L6-v2", a light weight model.
ef = model.DefaultEmbeddingFunction()

# Data from which embeddings are to be generated 
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

embeddings = ef.encode_documents(docs)

# Print embeddings
print("Embeddings:", embeddings)
# Print dimension and shape of embeddings
print("Dim:", ef.dim, embeddings[0].shape)