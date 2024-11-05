from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, AnnSearchRequest, RRFRanker, connections,
)

# 1. prepare a small corpus to search
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
query = "Who started AI research?"

# BGE-M3 model can embed texts as dense and sparse vectors.
# It is included in the optional `model` module in pymilvus, to install it,
# simply run "pip install pymilvus[model]".

bge_m3_ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")

docs_embeddings = bge_m3_ef(docs)
query_embeddings = bge_m3_ef([query])