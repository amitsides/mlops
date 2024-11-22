from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
import numpy as np


def faiss_to_neo4j(faiss_index, neo4j_driver, node_label, embedding_property):
    # Connect to Neo4j
    with neo4j_driver.session() as session:
        # Iterate through FAISS index
        for i in range(faiss_index.index.ntotal):
            vector = faiss_index.index.reconstruct(i)
            text = faiss_index.docstore.get(i)

            # Create Neo4j node with embedding
            session.run(
                f"CREATE (n:{node_label} {{text: $text, {embedding_property}: $vector}})",
                text=text,
                vector=vector.tolist()
            )


def neo4j_to_faiss(neo4j_driver, node_label, embedding_property, dimension):
    texts = []
    embeddings = []

    # Connect to Neo4j
    with neo4j_driver.session() as session:
        # Retrieve nodes with embeddings
        result = session.run(f"MATCH (n:{node_label}) RETURN n.text AS text, n.{embedding_property} AS embedding")

        for record in result:
            texts.append(record["text"])
            embeddings.append(record["embedding"])

    # Create FAISS index
    embeddings = np.array(embeddings).astype('float32')
    faiss_index = FAISS.from_embeddings(embeddings, texts)

    return faiss_index


# Example usage
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "password"
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Convert FAISS to Neo4j
faiss_index = FAISS.load_local("path/to/faiss/index")
faiss_to_neo4j(faiss_index, neo4j_driver, "Document", "embedding")

# Convert Neo4j to FAISS
dimension = 768  # Adjust based on your embedding dimension
new_faiss_index = neo4j_to_faiss(neo4j_driver, "Document", "embedding", dimension)