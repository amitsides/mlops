# multi-llm

This directory aims to merge multipl LLMs in a multiple ways,

```python
from vertexai.matching_engine import MatchingEngineIndex

def generate_multi_model_embeddings(text):
    embeddings = {
        'openai': openai_model.embed(text),
        'cohere': cohere_model.embed(text),
        'anthropic': anthropic_model.embed(text),
        'huggingface': hf_model.embed(text),
        'google': google_model.embed(text)
    }
    return embeddings

def create_unified_vector_index(embeddings_dataset):
    # Combine and normalize embeddings
    unified_embeddings = process_embeddings(embeddings_dataset)
    
    # Create Vertex AI Vector Search index
    index = MatchingEngineIndex.create(
        embeddings=unified_embeddings,
        dimensions=unified_embedding_dimension
    )
    return index

    