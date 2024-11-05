# Embedding
Useful Links:
https://www.youtube.com/watch?v=E4rNTYN3aIg
https://milvus.io/docs/embeddings.md

- generative-openai
- qna-openai
- ref2vec-centroid
- text2vec-cohere
- text2vec-huggingface / MTEB https://github.com/embeddings-benchmark/mteb/tree/main
- text2vec-openai

###  File2Vec - Store Embeddings on MongoDB Atlas or PineCone or Milvus
        Args:
            data_path: Path to data directory
            embedding_config: Configuration for embedding generation
            vector_db: Type of vector database ("pinecone" or "mongodb")
            db_config: Database configuration parameters


### AWS Bedrock embedding
https://github.com/aws-samples/amazon-bedrock-samples/blob/main/embeddings/Titan-V2-Embeddings.ipynb
```python
sample_model_input={
    "inputText": prompt_data,
    "dimensions": 256,
    "normalize": True
}

body = json.dumps(sample_model_input)

response = boto3_bedrock_runtime.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
embedding= bedrock_embeddings(text=prompt_data, dimensions=256, normalize=True)
print(f"The embedding vector has {len(embedding)} values\n{embedding[0:3]+['...']+embedding[-3:]}")
```
``#
pip install apache-airflow pandas scikit-learn pymilvus``
