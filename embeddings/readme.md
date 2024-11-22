# Embedding 

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
``

### GCP Embedding
https://cloud.google.com/vertex-ai/generative-ai/docs/use-embedding-models


### NVIDIA NIM with 

https://huggingface.co/Snowflake/snowflake-arctic-embed-l
https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/helm-charts/text-embedding-nim
```yaml

helm upgrade --install \
  --namespace nrem \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set image.repository=nvcr.io/nim/snowflake/arctic-embed-l \
  --set image.tag=1.0.1 \
  nemo-embedder \
  https://helm.ngc.nvidia.com/nim/nvidia/charts/text-embedding-nim-1.1.0.tgz```
  ```

Useful Links:
https://www.youtube.com/watch?v=E4rNTYN3aIg
https://milvus.io/docs/embeddings.md

### More
 MTEB https://github.com/embeddings-benchmark/mteb/tree/main
- generative-openai
- qna-openai
- ref2vec-centroid
- text2vec-cohere
- text2vec-huggingface / 
- text2vec-openai

###  Store Embeddings on MongoDB Atlas or PineCone or Milvus
        Args:
            data_path: Path to data directory
            embedding_config: Configuration for embedding generation
            vector_db: Type of vector database ("pinecone" or "mongodb")
            db_config: Database configuration parameters


