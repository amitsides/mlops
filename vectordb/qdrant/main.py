from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from qdrant_client import QdrantClient
import uvicorn

embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

qdrant_client = QdrantClient(
        url=os.environ['QDRANT_URL'], 
        api_key=os.environ['QDRANT_API_KEY'],
    )

app = FastAPI()

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')