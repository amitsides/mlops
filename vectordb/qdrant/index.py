from pydantic import BaseModel
from qdrant_client.http import models

class IndexQuery(BaseModel):
    collection: str
    documents: list

@app.post("/index")
async def index_documents(query: IndexQuery):
    collection = query.collection
    documents = query.documents

    qdrant_client.recreate_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
    )

    for document in documents:
        fulltext = ""
        response = requests.get(document)
        document_data = BytesIO(response.content)
        with pdfplumber.open(document_data) as pdf:
            for page in pdf.pages:
                fulltext += page.extract_text()

        text = fulltext

        chunks = []
        while len(text) > 500:
            last_period_index = text[:500].rfind('.')
            if last_period_index == -1:
                last_period_index = 500
            chunks.append(text[:last_period_index])
            text = text[last_period_index+1:]
        chunks.append(text)

        points = []
        i = 1
        for chunk in chunks:
            i += 1

            embeddings = embedding_model.encode(chunk, convert_to_numpy=True)

            points.append(models.PointStruct(id=i, vector=embeddings.tolist(), payload={"text": chunk}))
        
        qdrant_client.upsert(
            collection_name=collection,
            points=points
        )