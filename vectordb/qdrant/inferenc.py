import requests

class InferenceQuery(BaseModel):
    prompt: str
    collection: str

@app.post("/generate")
async def predict(query: InferenceQuery):
    prompt = query.prompt
    collection = query.collection

    embeddings = embedding_model.encode(prompt, convert_to_numpy=True)

    search_result = qdrant_client.search(
        collection_name=collection,
        query_vector=embeddings.tolist(),
        limit=5
    )

    context = "CONTEXT:\n"
    for result in search_result:
        context += result.payload['text'] + "\n\n"

    system_prompt = f"""Your goal is to answer questions based on the given context. 
        Make sure the answer is only from information in the context. 
        If the information is not in the context, reply with: "I do not have enough information to answer this question.". 
            Context: ```{context}```
            Question: ***{prompt}***
            Answer:"""

    response = requests.post(os.environ['LLM_URL'], json={"text": system_prompt}, headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"})

    return response.json()