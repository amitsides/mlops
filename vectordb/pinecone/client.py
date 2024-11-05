import config
from pinecone import Pinecone
from textwrap import wrap

def initialize_pinecone_client():
    pinecone = Pinecone(
        api_key=config.PINECONE_API_KEY, 
        source_tag="box_pinecone_integration"
    )
    return pinecone

import config
from pinecone import Pinecone
from textwrap import wrap

def chunk_text(text, chunk_size=4000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap adjustment
    return chunks

def store_metadata_in_pinecone(client, file_id, combined_text):
    index_name = config.PINECONE_INDEX
    
    # Ensure combined_text["text"] is not None
    text_content = combined_text.get("text")
    if text_content is None:
        print(f"No text representation available for file ID: {file_id}")
        return

    # Chunk the text with overlap
    text_chunks = chunk_text(text_content, chunk_size=2000, overlap=100)
    
    index = client.Index(index_name)
    
    for i, chunk in enumerate(text_chunks):
        # Get embeddings using the Pinecone inference API
        embeddings = client.inference.embed(
            model="multilingual-e5-large",
            inputs=[chunk],
            parameters={
                "input_type": "passage",
                "truncate": "END"
            }
        )

        # Check if embeddings were returned
        if not embeddings or len(embeddings) == 0:
            print(f"No embeddings returned for chunk {i} of file ID: {file_id}")
            continue

        # Extract the actual embedding values (list of floats) from the result
        vectors = embeddings[0]['values']
        
        # Store the chunk text in metadata
        minimal_metadata = {
            "chunk_id": i, 
            "file_id": file_id, 
            "file_name": combined_text.get("file_name"), 
            "created_at": combined_text.get("created_at"), 
            "modified_at": combined_text.get("modified_at"), 
            "size": combined_text.get("size", 0),
            "chunk_text": chunk  # Store the chunk text as part of the metadata
        }
        
        # Upsert the vector with the chunk ID and metadata into Pinecone
        index.upsert([(f"{file_id}_chunk_{i}", vectors, minimal_metadata)])