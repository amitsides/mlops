handler = EmbeddingHandler([
    "path/to/first_embedding.safetensors",
    "path/to/second_embedding.safetensors"
])
handler.load_embeddings()
processed = handler.run_on_vllm("Your text here")
handler.merge_embeddings("merged_output.safetensors")