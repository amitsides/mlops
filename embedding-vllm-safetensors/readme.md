Loading SafeTensors:

The EmbeddingHandler class takes a list of paths to safetensors files
The load_embeddings() method loads all models into memory


Running on vLLM:

The run_on_vllm() method processes the embeddings through vLLM
You can specify the base model and maximum tokens
It handles both single strings and lists of input text


Merging Embeddings:

The merge_embeddings() method combines all loaded tensors
Tensors are concatenated along the last dimension
The merged result is saved as a new safetensors file