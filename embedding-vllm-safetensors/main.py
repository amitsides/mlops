import os
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from vllm import LLM, SamplingParams
from typing import List, Dict, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingHandler:
    def __init__(self, model_paths: List[str]):
        """
        Initialize the handler with paths to safetensors embedding models.
        
        Args:
            model_paths: List of paths to .safetensors files
        """
        self.model_paths = model_paths
        self.loaded_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        self.merged_tensor = None
        
    def load_embeddings(self) -> None:
        """Load all specified safetensors embedding models."""
        for path in self.model_paths:
            try:
                logger.info(f"Loading embedding model from {path}")
                with safe_open(path, framework="pt", device="cpu") as f:
                    tensors = {key: f.get_tensor(key) for key in f.keys()}
                    self.loaded_tensors[path] = tensors
            except Exception as e:
                logger.error(f"Error loading model from {path}: {str(e)}")
                raise

    def run_on_vllm(self, input_text: Union[str, List[str]], 
                    model_name: str = "bert-base-uncased",
                    max_tokens: int = 512) -> Dict[str, torch.Tensor]:
        """
        Run the loaded embeddings through vLLM.
        
        Args:
            input_text: Text or list of texts to embed
            model_name: Name of the base model to use
            max_tokens: Maximum number of tokens to process
            
        Returns:
            Dictionary of embeddings for each input
        """
        try:
            logger.info(f"Initializing vLLM with model {model_name}")
            llm = LLM(model=model_name)
            
            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic output
                max_tokens=max_tokens
            )
            
            if isinstance(input_text, str):
                input_text = [input_text]
                
            outputs = llm.generate(input_text, sampling_params)
            
            # Process outputs and combine with loaded embeddings
            results = {}
            for output, tensors in zip(outputs, self.loaded_tensors.values()):
                embedding = output.outputs[0].embedding  # Assuming vLLM returns embeddings
                for key, tensor in tensors.items():
                    # Combine original embedding with vLLM output
                    results[key] = torch.cat([tensor, embedding], dim=-1)
                    
            return results
            
        except Exception as e:
            logger.error(f"Error running vLLM inference: {str(e)}")
            raise

    def merge_embeddings(self, output_path: str) -> None:
        """
        Merge all loaded embedding tensors into one and save as safetensors.
        
        Args:
            output_path: Path to save the merged safetensors file
        """
        try:
            logger.info("Merging embeddings")
            merged = {}
            
            # Combine all tensors
            for model_tensors in self.loaded_tensors.values():
                for key, tensor in model_tensors.items():
                    if key not in merged:
                        merged[key] = tensor
                    else:
                        # Concatenate along the last dimension
                        merged[key] = torch.cat([merged[key], tensor], dim=-1)
            
            self.merged_tensor = merged
            
            # Save merged tensor
            logger.info(f"Saving merged embeddings to {output_path}")
            save_file(merged, output_path)
            
        except Exception as e:
            logger.error(f"Error merging embeddings: {str(e)}")
            raise

def main():
    # Example usage
    model_paths = [
        "path/to/embedding1.safetensors",
        "path/to/embedding2.safetensors"
    ]
    
    handler = EmbeddingHandler(model_paths)
    
    # Load embeddings
    handler.load_embeddings()
    
    # Run through vLLM
    test_input = ["This is a test sentence.", "Another test sentence."]
    processed_embeddings = handler.run_on_vllm(test_input)
    
    # Merge and save
    handler.merge_embeddings("path/to/merged_embeddings.safetensors")

if __name__ == "__main__":
    main()