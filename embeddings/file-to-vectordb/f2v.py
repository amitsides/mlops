import os
from typing import Optional, List, Union, Dict, Any
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass

# Vector DB imports
import pinecone
from pymongo import MongoClient
from pymongo.errors import ConnectionError

# ML related imports
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd

@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 512
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class DataEmbeddingPipeline:
    """
    A pipeline for loading data, generating embeddings, and storing in vector databases.
    """
    def __init__(
        self,
        data_path: str = "/usr/data",
        embedding_config: Optional[EmbeddingConfig] = None,
        vector_db: str = "pinecone",
        db_config: Dict[str, Any] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            data_path: Path to data directory
            embedding_config: Configuration for embedding generation
            vector_db: Type of vector database ("pinecone" or "mongodb")
            db_config: Database configuration parameters
        """
        self.data_path = Path(data_path)
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.vector_db = vector_db.lower()
        self.db_config = db_config or {}
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Initialize vector database connection
        self._init_vector_db()

    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def _init_embedding_model(self):
        """Initialize the embedding model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_config.model_name
            )
            self.model = AutoModel.from_pretrained(
                self.embedding_config.model_name
            ).to(self.embedding_config.device)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise

    def _init_vector_db(self):
        """Initialize vector database connection."""
        try:
            if self.vector_db == "pinecone":
                pinecone.init(
                    api_key=self.db_config.get("api_key"),
                    environment=self.db_config.get("environment")
                )
                self.index = pinecone.Index(self.db_config.get("index_name"))
            
            elif self.vector_db == "mongodb":
                self.client = MongoClient(self.db_config.get("connection_string"))
                self.db = self.client[self.db_config.get("database")]
                self.collection = self.db[self.db_config.get("collection")]
            
            else:
                raise ValueError(f"Unsupported vector database: {self.vector_db}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {str(e)}")
            raise

    @torch.no_grad()
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for input texts."""
        all_embeddings = []
        
        for i in range(0, len(texts), self.embedding_config.batch_size):
            batch_texts = texts[i:i + self.embedding_config.batch_size]
            
            # Tokenize and prepare input
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.embedding_config.max_length,
                return_tensors="pt"
            ).to(self.embedding_config.device)
            
            # Generate embeddings
            outputs = self.model(**encoded)
            
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)

    def load_data(self) -> List[Dict[str, Any]]:
        """Load data from the specified directory."""
        all_data = []
        
        try:
            for file_path in self.data_path.glob("**/*"):
                if file_path.is_file():
                    if file_path.suffix == ".csv":
                        df = pd.read_csv(file_path)
                    elif file_path.suffix == ".json":
                        df = pd.read_json(file_path)
                    elif file_path.suffix in [".txt", ".md"]:
                        with open(file_path, "r") as f:
                            text = f.read()
                            df = pd.DataFrame({"text": [text]})
                    else:
                        continue
                    
                    # Convert DataFrame to list of dictionaries
                    records = df.to_dict("records")
                    for record in records:
                        record["source_file"] = str(file_path)
                        record["timestamp"] = datetime.now().isoformat()
                    
                    all_data.extend(records)
                    
            self.logger.info(f"Loaded {len(all_data)} records from {self.data_path}")
            return all_data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def store_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """Store vectors and metadata in the vector database."""
        try:
            if self.vector_db == "pinecone":
                # Prepare data for Pinecone
                vectors_with_ids = [
                    (str(i), vec.tolist(), meta)
                    for i, (vec, meta) in enumerate(zip(vectors, metadata))
                ]
                
                # Upsert to Pinecone in batches
                batch_size = 100
                for i in range(0, len(vectors_with_ids), batch_size):
                    batch = vectors_with_ids[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                
            elif self.vector_db == "mongodb":
                # Prepare data for MongoDB
                documents = [
                    {
                        "vector": vec.tolist(),
                        "metadata": meta,
                        "timestamp": datetime.now()
                    }
                    for vec, meta in zip(vectors, metadata)
                ]
                
                # Insert into MongoDB
                self.collection.insert_many(documents)
                
            self.logger.info(f"Successfully stored {len(vectors)} vectors in {self.vector_db}")
            
        except Exception as e:
            self.logger.error(f"Error storing vectors: {str(e)}")
            raise

    def run_pipeline(self):
        """Run the complete pipeline."""
        try:
            # Load data
            self.logger.info("Starting data loading...")
            data = self.load_data()
            
            # Extract texts for embedding
            texts = [record.get("text", "") for record in data]
            
            # Generate embeddings
            self.logger.info("Generating embeddings...")
            embeddings = self.generate_embeddings(texts)
            
            # Store in vector database
            self.logger.info("Storing vectors...")
            self.store_vectors(embeddings, data)
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    # Example configuration
    db_config = {
        "pinecone": {
            "api_key": "your-pinecone-api-key",
            "environment": "your-environment",
            "index_name": "your-index-name"
        },
        "mongodb": {
            "connection_string": "your-mongodb-connection-string",
            "database": "your-database-name",
            "collection": "your-collection-name"
        }
    }

    # Initialize and run pipeline
    pipeline = DataEmbeddingPipeline(
        data_path="/usr/data",
        vector_db="pinecone",  # or "mongodb"
        db_config=db_config["pinecone"]  # or db_config["mongodb"]
    )
    
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()