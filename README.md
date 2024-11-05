<img src="./logo.png" alt="Appwrite Logo">

### MLPlay Philosopy: Machine Learning Operations Playground
#### MLOPS/DataOps/Infrastructure Playground
MLPlay is a <b>Playground</b> for MLOPS/DataOps/ML Data Science <b>Orchestration</b>,
it suggests some pipelines & workflows for common ML ELT/ETL routines with <b>DRY</b> design in mind.
The aim is to start with raw data and, while using various models and Deep-Neural-Networks, reaching Embeddings VectorStore in VectorDBs to allow Inference with Kubernetes as Infrastructure.  

Supported Clouds GCP & AWS 

## Project Structure
- ./embedding containts various scripts to generate Embeddings/Vectors for various data types
- ./vectordb (provides clients and example how to connect to various vector-db like Pinecone/Weaviate/Milvus/mongoDb-atlas)
- ./chart helm chart for deploying ml models to k8s (including secret management)
- ./DAG (Workflow/Pipeline/DAG/Airflow/ELT)  
  1. Extract/Load relevant data from databases, datalake or storage for processing.
  2. Transform/Embedding: Transforms into High-Dimensional vector embedding using various ebmedding mechanisms for NLP/Computer Vision.
  3. VectorDb & VectorStore Library and Client  Loading into multiple VectorDbs such as Pinecone (see `./vectordb/README.md`)
  4. Inference using Kubernetes with GPUs


#### Suggest Integrations
- Databricks (for data processing and model training at scale)
- Weight & Biases (for experiment tracking and model monitoring) 
- Milvus (for scalable vector similarity search)
- Jupyter Notebook (for interactive data analysis and prototyping)
- Docker (for containerizing applications and models)
- MLflow (for managing the machine learning lifecycle)
- Airflow (for orchestrating ML workflows)
- Prometheus + Grafana (for monitoring model performance)

