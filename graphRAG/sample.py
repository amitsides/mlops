import neo4j
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

llm=OpenAILLM(
   model_name="gpt-4o-mini",
   model_params={
       "response_format": {"type": "json_object"}, # use json_object formatting for best results
       "temperature": 0 # turning temperature down for more deterministic results
   }
)

#create text embedder
embedder = OpenAIEmbeddings()