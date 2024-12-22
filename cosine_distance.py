import requests 
import json
import pandas as pd 
from dateutil.parser import parse
#from openai.embeddings_utils import  distances_from_embeddings
from scipy.spatial.distance import cosine 

import openai
openai.api_base = "https://api.openai.com/v1/"
openai.api_key =  "OPEN_API_KEY"


df = pd.read_csv("text_embeddings.csv", index_col=0)


question = "When did Russia invade Ukraine ?"

EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'
#question_embeddings = get_embedding(question, engine=EMBEDDING_MODEL_NAME)

response = openai.embeddings.create(input=[question],model=EMBEDDING_MODEL_NAME)

q_embedding = (response.data[0].embedding)
print(df['embeddings'])

cos_d = []
for embedding  in df['embeddings']:
	print("length embedding = {}  , length q_embedding = {}".format(len(list(embedding)),len(q_embedding)))
	cos_d.append(1- cosine(q_embedding,embedding)) 

df['cos_d'] = cos_d

print(df.head)
