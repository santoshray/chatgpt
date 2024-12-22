import requests 
import json
import pandas as pd 
from dateutil.parser import parse
from scipy.spatial.distance import cosine 

import openai
openai.api_base = "https://api.openai.com/v1/"
openai.api_key = "OPEN_API_KEY"



params = {
    "action": "query", 
    "prop": "extracts",
    "exlimit": 1,
    "titles": "2022",
    "explaintext": 1,
    "formatversion": 2,
    "format": "json"
}
resp = requests.get("https://en.wikipedia.org/w/api.php", params=params)
response_dict = resp.json()
data_list = response_dict["query"]["pages"][0]["extract"].split("\n")


#response = requests.get('https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exlimit=1&titles=2022&explaintext=1&formatversion=2&format=json')
#print(response)
#data_list = response.json()['query']['pages'][0]['extract'].split('\n')
#print(data_list[:5])



df =pd.DataFrame()

df['text'] = data_list
df = pd.DataFrame()
df["text"] = response_dict["query"]["pages"][0]["extract"].split("\n")

df = df[(df["text"].str.len() > 0) & (~df["text"].str.startswith("=="))]

# In some cases dates are used as headings instead of being part of the
# text sample; adjust so dated text samples start with dates
prefix = ""
for (i, row) in df.iterrows():
    # If the row already has " - ", it already has the needed date prefix
    if " – " not in row["text"]:
        try:
            # If the row's text is a date, set it as the new prefix
            parse(row["text"])
            prefix = row["text"]
        except:
            # If the row's text isn't a date, add the prefix
            row["text"] = prefix + " – " + row["text"]
df = df[df["text"].str.contains(" – ")].reset_index(drop=True)

#print(df['text'].tolist())



#ukraine_prompt = """
#Question: When did Russia invade Ukraine?
#Answer: 
#"""

#ukraine_answer = openai.completions.create(
#    model="gpt-3.5-turbo-instruct",
#    prompt=ukraine_prompt
#)

#print(ukraine_answer.choices[0].text)
question = "When did Russia invade Ukraine ?"
#print(ukraine_answer)
embeddings_list = [] 
EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'
response = openai.embeddings.create(input=df['text'].tolist(),model=EMBEDDING_MODEL_NAME)

#print(response)
for resp_data in response.data:
	embeddings_list.append(resp_data.embedding)

df['embeddings'] = embeddings_list

#print(df.head)
#df.to_csv("text_embeddings.csv")


response = openai.embeddings.create(input=[question],model=EMBEDDING_MODEL_NAME)

q_embedding = (response.data[0].embedding)
#print(df['embeddings'])

cos_d = []
for embedding  in embeddings_list:
	#print("length embedding = {}  , length q_embedding = {}".format(len(list(embedding)),len(q_embedding)))
	cos_d.append(1- cosine(q_embedding,embedding)) 

df['cos_d'] = cos_d

df_copy = df.copy()
df_copy.sort_values("cos_d", ascending=False, inplace=True)

print(df_copy.head)

df_copy.to_csv('sorted_context.csv')

cos_d.sort(reverse=True)

print(cos_d[0:10])

#print(df.head)

#ukraine_prompt = 
#Question: "When did Russia invade Ukraine?"
#Answer:

#initial_ukraine_answer = openai.completions.create(
#    model="gpt-3.5-turbo-instruct",
#    prompt=ukraine_prompt,
#    max_tokens=150
#)["choices"][0]["text"].strip()
#print(initial_ukraine_answer)


