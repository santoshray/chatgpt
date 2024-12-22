import requests 
import json
import pandas as pd 
import tiktoken
from openai import OpenAI
import openai

openai.base_url = "https://api.openai.com/v1/"
client = OpenAI(
  api_key="OPEN_API_KEY"
)


df = pd.read_csv('sorted_context.csv' ,index_col=0)

max_token_count = 4000
"""
Given a question and a dataframe containing rows of text and their
embeddings, return a text prompt to send to a Completion model
"""
# Create a tokenizer that is designed to align with our embeddings
tokenizer = tiktoken.get_encoding("cl100k_base")

# Count the number of tokens in the prompt template and question
prompt_template = """
Answer the question based on the context below, and if the question
can't be answered based on the context, say "I don't know"

Context: 

{}

---

Question: {}
Answer:"""

question = """ When did Russia invade Ukraine ?"""
current_token_count = len(tokenizer.encode(prompt_template)) + \
                        len(tokenizer.encode(question))

print("token_count = {}".format(current_token_count))
context =""
text_token_count  =0 
for text in df['text'].values:

	text_token_count += len(tokenizer.encode(text))
	if (current_token_count + text_token_count) < max_token_count :
		context = text + "\n\n###\n\n"

print(" token len context = {}".format(text_token_count))
prompt = prompt_template.format(context,question)

print(prompt)

COMPLETION_MODEL_NAME = "gpt-3.5-turbo-instruct"


completion = client.completions.create(model=COMPLETION_MODEL_NAME,prompt=prompt)
print(completion.choices[0].text)
print(dict(completion).get('usage'))
print(completion.model_dump_json(indent=2))
answer = completion.choices[0].text
print(answer)

