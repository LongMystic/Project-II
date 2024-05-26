#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
from dotenv import load_dotenv

load_dotenv()

# API KEY
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')


# In[6]:


from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'langchain-vector'
index = pc.Index(index_name)


# In[10]:


from langchain_community.embeddings.openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from IPython.display import Markdown

embedding = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# In[12]:


# system message to 'prime' the model
# primer = f"""You are Q&A bot. A highly intelligent system that answers
# user questions based on the information provided by the user above
# each question. If the information can not be found in the information
# provided by the user you truthfully say "I don't know. Đặc biệt hãy trả lời bằng ngôn ngữ Tiếng Việt".
# """
primer = f"""Bạn là một trợ lý luật sư về luật hôn nhân và gia đình. Bạn sẽ trả lời câu hỏi
của tôi dựa trên thông tin được cung cấp cho từng câu hỏi bởi tôi. Nếu câu trả lời không được tim thấy
trong phần thông tin mà tôi cung cấp, hãy thành thật nói rằng "Tôi không biết". Các câu trả lời của bạn
hoàn toàn là tiếng Việt      
"""


# In[13]:


import chainlit as cl

@cl.on_message
async def main(mess):
    query = str(mess.content)
    vector = embedding.embed_query(query)
    res = index.query(vector=vector, top_k=3, include_metadata=True)

    contexts = [item['metadata']['text'] for item in res['matches']]

    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query
    print(augmented_query)
    chats = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query}
        ]
    )

    # display(Markdown(chats.choices[0].message.content))
    await cl.Message(content=chats.choices[0].message.content).send()


# In[ ]:




