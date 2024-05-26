#%%
import os
from dotenv import load_dotenv

load_dotenv()

# API KEY
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
#%%
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'llm-chatbot-gpt'
#%%
from langchain_community.embeddings.openai import OpenAIEmbeddings
from openai import AsyncOpenAI
from IPython.display import Markdown

embedding = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
#%%
pVS = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding
)
#%%
primer3 = f"""Bây giờ bạn hãy đóng vai trò là một luật sư xuất sắc về luật hôn nhân và gia đình ở Việt Nam.
Tôi sẽ hỏi bạn các câu hỏi về tình huống thực tế liên quan tới luật hôn nhân và gia đình. Bạn hãy tóm tắt tình huống
và đưa ra các câu hỏi ngắn gọn gồm từ khoá liên quan tới pháp luật được suy luận từ phần thông tin có trong tình huống. Các câu trả lời của bạn
đều là tiếng việt.

Sau đây là một số ví dụ và phần tóm tắt:
1. Tình huống: Biết mình đủ tuổi kết hôn và đáp ứng các điều kiện kết hôn, Anh S và chị Y dự định đi đăng ký kết hôn trước khi tổ chức lễ cưới 02 tháng. Chị Y và anh S có hộ khẩu thường trú ở hai tỉnh khác nhau, anh chị muốn biết việc đăng ký kết hôn thực hiện tại cơ quan nào và cần thực hiện thủ tục gì?
-> Tóm tắt: Thủ tục đăng ký kết hôn là gì, hộ khẩu thường trú trong thủ tục kết hôn

2. Ông bà B có con trai đã 25 tuổi, bị bệnh đao bẩm sinh. Vì muốn lấy vợ cho con trai, bà B đã tìm cách vu cáo cho chị Y – người giúp việc lấy trộm số tiền 1.000.000 đồng. Bà B  đe dọa nếu chị Y không muốn bị báo công an, không muốn bị đi tù thì phải lấy con trai bà, vừa được làm chủ nhà, không phải làm người giúp việc lại có cuộc sống sung túc. Vì nhận thức hạn chế, trình độ văn hóa thấp nên chị Y đã đồng ý lấy con trai bà B. Hôn lễ chỉ tổ chức giữa hai gia đình mà không làm thủ tục đăng ký kết hôn tại phường. Việc làm của bà B có vi phạm pháp luật không? Nếu có thì bị xử phạt như thế nào?
-> Tóm tắt: cưỡng ép kết hôn có bị xử phạt không, cưỡng ép kết hôn bị xử phạt như thế nào 

3. Tôi đã kết hôn được 6 tháng, nhưng chưa chuyển hộ khẩu về nhà chồng (ở xã X, huyện B, tỉnh A), hộ khẩu của tôi vẫn đang ở nhà bố mẹ đẻ (xã Y, huyện C, tỉnh D). Nay tôi có nguyện vọng chuyển hộ khẩu về nhà chồng thì có được không và thủ tục thực hiện như thế nào? Ai có thẩm quyền giải quyết?
-> tóm tắt: có được chuyển hộ khẩu về nhà chồng không, Thủ tục chuyển hộ khẩu, Ai giải quyết thủ tục chuyển hộ khẩu
"""

primer1 = f"""Bây giờ bạn hãy đóng vai trò là một luật sư xuất sắc về luật hôn nhân và gia đình ở Việt Nam.
Tôi sẽ hỏi bạn các câu hỏi về tình huống thực tế liên quan tới luật hôn nhân và gia đình. Bạn sẽ trả lời câu hỏi
dựa trên thông tin tôi cung cấp và thông tin có trong câu hỏi. Nếu thông tin tôi cung cấp không đủ để trả lời
hãy nói rằng 'Tôi không biết'. Các câu trả lời của bạn đều là tiếng việt. 
Lưu ý: Nêu rõ điều luật số mấy để trả lời tình huống.
"""
#%%
import chainlit as cl

@cl.on_message
async def main(mess):
    query = str(mess.content)
    chats = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer3},
            {"role": "user", "content": query}
        ]
    )
    
    new_query = chats.choices[0].message.content
    context = ""

    texts = new_query.split(',')
    
    for text in texts:
        print(text)
        relevant_docs = pVS.similarity_search(query=text, k=1)
    
        context += relevant_docs[0].page_content

    relevant_docs = pVS.similarity_search(query=query, k=1)
    context += relevant_docs[0].page_content
    context += "\n" + query

    chats = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": primer1},
            {"role": "user", "content": context}
        ]
    )

    # display(Markdown(chats.choices[0].message.content))
    await cl.Message(content=chats.choices[0].message.content).send()