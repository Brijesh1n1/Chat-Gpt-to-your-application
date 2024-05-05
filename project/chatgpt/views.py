# Create your views here.
from django.shortcuts import render
from .models import *
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import openai
from django.conf import settings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import json
from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from django.contrib.auth.models import User
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
import os


def prompt():
    template = "You are a helpful assistant that translates {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    return chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")

def check_answer():
    llm = OpenAI(openai_api_key=settings.OPEN_AI_KEY, temperature=0.9)
    llm.predict("What would be a good company name for a company that makes colorful socks?")
   
    users = User.objects.all()
    chat = ChatOpenAI(openai_api_key=settings.OPEN_AI_KEY, temperature=0)
    chat.predict("Translate this sentence from English to Russian. I love programming.")
    val = "colorful socks"
    prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
    prompt.format(product=val)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run("colorful socks")

total_employee = User.objects.all().count()


def chat(request):
    chats = Chat.objects.all()
    return render(request, 'chat.html', {
        'chats': chats,
    })

def answerfrom_text_file(text):
    with open("chatgpt\\static\\loc.txt", "w+") as f:
        f.write(f"\nOur company name is wangoes.\nTotal number of employees are {total_employee}.")
        f.write("My name is brijesh.\n")
    loaders = DirectoryLoader(settings.BASE_DIR, glob="**//*.txt",)
    docs = loaders.load()
    char_txt_splitter = CharacterTextSplitter(chunk_size=580, chunk_overlap = 0)
    doc_texts = char_txt_splitter.split_documents(docs)
    openA_embiddings = OpenAIEmbeddings(openai_api_key=settings.OPEN_AI_KEY)
    vstore = Chroma.from_documents(doc_texts, openA_embiddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=settings.OPEN_AI_KEY, temperature=0), chain_type="stuff", retriever=vstore.as_retriever())
    question = text
    result = qa({"query": question})
    os.remove("chatgpt\\static\\loc.txt")
    return result["result"]
   

@csrf_exempt
def Ajax(request):
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest': # Check if request is Ajax
        text = request.POST.get('text')
        response = answerfrom_text_file(text)
        if response.lower() == " i don't know.":
            print('---------------------------------')
            openai.api_key = settings.OPEN_AI_KEY # Here you have to add your api key.
            res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"{text}"}
                ]
            )
            response = res.choices[0].message["content"]
        return JsonResponse({'data': response})
    return JsonResponse({})