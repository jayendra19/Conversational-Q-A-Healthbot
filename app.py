import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai 
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS

from langchain.schema import HumanMessage,SystemMessage,AIMessage


# Define the system and human templates
system_template = "You are a knowledgeable doctor AI assistant. When the user provides their symptoms, you should only diagnose the disease give only name  and suggest only appropriate treatment and medicine so can normal user understand."

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.5, convert_system_message_to_human=True)

# Streamlit UI
st.set_page_config(page_title="Conversational Q&A Healthbot")
st.header("Hey Let's Chat")

if 'flow' not in st.session_state:
    st.session_state['flow'] = [SystemMessage(content=system_template)]

# Functions to load AI chat model
def get_chatmodel_response(symptoms):
    st.session_state['flow'].append(HumanMessage(content=symptoms))
    answer = model.invoke(st.session_state['flow'])  # Invoke is used to invoke performance model
    st.session_state['flow'].append(AIMessage(content=answer.content))
    return answer.content

symptoms = st.text_input('Enter your Symptoms: ', key="symptoms")
submit = st.button('Get Diagnosis')

# If ask button is clicked
if submit:
    response = get_chatmodel_response(symptoms)
    st.subheader("The Response is")
    st.write(response)





#this function is used to go all the pages and extract the texts from each pages
def get_pdf_text(pdf):
    # Read the PDF file from the BytesIO object
    pdf_reader =  PdfReader(pdf)
    text=""
    for page in range(len(pdf_reader.pages)):
        text+=pdf_reader.pages[page]. extract_text()

    return text

#Now i have text now i'll divide this text into chunks 
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversation():
    prompt_temp="""You are a knowledgeable doctor Answer the question as detailed as possible from the provided context,make sure to provide all the details ,if answer is not in the 
    provided context just say ,"answer is not available in the provided context",don't provide the wrong answer\n\n\
    Context:\n{context}?\n
    question:\n{question}\n

    Answer:
    """

    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompts=PromptTemplate(template=prompt_temp,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompts)

    return chain

def user_input(user_question):
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db=FAISS.load_local("faiss_index",embedding)
    docs=new_db.similarity_search(user_question)

    chain=get_conversation()

    response=chain({"input_documents":docs,"question":user_question},return_only_outputs=True)

    print(response)
    st.sidebar.write("Reply:",response["output_text"])

# Add a sidebar title
st.sidebar.title("Upload your medical report")
# Add a PDF uploader to the sidebar
pdf_docs = st.sidebar.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

# Add a button to process the PDFs
if st.sidebar.button("Submit & Process"):
    for pdf_doc in pdf_docs:
        raw_text = get_pdf_text(pdf_doc)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
    st.sidebar.success("Done")

user_question = st.sidebar.text_input("Ask a Question from the PDF Files")
# Add a button to answer the question
if user_question and st.sidebar.button("Answer Question"):
    user_input(user_question)























