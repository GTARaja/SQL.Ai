import google.generativeai as genai
from dotenv import load_dotenv
from IPython.display import display
from IPython.display import Markdown
import textwrap
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


GOOGLE_API_KEY='AIzaSyD0W_tAHffQRKH5_sZMa_Tuy0NJfrsDK8E'

genai.configure(api_key=GOOGLE_API_KEY)
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
model = genai.GenerativeModel('gemini-pro')
#response = model.generate_content("What is the meaning of life?")
#print(response.text)

loader = PyPDFDirectoryLoader("error")
data = loader.load_and_split()
#print(data)
context = "\n".join(str(p.page_content) for p in data)
print("The total number of words in the context:", len(context))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
context = "\n\n".join(str(p.page_content) for p in data)
texts = text_splitter.split_text(context)
#print(len(texts))
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
vector_index = Chroma.from_texts(texts, embeddings).as_retriever()
while(True):

    question = "what options determine how the financial portion of inventory transactions are recorded in Merchandising"
    question = input("Enter your Query")

    docs = vector_index.get_relevant_documents(question)
    #print(docs)
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3,apikey='AIzaSyD0W_tAHffQRKH5_sZMa_Tuy0NJfrsDK8E')
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain({"input_documents":docs, "question": question}, return_only_outputs=True)
    print(response['output_text'])