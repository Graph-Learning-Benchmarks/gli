'''

This script is intended to used as Github Action
02/25/2024, created by Benhao Huang

'''

import argparse
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import base64


# Set the OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your openai key"

# Read the OpenAI API key from an environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Ensure the OpenAI API key is set in the environment for library use
os.environ["OPENAI_API_KEY"] = openai_api_key

root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


vector_path = os.path.join(root, ".github")

# Initialize the embeddings
embeddings = OpenAIEmbeddings()

# Setup argument parser
argparser = argparse.ArgumentParser()
argparser.add_argument("--query", help="your question")

query = os.getenv("QUESTION", "say: Please provide something.")

# Initialize GPT-4
gpt4 = ChatOpenAI(model="gpt-4-0125-preview")

# Load the vector store
index_file_path = os.path.join(vector_path, "vector_store")
vector = FAISS.load_local(index_file_path, embeddings, index_name="my_vector_index")

# Create the prompt template
prompt = ChatPromptTemplate.from_template("""You should answer following problem strictly follow the documents provided. Answer the following question based only on the provided context in a format which is suitable for Github comment:

<context>
{context}
</context>

Question: {input}""")

# Create the document chain
document_chain = create_stuff_documents_chain(gpt4, prompt)

# Create the retriever
retriever = vector.as_retriever()

# Create the retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

query_str = "{}".format(query)

# Invoke the retrieval chain with a sample input
response = retrieval_chain.invoke({"input": query_str})["answer"]

encoded_answer = base64.b64encode(response.encode("utf-8")).decode("utf-8")

print(encoded_answer)

