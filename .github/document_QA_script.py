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

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Determine the current file path
current_file_path = os.path.dirname(os.path.abspath(__file__))

# Initialize the embeddings
embeddings = OpenAIEmbeddings()

# Setup argument parser
argparser = argparse.ArgumentParser()
argparser.add_argument("--query", help="your question")

query = os.getenv("QUESTION", "say: Please provide something.")

# Initialize GPT-4
gpt4 = ChatOpenAI(model="gpt-4")

# Load the vector store
index_file_path = os.path.join(current_file_path, "vector_store")
vector = FAISS.load_local(index_file_path, embeddings, index_name="my_vector_index")

# Create the prompt template
prompt = ChatPromptTemplate.from_template("""You should answer following problem strictly follow the documents provided. Answer the following question based only on the provided context in Markdown format:

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

# Invoke the retrieval chain with a sample input
response = retrieval_chain.invoke({"input": argparser.parse_args().query})

# Print the response
print(f"::set-output name=answer::{response}")

