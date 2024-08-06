import logging
from flask import Flask, request, jsonify, render_template
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize OpenAI Embeddings with your API key
openai_api_key = "sk-proj-kogDv8AdIn3QAUaCLfKNT3BlbkFJyTnzROdCZ80hRkCV8t2U"
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Qdrant client setup
url = "https://e9c601c3-b7ae-40bc-ba31-fa7a6448f627.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key = "NExfPwZ55hA9RGpkwJCn977eEl8UlS68D-qLuSYtpq_TGGfC2pVhYA"


client = QdrantClient(url=url, api_key=api_key)
llm = ChatOpenAI(model="gpt-4", temperature=0.6, openai_api_key=openai_api_key)
db = Qdrant(client=client, embeddings=embeddings, collection_name="my_documents")

# Define a simple document class
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Assuming db is your database connection object and llm is your language model
def summarize_documents(query, k=10):
    # Perform similarity search
    data = db.similarity_search_with_score(query=query, k=k)

    # Extract the Document objects from the search results
    docs = [result[0] for result in data]
    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please summarize the content in concise points.
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    
    # Reduce
    reduce_template = """The following is a set of summaries:
    {docs}
    Take these and distill them into a final, consolidated summary with the bullet points and dont include any other words or contents other than the summarization.
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Combining documents by mapping a chain over them, then combining results
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combining and iteratively reducing the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=4000,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

    # Run the MapReduce chain
    result = map_reduce_chain.invoke(docs)

    # Return the final output
    return result["output_text"]

def datewise_similarity_search(query, date):
    try:
        filter_query = {"date": date}
        logging.debug(f"Filter query for datewise search: {filter_query}")
        results = db.similarity_search_with_score(
            query=query,
            filter=filter_query,
            k=10
        )
        logging.debug(f"Datewise similarity search results: {results}")
        return results
    except Exception as e:
        logging.error(f"Error during datewise similarity search: {e}")
        return []

def category_filtered_similarity_search(query="", category=None):
    try:
        filter_query = {"category": category}
        logging.debug(f"Filter query for category search: {filter_query}")
        results = db.similarity_search_with_score(
            query=query,
            filter=filter_query,
            k=15
        )
        logging.debug(f"Category filtered similarity search results: {results}")
        return results
    except Exception as e:
        logging.error(f"Error during category filtered similarity search: {e}")
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query')
    category = data.get('category')
    date = data.get('date')
    
    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    logging.debug(f"Received query: {query_text}")

    try:
        if date:
            logging.debug(f"Performing datewise similarity search for date: {date}")
            results = datewise_similarity_search(query_text, date)
        elif category:
            logging.debug(f"Performing category filtered similarity search for category: {category}")
            results = category_filtered_similarity_search(query_text, category)
        else:
            logging.debug("Performing general similarity search")
            results = db.similarity_search_with_score(query=query_text, k=10)

        response_content = [{
            'metadata': result[0].metadata,
            'page_content': result[0].page_content,
            'score': result[1]
        } for result in results]
        
        if not response_content:
            response_content = "No relevant articles found."

        logging.debug(f"Query response: {response_content}")
        return jsonify(response_content)
    except Exception as e:
        logging.error(f"Error during query processing: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/datewise-search', methods=['POST'])
def datewise_search():
    data = request.json
    date = data.get('date')
    
    if not date:
        return jsonify({"error": "No date provided"}), 400

    logging.debug(f"Performing datewise similarity search for date: {date}")
    
    results = datewise_similarity_search("", date)
    
    response = [{
        'metadata': result[0].metadata,
        'page_content': result[0].page_content,
        'score': result[1]
    } for result in results]

    return jsonify(response)

@app.route('/category-search', methods=['POST'])
def category_search():
    data = request.json
    category = data.get('category')
    query_text = data.get('query', "")

    if not category:
        return jsonify({"error": "No category provided"}), 400

    logging.debug(f"Performing category filtered similarity search for category: {category}")
    
    results = category_filtered_similarity_search(query_text, category)
    
    response = [{
        'metadata': result[0].metadata,
        'page_content': result[0].page_content,
        'score': result[1]
    } for result in results]

    return jsonify(response)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    query = data.get('query')

    if not query:
        logging.error("No query provided for summarization")
        return jsonify({"error": "No query provided"}), 400

    logging.debug(f"Received summarization query: {query}")

    try:
        summary = summarize_documents(query)
        logging.debug(f"Summarization result: {summary}")
        return jsonify({"summary": summary})
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)