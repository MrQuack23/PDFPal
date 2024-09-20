from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import VectorStoreIndex

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize global variables
index = None
query_engine = None

def initialize_chatbot():
    global index, query_engine

    # Initialize LLM
    llm = Ollama(model="phi3.5", request_timeout=600)

    # Set up embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    Settings.llm = llm

    # Load and process documents
    documents = SimpleDirectoryReader(UPLOAD_FOLDER).load_data()
    print('document')
    # Split documents into nodes
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, 
        breakpoint_percentile_threshold=95, 
        embed_model=Settings.embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print('nodes')
    # Create index
    index = VectorStoreIndex(nodes)
    print('index')
    # Configure retriever and response synthesizer
    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)
    response_synthesizer = get_response_synthesizer()

    # Set up query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)],
    )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({'message': 'No file part'})
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'message': 'No selected file'})
    
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        initialize_chatbot()
        return jsonify({'message': 'File uploaded successfully'})
    else:
        return jsonify({'message': 'Invalid file type'})

@app.route('/query', methods=['POST'])
def query():
    if query_engine is None:
        return jsonify({'message': 'Please upload a PDF file first'})

    user_query = request.form.get('text')
    print(user_query)
    if not user_query:
        return jsonify({'message': 'No query provided'})

    response = query_engine.query(user_query)
    print(response)
    return jsonify({'user_query': user_query, 'ai_response': str(response)})

if __name__ == '__main__':
    app.run(debug=True)