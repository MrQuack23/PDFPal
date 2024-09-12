from flask import Flask, render_template, request, jsonify
import os
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from  llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import VectorStoreIndex

app = Flask(__name__)

documents = None
nodes = None
index = None
query_engine = None
response_synthesizer = None


UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize LLM and Embedding models
llm = Ollama(model="phi3.5", request_timeout=600)
embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
Settings.llm = llm
Settings.embed_model = embedding_model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global file_path
    if 'pdf' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['pdf']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file and file.filename.endswith('.pdf'):
        # Save the PDF file to the upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        print(f'File uploaded: {file_path}')

        # Process the PDF
        try:
            # Load and split the document
            documents = SimpleDirectoryReader(input_dir=str(UPLOAD_FOLDER)).load_data()
            print("doc ready")
            splitter = SemanticSplitterNodeParser(
                buffer_size=1, 
                breakpoint_percentile_threshold=95, 
                embed_model=embedding_model
            )
            nodes = splitter.get_nodes_from_documents(documents)
            print('nodes ready')
            # Index the nodes using VectorStoreIndex
            index = VectorStoreIndex(nodes)
            print("index ready")
            # Configure retriever and query engine
            response_synthesizer = get_response_synthesizer()
            global query_engine
            query_engine = RetrieverQueryEngine(
                retriever=index.as_query_engine(verbose=True),
                response_synthesizer=response_synthesizer,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)],
            )            

            return jsonify({'message': 'File uploaded and processed successfully'}), 200
        except Exception as e:
            return jsonify({'message': f'Error processing PDF: {str(e)}'}), 500
    else:
        return jsonify({'message': 'Invalid file format, please upload a PDF file'}), 400
    

@app.route('/query', methods=['POST'])
def query():
    text = request.form['text']
    print(f'Query: {text}')
    return render_template('index.html', {'user_message': text})



if __name__ == '__main__':
    app.run(debug=True)