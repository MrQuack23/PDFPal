from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Ensure the directory exists where the files will be uploaded
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

# Route to upload a PDF file
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
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
        return jsonify({'message': 'File uploaded successfully', 'filename': file.filename}), 200
    else:
        return jsonify({'message': 'Invalid file format, please upload a PDF file'}), 400


if __name__ == '__main__':
    app.run(debug=True)
