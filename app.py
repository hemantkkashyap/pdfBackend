from flask import Flask, request, Response, stream_with_context
import pdfplumber
from transformers import pipeline
import time
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set PyTorch CUDA memory config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'error': 'No selected file'}, 400

    # Extract text from PDF
    text = ''
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''

    if not text:
        return {'error': 'No text found in PDF.'}, 400

    # Save the extracted text for later
    global extracted_text
    extracted_text = text

    return {'message': 'File uploaded successfully. You can now stream summaries.'}, 200

@app.route('/upload/stream', methods=['GET'])
def stream_summary():
    if 'extracted_text' not in globals():
        return {'error': 'No text to summarize'}, 400

    def generate_summary():
        # Load model on demand
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=-1)

        chunks = [extracted_text[i:i + 500] for i in range(0, len(extracted_text), 500)]  # Smaller chunk size
        for chunk in chunks:
            summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            yield f"data: {summary}\n\n"
            time.sleep(1)  # Simulate processing time

        # Unload model
        del summarizer

    return Response(stream_with_context(generate_summary()), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
