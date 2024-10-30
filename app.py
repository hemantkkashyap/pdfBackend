from flask import Flask, request, Response, stream_with_context
import pdfplumber
from transformers import pipeline
import time

app = Flask(__name__)
from flask_cors import CORS
CORS(app)  # Enable CORS

summarizer = pipeline("summarization", model="sshleifer/distilbart-xsum-6-6", torch_dtype="float16", device=0)


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

    # Save the extracted text to process later
    global extracted_text
    extracted_text = text

    return {'message': 'File uploaded successfully. You can now stream summaries.'}, 200

@app.route('/upload/stream', methods=['GET'])
def stream_summary():
    if 'extracted_text' not in globals():
        return {'error': 'No text to summarize'}, 400

    def generate_summary():
        chunks = [extracted_text[i:i + 1000] for i in range(0, len(extracted_text), 1000)]
        for chunk in chunks:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            yield f"data: {summary}\n\n"
            time.sleep(1)  # Simulate processing time

    return Response(stream_with_context(generate_summary()), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
