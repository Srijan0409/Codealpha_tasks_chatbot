import os
import sys
from flask import Flask, request, jsonify, render_template

# Ensure the chatbot directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    from chatbot import get_answer
except ImportError:
    # Fallback to local import if run from a different directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from chatbot import get_answer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # Retrieve the request JSON payload
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    
    # Call the similarity matching logic from chatbot.py
    answer = get_answer(message)
    
    # Return JSON response containing the answer
    return jsonify({'answer': answer})

if __name__ == '__main__':
    # Start the Flask development server on port 5000
    app.run(debug=True, port=5000)
