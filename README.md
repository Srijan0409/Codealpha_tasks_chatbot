# FAQ Chatbot for Healthcare & Diet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![CodeAlpha Internship](https://img.shields.io/badge/CodeAlpha-Internship_Task-purple.svg)](https://www.codealpha.tech/)

An NLP-powered local FAQ retrieval assistant that matches healthcare, diet, and hospital queries using TF-IDF and cosine similarity.

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How It Works](#how-it-works)
- [Sample Interaction](#sample-interaction)
- [FAQ Dataset](#faq-dataset)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## About the Project
This project is an NLP-based FAQ chatbot designed to match user inquiries with pre-written healthcare, hospital, and diet-related responses using classical text similarity algorithms. It operates completely locally and does not rely on any external LLM or AI APIs, ensuring data privacy and fast response times. By retrieving answers from a curated CSV database, it serves as a lightweight assistant to resolve common patient queries. This project was developed as Task 2 during the CodeAlpha software engineering internship program.

## Features
- NLP-based text preprocessing (tokenization, stopword removal, lemmatization).
- TF-IDF vectorization for word weight calculation.
- Cosine similarity matching using scikit-learn.
- Fallback response mechanism for unmatched or low-confidence questions.
- 50+ manually curated healthcare and diet FAQ question-answer pairs.
- Confidence score display for matched FAQs in terminal and logs.
- Interactive terminal-based user conversation interface.
- Sleek and modern optional Flask web UI with a glassmorphism theme and AJAX communication.
- Dynamic FAQ dataset reloading (automatically refreshes when the CSV is modified).
- Input validation to handle empty inputs and very short queries (under 3 characters).
- "Did you mean...?" suggestions for mid-confidence matching scores (between 0.2 and 0.3).

## Tech Stack
| Technology | Purpose |
| --- | --- |
| Python 3.x | Primary programming language used for script execution and application logic. |
| NLTK | Natural Language Toolkit used for text preprocessing (tokenization, stopword removal, and lemmatization). |
| SpaCy | Advanced NLP library for optional linguistic analysis and extended entity extraction. |
| scikit-learn | Machine learning library used to compute TF-IDF vector matrices and calculate cosine similarity scores. |
| pandas | Data analysis library for structured FAQ dataset manipulation and loading (optional). |
| Flask (optional) | Micro-web framework used to build and run the web application interface and `/chat` API route. |
| CSV | Local database source (`faqs.csv`) containing the question-answer pairs. |

## Project Structure
```text
faq-chatbot/
├── faqs.csv
├── preprocess.py
├── matcher.py
├── chatbot.py
├── app.py (optional)
├── templates/
│   └── index.html (optional)
├── requirements.txt
└── README.md
```

## Installation & Setup
1. Clone the repository to your local system:
   ```bash
   git clone https://github.com/your-username/faq-chatbot.git
   cd faq-chatbot
   ```
2. Create a virtual environment to manage dependencies:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   * On Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * On Unix/macOS:
     ```bash
     source venv/bin/activate
     ```
4. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Download the necessary NLTK data resource packages:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```
6. Run the terminal-based interactive chatbot:
   ```bash
   python chatbot.py
   ```
7. (Optional) Run the Flask web app server and visit `http://127.0.0.1:5000` in your web browser:
   ```bash
   python app.py
   ```

## How It Works
1. **Load CSV**: The chatbot reads the local `faqs.csv` file to extract questions and answers.
2. **Preprocess Text**: The text undergoes lowercasing, special characters removal, tokenization (using NLTK), stopword filtering, and lemmatization (using WordNetLemmatizer) to normalize vocabulary terms.
3. **Vectorize using TF-IDF**: The vectorizer converts the preprocessed FAQ questions into numeric vectors using Term Frequency-Inverse Document Frequency (TF-IDF).
4. **Compute Cosine Similarity**: When a user inputs a query, it is preprocessed and converted into a vector. The system calculates the cosine similarity score between the query vector and all FAQ question vectors.
5. **Return Match or Fallback**: If the top similarity score is 0.3 or higher, the matched answer is returned. If the score is between 0.2 and 0.3, a "Did you mean...?" question is suggested. Otherwise, a fallback message is returned.

## Sample Interaction
```text
============================================================
           Welcome to the CodeAlpha FAQ Chatbot!          
============================================================
Type 'exit', 'quit', or 'bye' to end the conversation.
Loaded 50 FAQs. Ready to answer your questions.

You: How can I prevent common infections?
Chatbot: Wash hands regularly, eat well, sleep adequately, stay vaccinated, and avoid close contact with sick individuals.
(Matched FAQ #49 with score: 0.7071)

You: 
Chatbot: Please type something.

You: hi
Chatbot: Please provide more detail (at least 3 characters).

You: stop my appointments
Chatbot: Did you mean: 'How do I cancel or reschedule my appointment?'
(Suggested FAQ #5 with score: 0.2887)

You: What are the hospital visiting hours?
Chatbot: Visiting hours are from 9am to 12pm and 4pm to 7pm on all days including weekends.
(Matched FAQ #2 with score: 1.0000)

You: quit
Chatbot: Goodbye! Have a great day!
```

## FAQ Dataset
The dataset contains 50+ manually curated question-answer pairs covering hospital appointments, diet and nutrition, medicines, health checkups, and wellness topics stored in `faqs.csv`. The questions are structured to capture common inquiries regarding scheduling consultations, healthy dieting guidelines, managing blood pressure and cholesterol levels, daily hydration needs, and recognizing early warning signs of cardiovascular emergencies. This localized database allows the chatbot to retrieve reliable medical information instantly without any external network dependencies.

## Future Improvements
- Adding a web UI with Flask to provide a modern, responsive chat page (completed).
- Integrating a transformer model like BERT to capture deeper semantic relationships instead of exact word overlaps.
- Adding multilingual support to answer patient queries in multiple regional languages.
- Implementing voice input and text-to-speech features for enhanced patient accessibility.
- Expanding the FAQ dataset by integrating a database parser that pulls from open-source medical platforms.

## Acknowledgements
We would like to express our gratitude to the CodeAlpha internship program team for providing the learning framework, guidelines, and prompt coordination. This project was built and submitted as Task 2 of the internship program.

## License
This project is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT).
