#!/usr/bin/env python3
"""
matcher.py

This script processes FAQ questions from preprocess.py, converts them to TF-IDF vectors,
and provides a function to find the best matching FAQ question for a given user query
using cosine similarity.
"""

import os
import sys
import subprocess

# Auto-install requirements if missing
required_packages = {
    'sklearn': 'scikit-learn',
    'numpy': 'numpy'
}

for module_name, pip_name in required_packages.items():
    try:
        __import__(module_name)
    except ImportError:
        print(f"\033[93m[{module_name.upper()} MISSING] '{pip_name}' is not installed. Installing...\033[0m")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"\033[92m[{module_name.upper()} INSTALLED] Successfully installed {pip_name}.\033[0m")
        except Exception as e:
            print(f"\033[91m[ERROR] Failed to install {pip_name} automatically: {e}\033[0m")
            print(f"Please install it manually using: pip install {pip_name}")
            sys.exit(1)

# Now import the dependencies safely
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import preprocessing pipeline
try:
    from preprocess import load_and_preprocess_faqs, preprocess_text
except ImportError:
    # If preprocess.py is in the same directory but sys.path doesn't include it
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocess import load_and_preprocess_faqs, preprocess_text

# ANSI color styling helper for premium terminal output
class Style:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "faqs.csv")

# Load and preprocess FAQs
print(f"{Style.CYAN}[SYSTEM] Loading FAQs from {CSV_PATH}...{Style.ENDC}")
raw_faq_data = []

# If CSV doesn't exist, we can let preprocess.py's main handle it or check here
if not os.path.exists(CSV_PATH):
    print(f"{Style.WARNING}[WARNING] faqs.csv not found. Running preprocess.py to generate sample FAQs...{Style.ENDC}")
    try:
        # Run preprocess.py as a script to auto-generate the sample csv
        preprocess_script = os.path.join(SCRIPT_DIR, "preprocess.py")
        subprocess.run([sys.executable, preprocess_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"{Style.FAIL}[ERROR] Failed to auto-generate faqs.csv: {e}{Style.ENDC}")

# Load the FAQs
# load_and_preprocess_faqs returns list of (cleaned_question, answer) tuples
cleaned_faqs = load_and_preprocess_faqs(CSV_PATH)

if not cleaned_faqs:
    print(f"{Style.FAIL}[ERROR] No FAQ data loaded. Please make sure faqs.csv contains valid data.{Style.ENDC}")
    # Initialize empty variables to avoid NameErrors
    faq_questions = []
    faq_answers = []
    vectorizer = None
    tfidf_matrix = None
else:
    # Separate questions and answers while keeping index alignment
    # cleaned_faqs is list of (cleaned_question, answer)
    # We also keep raw questions for display purposes by reading the CSV again,
    # or just storing them if needed. But let's load raw data so we can print nicely.
    faq_questions = [item[0] for item in cleaned_faqs]
    faq_answers = [item[1] for item in cleaned_faqs]
    
    # We also want to display the original questions in the demo, so let's load original questions
    original_questions = []
    try:
        import csv
        with open(CSV_PATH, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row:
                    original_questions.append(row[0])
    except Exception:
        original_questions = faq_questions  # Fallback to cleaned questions if reading fails
        
    print(f"{Style.GREEN}[SUCCESS] Successfully loaded and preprocessed {len(faq_questions)} FAQ items.{Style.ENDC}")

    # Initialize and fit the TfidfVectorizer on the cleaned FAQ questions
    # Since the input text is already preprocessed (lowercased, stopwords removed, lemmatized),
    # we don't need any complex tokenization or lowercasing from TfidfVectorizer.
    # Passing token_pattern=r"(?u)\b\w+\b" ensures single-character words aren't ignored if they are lemmatized.
    vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = vectorizer.fit_transform(faq_questions)


def find_best_match(user_question):
    """
    Takes a raw user input, preprocesses it using the same NLTK pipeline,
    vectorizes it, computes cosine similarity against all FAQ questions,
    and returns the index of the best match along with its similarity score.
    
    Parameters:
        user_question (str): The raw question from the user.
        
    Returns:
        tuple: (best_index, similarity_score)
               best_index (int): Index of the closest match, or -1 if no match.
               similarity_score (float): Cosine similarity score.
    """
    if not vectorizer or tfidf_matrix is None or len(faq_questions) == 0:
        print(f"{Style.FAIL}[ERROR] Vectorizer is not initialized or FAQ list is empty.{Style.ENDC}")
        return -1, 0.0

    if not isinstance(user_question, str) or not user_question.strip():
        return -1, 0.0

    # 1. Preprocess the raw user input using the NLTK pipeline from preprocess.py
    cleaned_user_q = preprocess_text(user_question)
    
    # If the user question resolves to an empty string after preprocessing (e.g., only stop words/punctuation),
    # we return a default fallback since it won't produce a useful TF-IDF vector.
    if not cleaned_user_q.strip():
        return -1, 0.0

    # 2. Vectorize the cleaned user query using the pre-fitted vectorizer
    user_vector = vectorizer.transform([cleaned_user_q])

    # 3. Compute cosine similarity against all FAQ question TF-IDF vectors
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # 4. Find the index of the best match
    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])

    return best_index, best_score


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"{Style.BOLD}{Style.HEADER}    FAQ matcher.py - Similarity Demonstration    {Style.ENDC}")
    print("=" * 60)
    
    # Display the loaded FAQs
    print(f"\n{Style.BOLD}Available FAQ Questions in Database:{Style.ENDC}")
    for idx, orig_q in enumerate(original_questions):
        print(f"  {Style.BLUE}[{idx}]{Style.ENDC} {orig_q}")
    print("-" * 60)

    # Test queries to demonstrate matching accuracy
    test_queries = [
        "How can I return an item?",
        "Can I track my package?",
        "Do you ship to Germany or other countries?",
        "I want to stop my monthly subscription",
        "What is the weather today?"  # Out of domain query
    ]

    print(f"\n{Style.BOLD}Running Search Matches:{Style.ENDC}")
    for query in test_queries:
        best_idx, score = find_best_match(query)
        
        print(f"\n{Style.BOLD}User Query:{Style.ENDC} \"{query}\"")
        print(f"{Style.CYAN}Preprocessed Query:{Style.ENDC} \"{preprocess_text(query)}\"")
        
        if best_idx != -1 and score > 0.15:  # Arbitrary threshold to determine if it's a reasonable match
            matched_q = original_questions[best_idx]
            matched_a = faq_answers[best_idx]
            print(f"{Style.GREEN}[FOUND] Match Found (Score: {score:.4f}){Style.ENDC}")
            print(f"  {Style.BOLD}Best FAQ Match [{best_idx}]:{Style.ENDC} {matched_q}")
            print(f"  {Style.BOLD}Answer:{Style.ENDC} {matched_a}")
        else:
            print(f"{Style.FAIL}[NONE] No confident match found (Best Score: {score:.4f}){Style.ENDC}")
            if best_idx != -1:
                print(f"  (Best candidate was [{best_idx}] with score {score:.4f}, but similarity was too low)")
    print("=" * 60 + "\n")
