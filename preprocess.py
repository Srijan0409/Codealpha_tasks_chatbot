import csv
import re
import sys
import subprocess
import os
import pandas as pd

# Auto-install nltk if missing
try:
    import nltk
except ImportError:
    print("nltk is not installed. Installing nltk...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
        import nltk
    except Exception as e:
        print(f"Failed to install nltk automatically: {e}")
        print("Please install it manually using: pip install nltk")
        sys.exit(1)

# Auto-download required NLTK resources
required_resources = {
    'punkt': 'tokenizers/punkt',
    'punkt_tab': 'tokenizers/punkt_tab',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
    'omw-1.4': 'corpora/omw-1.4'
}

for resource, path in required_resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        print(f"Downloading NLTK resource '{resource}'...")
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Failed to download NLTK resource '{resource}': {e}")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    """
    Preprocess input text:
    1. Lowercase
    2. Remove punctuation and special characters
    3. Tokenize
    4. Remove stopwords
    5. Lemmatize
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation and special characters (keep alphanumeric and whitespace)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 3. Tokenize
    tokens = word_tokenize(text)
    
    # 4. Remove stopwords & 5. Lemmatize
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        # Fallback to standard download if not loaded
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        
    try:
        lemmatizer = WordNetLemmatizer()
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        lemmatizer = WordNetLemmatizer()
        
    cleaned_tokens = []
    for word in tokens:
        if word not in stop_words:
            try:
                lemma = lemmatizer.lemmatize(word)
            except Exception:
                # Fallback if lemmatizer fails
                lemma = word
            cleaned_tokens.append(lemma)
            
    # Reconnect tokens into a single string
    return " ".join(cleaned_tokens)

def load_and_preprocess_faqs(csv_path):
    """
    Loads faqs.csv using pandas, preprocesses the questions, and returns
    a tuple of three lists: (processed_questions, original_questions, original_answers).
    """
    processed_questions = []
    original_questions = []
    original_answers = []
    
    if not os.path.exists(csv_path):
        print(f"Error: The file {csv_path} was not found.")
        return processed_questions, original_questions, original_answers
        
    try:
        # Since the CSV contains unquoted commas in the answer field,
        # we read using csv.reader, re-assemble split fields, and convert to pandas DataFrame.
        rows = []
        with open(csv_path, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader, None)
            if header:
                for row in reader:
                    if len(row) >= 2:
                        q = row[0]
                        a = ",".join(row[1:])
                        rows.append({"question": q, "answer": a})
                        
        df = pd.DataFrame(rows)
        
        # Normalize column headers to lowercase and strip whitespace
        df.columns = df.columns.str.strip().str.lower()
        
        if 'question' in df.columns and 'answer' in df.columns:
            # Drop rows where question or answer is missing
            df = df.dropna(subset=['question', 'answer'])
            
            for _, row in df.iterrows():
                q = str(row['question'])
                a = str(row['answer'])
                cleaned_q = preprocess_text(q)
                processed_questions.append(cleaned_q)
                original_questions.append(q)
                original_answers.append(a)
        else:
            print("Error: CSV file must contain 'question' and 'answer' columns.")
    except Exception as e:
        print(f"Error reading CSV with pandas: {e}")
        
    return processed_questions, original_questions, original_answers

if __name__ == "__main__":
    # Resolve CSV file path in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "faqs.csv")
    
    # Auto-generate a sample faqs.csv if it does not exist
    if not os.path.exists(csv_path):
        print(f"Creating a sample 'faqs.csv' at '{csv_path}' for testing...")
        sample_faqs = [
            {"question": "What is your return policy?", "answer": "You can return any item within 30 days of purchase."},
            {"question": "How can I track my order?", "answer": "Use the tracking link sent to your registration email."},
            {"question": "Do you offer international shipping?", "answer": "Yes, we ship worldwide to over 150 countries."},
            {"question": "Can I cancel my subscription?", "answer": "Yes, you can cancel anytime from your account settings."}
        ]
        try:
            with open(csv_path, mode='w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=["question", "answer"])
                writer.writeheader()
                writer.writerows(sample_faqs)
        except Exception as e:
            print(f"Failed to create sample CSV: {e}")

    print(f"Processing CSV file: {csv_path}")
    processed_questions, original_questions, original_answers = load_and_preprocess_faqs(csv_path)
    
    print("\nResulting Cleaned Python Lists:")
    print("Questions sample (first 2):", processed_questions[:2])
    print("Original Questions sample (first 2):", original_questions[:2])
    print("Answers sample (first 2):", original_answers[:2])
