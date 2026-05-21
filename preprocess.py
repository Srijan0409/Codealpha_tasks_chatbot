import csv
import re
import sys
import subprocess
import os

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
    Loads faqs.csv, preprocesses the questions, and returns a list of (cleaned_question, answer) tuples.
    """
    cleaned_data = []
    if not os.path.exists(csv_path):
        print(f"Error: The file {csv_path} was not found.")
        return cleaned_data
        
    try:
        with open(csv_path, mode='r', encoding='utf-8') as infile:
            # We first try to read with DictReader
            reader = csv.DictReader(infile)
            
            # Normalize headers
            headers = [h.strip().lower() for h in (reader.fieldnames or [])]
            
            if 'question' in headers and 'answer' in headers:
                # Map headers back to their original spelling
                q_col = reader.fieldnames[headers.index('question')]
                a_col = reader.fieldnames[headers.index('answer')]
                for row in reader:
                    q = row[q_col]
                    a = row[a_col]
                    cleaned_q = preprocess_text(q)
                    cleaned_data.append((cleaned_q, a))
            else:
                # Fallback: assume column 0 is question, column 1 is answer
                infile.seek(0)
                raw_reader = csv.reader(infile)
                
                # Check if there is a header row to skip
                try:
                    first_row = next(raw_reader)
                except StopIteration:
                    return cleaned_data
                    
                # Simple check if first row is header
                first_row_normalized = [col.strip().lower() for col in first_row]
                if 'question' in first_row_normalized or 'answer' in first_row_normalized:
                    # Skip first row
                    pass
                else:
                    # Re-include first row as data
                    if len(first_row) >= 2:
                        cleaned_q = preprocess_text(first_row[0])
                        cleaned_data.append((cleaned_q, first_row[1]))
                        
                for row in raw_reader:
                    if len(row) >= 2:
                        cleaned_q = preprocess_text(row[0])
                        cleaned_data.append((cleaned_q, row[1]))
                        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        
    return cleaned_data

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
    processed_faqs = load_and_preprocess_faqs(csv_path)
    
    print("\nResulting Cleaned Python List:")
    print(processed_faqs)
