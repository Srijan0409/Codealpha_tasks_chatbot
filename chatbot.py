#!/usr/bin/env python3
"""
chatbot.py

A terminal-based chatbot that matches user queries against the FAQ database.
It uses matcher.py for TF-IDF calculations and preprocess.py to retrieve FAQ answers.
"""

import os
import sys

# Import match function from matcher.py
try:
    from matcher import find_best_match
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from matcher import find_best_match

# Import loader from preprocess.py to obtain the FAQ answers list
try:
    from preprocess import load_and_preprocess_faqs
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocess import load_and_preprocess_faqs

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
    DIM = '\033[2m'

# Resolve paths and load FAQ answers
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "faqs.csv")

# Variables to store loaded state
faq_answers = []
original_questions = []
last_loaded_mtime = 0.0

def load_faq_lists():
    global faq_answers, original_questions, last_loaded_mtime
    
    if not os.path.exists(CSV_PATH):
        return
        
    try:
        mtime = os.path.getmtime(CSV_PATH)
    except OSError:
        mtime = 0.0
        
    if mtime == last_loaded_mtime and faq_answers:
        return
        
    faq_questions, original_questions, faq_answers = load_and_preprocess_faqs(CSV_PATH)
    last_loaded_mtime = mtime

# Initial load
load_faq_lists()

def get_answer(user_input):
    """
    Finds the best matching FAQ answer for a user input.
    
    Parameters:
        user_input (str): The raw text entered by the user.
        
    Returns:
        str: The matched FAQ answer, a suggestion, or a validation/fallback message.
    """
    # Dynamically check/reload if faqs.csv has been updated
    load_faq_lists()

    if not faq_answers:
        return "Sorry, the FAQ database is currently empty or failed to load."
        
    if not user_input or not user_input.strip():
        return "Please type something."
        
    if len(user_input.strip()) < 3:
        return "Please provide more detail (at least 3 characters)."
        
    # Call find_best_match to retrieve index and score
    best_idx, score = find_best_match(user_input)
    
    # Check if the similarity score is above the thresholds
    if best_idx != -1:
        if score >= 0.3:
            return faq_answers[best_idx]
        elif 0.2 <= score < 0.3:
            suggested_q = original_questions[best_idx] if best_idx < len(original_questions) else "another question"
            if suggested_q.endswith('?'):
                return f"Did you mean: '{suggested_q}'"
            else:
                return f"Did you mean: '{suggested_q}'?"
            
    return "Sorry, I could not find an answer to that question."

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"{Style.BOLD}{Style.HEADER}           Welcome to the CodeAlpha FAQ Chatbot!          {Style.ENDC}")
    print("=" * 60)
    print(f"{Style.DIM}Type 'exit', 'quit', or 'bye' to end the conversation.{Style.ENDC}")
    print(f"Loaded {len(faq_answers)} FAQs. Ready to answer your questions.\n")

    while True:
        try:
            # Get user input
            user_input = input(f"{Style.BOLD}{Style.BLUE}You:{Style.ENDC} ")
            
            # Check for exit commands
            if user_input.strip().lower() in ['exit', 'quit', 'bye']:
                print(f"\n{Style.GREEN}Chatbot: Goodbye! Have a great day!{Style.ENDC}\n")
                break
                
            if not user_input.strip():
                print(f"{Style.GREEN}Chatbot:{Style.ENDC} Please type something.\n")
                continue
                
            if len(user_input.strip()) < 3:
                print(f"{Style.GREEN}Chatbot:{Style.ENDC} Please provide more detail (at least 3 characters).\n")
                continue
                
            # Get matching answer and also retrieve details for display/debugging
            best_idx, score = find_best_match(user_input)
            
            # Print chatbot response
            if best_idx != -1 and score >= 0.3:
                response = faq_answers[best_idx]
                print(f"{Style.GREEN}Chatbot:{Style.ENDC} {response}")
                print(f"{Style.DIM}(Matched FAQ #{best_idx} with score: {score:.4f}){Style.ENDC}\n")
            elif best_idx != -1 and 0.2 <= score < 0.3:
                suggested_q = original_questions[best_idx] if best_idx < len(original_questions) else "another question"
                if suggested_q.endswith('?'):
                    print(f"{Style.GREEN}Chatbot:{Style.ENDC} Did you mean: '{suggested_q}'")
                else:
                    print(f"{Style.GREEN}Chatbot:{Style.ENDC} Did you mean: '{suggested_q}'?")
                print(f"{Style.DIM}(Suggested FAQ #{best_idx} with score: {score:.4f}){Style.ENDC}\n")
            else:
                print(f"{Style.GREEN}Chatbot:{Style.ENDC} Sorry, I could not find an answer to that question.")
                if best_idx != -1:
                    print(f"{Style.DIM}(Best candidate score was too low: {score:.4f}){Style.ENDC}\n")
                else:
                    print(f"{Style.DIM}(No matching candidate could be computed){Style.ENDC}\n")
                    
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{Style.GREEN}Chatbot: Goodbye! Have a great day!{Style.ENDC}\n")
            break
