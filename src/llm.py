import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv() # <-- ADD THIS BACK IN

def get_llm():
    """
    Initializes and returns a remote LLM instance using Groq.
    It reads the API key from the .env file.
    """
    print("Loading remote LLM: llama-3.1-8b-instant via Groq...")
    
    try:
        # Use the model from your screenshot
        llm = ChatGroq(model="llama-3.1-8b-instant")
        
        llm.invoke("Test prompt")
        print("Groq LLM loaded successfully.")
        return llm
    except Exception as e:
        print(f"Error loading Groq LLM: {e}")
        print("Please ensure your GROQ_API_KEY is set correctly in your .env file.")
        return None
def extract_usage_metrics(api_response: dict) -> int:
    """
    Extracts total tokens from a raw LLM API response.
    """
    return api_response["usage"]["total_tokens"]
# Create a single, pre-loaded instance
llm = get_llm()

if __name__ == '__main__':
    print("Testing LLM generation...")
    if llm:
        response = llm.invoke("What is the capital of France?")
        print(f"Test response: {response.content}")
    else:
        print("LLM not available. Skipping test.")