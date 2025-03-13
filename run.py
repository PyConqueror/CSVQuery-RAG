import os
import sys
from frontend.app import demo

if __name__ == "__main__":
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("Warning: .env file not found. Please create one from .env.example")
        print("Copy .env.example to .env and add your OpenAI API key")
        sys.exit(1)
        
    # Launch the Gradio interface
    print("Starting CSVQuery-RAG system...")
    print("The interface will be available at http://localhost:7860")
    demo.launch() 