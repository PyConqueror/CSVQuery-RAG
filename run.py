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
    print("\033[34m\033[4mhttp://localhost:7860\033[0m")
    demo.launch() 