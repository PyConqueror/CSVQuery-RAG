import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from backend.rag_system import get_rag_system

# Initialize the RAG system
rag_system = get_rag_system()

def process_query(query):
    if not query.strip():
        return "Please enter a query."
    response = rag_system.query(query)
    return response

# Create the Gradio interface
demo = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Enter your query about the CSV data...",
        label="Query"
    ),
    outputs=gr.Textbox(
        lines=5,
        label="Response"
    ),
    title="CSVQuery-RAG",
    description="Ask questions about your CSV data using natural language.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 