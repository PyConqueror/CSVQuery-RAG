import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from backend.rag_system import get_rag_system
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the RAG system
rag_system = get_rag_system()

# Example conversation pairs showing memory capability
EXAMPLES = [
    ["How make fields in the dataset?"],
    ["Tell me about the dataset"],
    ["Which month had the highest Ford Mustang sales?"],
    ["How many cars were sold between January and June 2020?"],
    ["How much revenue did BMW generate in Q3 2020?"],
    ["Did Ford or BMW sell more cars in December 2022?"],
    ["Compare the sales of Toyota and Honda in 2020"],
    ["Which brands sold more than 1000 cars in 2020?"],
    ["How many Volkswagen models are there?"],
]

def format_history(history):
    """Format the chat history into a format suitable for the RAG system"""
    formatted_history = []
    for human, ai in history:
        if human and ai:  # Only add complete exchanges
            formatted_history.extend([
                {"type": "human", "content": human},
                {"type": "assistant", "content": ai}
            ])
    return formatted_history

def process_chat(message, history):
    if not message.strip():
        return "Please enter a query."
    
    # Add typing indicator
    yield "Thinking... 🤔"
    
    try:
        # Format the chat history for the RAG system
        formatted_history = format_history(history)
        
        # Get response from RAG system with chat history context
        response = rag_system.query({
            "question": message,
            "chat_history": formatted_history
        })
        
        # Return final response
        yield response
    except Exception as e:
        yield f"An error occurred: {str(e)}"

def create_demo():
    """Create the Gradio interface with proper event handling"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        chatbot = gr.ChatInterface(
            fn=process_chat,
            examples=EXAMPLES,
            title="CSVQuery-RAG Assistant",
            description="Ask questions about your CSV data using natural language. I'll help you analyze and understand your data! You can refer to previous questions in your follow-up queries.",
            retry_btn=None,
            undo_btn=None,
            clear_btn="Clear Chat & Reset Memory",
        )

        # Add clear event handler
        chatbot.clear_btn.click(
            fn=lambda: rag_system.query({
                "question": "__reset_memory__",
                "chat_history": []
            }),
            inputs=None,
            outputs=None,
        )

    return demo

# Create the demo interface
demo = create_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 