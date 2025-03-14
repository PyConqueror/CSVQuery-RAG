import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from backend.rag_system import get_rag_system

# Initialize the RAG system
rag_system = get_rag_system()

# Example prompts for the interface
EXAMPLE_PROMPTS = [
    "List all the columns available in the dataset.?",
    "Tell me about the dataset",
    "Find the top 5 entries with the highest total revenue value",
    "Which brand has the lowest revenue in 2021?",
    "Tell me the total sales of civic in 2020",
    "Which month had the highest Ford Mustang sales?",
    "How much revenue did BMW generate in Q3 2020?",
    "Compare the sales of Toyota and Honda in 2020",
    "Which brands sold more than 1000 cars in 2020?",
    "How many Volkswagen models are there?"
]

def process_query(message, history):
    if not message.strip():
        return history, ""
        
    # Convert chat history to format expected by Langchain
    chat_history = []
    for human, ai in history:
        chat_history.append({"question": human, "answer": ai})
        
    # Query with chat history context
    response = rag_system.query({
        "question": message,
        "chat_history": chat_history
    })
    
    history = history + [(message, response)]
    return history, ""

def clear_chat():
    # Reset the conversation memory in the backend
    rag_system.reset_memory()
    return []

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # CSVQuery-RAG
        Ask questions about your CSV data using natural language.
        """
    )
    
    chatbot = gr.Chatbot(
        show_label=False,
        container=True,
        height=600,
        elem_classes="chatbot",
        scroll_to_output=True
    )
    
    with gr.Row():
        with gr.Column(scale=8):
            msg = gr.Textbox(
                show_label=False,
                placeholder="Enter your query about the CSV data...",
                container=False
            )
        with gr.Column(scale=1):
            submit = gr.Button("Submit", variant="primary")
    
    with gr.Column(scale=1):
        clear = gr.Button("Clear Chat", size="sm")
        
    gr.Examples(
        examples=EXAMPLE_PROMPTS,
        inputs=msg,
        label="Example Queries",
    )
    
    # Set up event handlers
    submit_click = submit.click(
        process_query,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        process_query,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    clear.click(
        clear_chat,
        outputs=chatbot
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 