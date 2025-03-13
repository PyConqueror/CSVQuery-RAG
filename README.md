# CSVQuery-RAG

A Retrieval Augmented Generation (RAG) system that allows you to query CSV data using natural language. Built with LangChain and Gradio.

## Project Structure

```
.
├── backend/
│   ├── db/
│   │   └── MOCK_DATA.csv
│   └── rag_system.py
├── frontend/
│   └── app.py
├── requirements.txt
├── .env.example
└── run.py
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` and add your OpenAI API key.

## Running the Application

Simply run:
```bash
python run.py
```

The application will be available at http://localhost:7860

## Features

- Load and process CSV data for question answering
- Natural language querying of CSV data
- Conversation memory to maintain context
- Simple and intuitive Gradio interface
- Smart query validation to ensure responses stay within data context
- Comprehensive system prompting for accurate and relevant answers

## Query Guidelines

The system is designed to:
- Answer questions specifically about the data in the CSV file
- Provide statistical insights when appropriate
- Maintain data privacy and security
- Reject questions that are outside the scope of the data

Example valid queries:
- "What are the most common values in column X?"
- "Show me the distribution of values in the data"
- "Find records where column Y matches specific criteria"

The system will politely decline to answer questions about:
- Current events or real-time information
- Personal opinions or advice
- Topics not present in the CSV data
- Code generation or programming tasks

## System Requirements

- Python 3.8 or higher
- OpenAI API key
- CSV data file (included in backend/db/MOCK_DATA.csv) 