# CSVQuery-RAG Technical Documentation

## System Architecture

CSVQuery-RAG is built on a modular architecture consisting of two main components:
1. Backend RAG System (`backend/rag_system.py`)
2. Frontend Gradio Interface (`frontend/app.py`)

## Technical Stack

- **Language**: Python 3.8+
- **RAG Implementation**: LangChain
- **Vector Store**: Chroma
- **LLM**: OpenAI GPT (gpt-3.5-turbo)
- **UI Framework**: Gradio
- **Embeddings**: OpenAI Embeddings

## Component Details

### 1. RAG System (`backend/rag_system.py`)

#### Class: `RAGSystem`

The core class that handles the Retrieval Augmented Generation functionality.

##### Initialization
```python
def __init__(self, csv_path: str)
```
- **Purpose**: Initializes the RAG system with a CSV file
- **Parameters**: 
  - `csv_path`: Path to the CSV file to be processed
- **Process**:
  1. Validates CSV file existence
  2. Initializes OpenAI embeddings with error handling
  3. Calls `initialize_system()`

##### Method: `initialize_system()`
```python
def initialize_system(self)
```
- **Purpose**: Sets up the RAG pipeline
- **Components**:
  1. CSV Data Processing
  2. Text Chunking
  3. Vector Store Creation
  4. Conversation Chain Setup
- **Key Features**:
  - Uses RecursiveCharacterTextSplitter (chunk_size=500, overlap=150)
  - Implements ConversationalRetrievalChain with memory
  - Configures ChatOpenAI with temperature=0 for consistent responses
  - Uses Chroma vector store with cosine similarity

##### Method: `is_valid_query()`
```python
def is_valid_query(self, question: str) -> bool
```
- **Purpose**: Validates if a query is appropriate for the dataset
- **Parameters**:
  - `question`: The user's input query
- **Returns**: Boolean indicating query validity
- **Validation Checks**:
  - Non-empty string
  - Not out-of-scope topics (using keyword filtering)
  - Relevant to CSV data context

##### Method: `reset_memory()`
```python
def reset_memory(self)
```
- **Purpose**: Clears the conversation memory
- **Usage**: Called when starting a new conversation
- **Process**: Clears the ConversationBufferMemory

##### Method: `query()`
```python
def query(self, input_data: Union[str, dict]) -> str
```
- **Purpose**: Processes user queries and returns responses
- **Parameters**:
  - `input_data`: Either a string question or a dict with question and chat history
- **Returns**: Generated response string
- **Features**:
  - Supports memory reset with special command
  - Adds column context to queries
  - Handles timeouts and errors gracefully
  - Validates input before processing

### 2. Frontend Interface (`frontend/app.py`)

#### Function: `process_query()`
```python
def process_query(message: str, history: list) -> tuple
```
- **Purpose**: Handles user input from Gradio interface
- **Parameters**:
  - `message`: Current user message
  - `history`: Chat history
- **Returns**: Updated history and empty message
- **Features**:
  - Converts chat history to LangChain format
  - Maintains conversation context
  - Handles empty messages

#### Function: `clear_chat()`
```python
def clear_chat() -> list
```
- **Purpose**: Resets the chat interface and backend memory
- **Returns**: Empty list for chat history
- **Process**: Calls backend reset_memory()

#### Gradio Interface Configuration

##### Block-based Interface
```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:
```
- **Components**:
  - Markdown header
  - Chatbot interface (600px height)
  - Query input textbox
  - Submit and Clear buttons
  - Example queries section
- **Features**:
  - Responsive layout
  - Example queries for user guidance
  - Clear chat functionality
  - Modern Soft theme

## Data Processing Pipeline

1. **Data Loading**
   - CSV file reading using pandas
   - Column information extraction
   - Data validation and error handling

2. **Text Processing**
   - Document creation from CSV rows
   - Text chunking (500 chars, 150 overlap)
   - Metadata preservation

3. **Vector Store**
   - Document embedding using OpenAI
   - Chroma vector store with cosine similarity
   - Efficient retrieval (k=100)

4. **Query Processing**
   - Query validation with keyword filtering
   - Context addition with column information
   - Response generation with timeout handling
   - Memory management

## System Requirements

### Software Dependencies
- Python 3.8+
- OpenAI API access
- Required Python packages (see requirements.txt):
  - langchain
  - langchain_openai
  - langchain_community
  - gradio
  - pandas
  - chromadb
  - python-dotenv

### Environment Configuration
- `.env` file with OPENAI_API_KEY
- CSV data file in `backend/db/`
- Sufficient system memory for vector operations

## Performance Considerations

1. **Vector Store**
   - Optimized chunk size (500) and overlap (150)
   - Cosine similarity for better matching
   - Retrieves top 100 similar chunks

2. **Query Processing**
   - 30-second timeout for queries
   - Memory management for conversations
   - Efficient chat history handling

3. **UI Performance**
   - Responsive chat interface
   - Efficient message handling
   - Example queries for better UX

## Security Considerations

1. **API Key Management**
   - Environment variable validation
   - Explicit error handling
   - Secure key loading

2. **Input Validation**
   - Query sanitization
   - Out-of-scope detection
   - Error boundary implementation

3. **Data Privacy**
   - Local vector store
   - No external data storage
   - Secure conversation handling

## Error Handling

1. **System Initialization**
   - CSV file validation
   - API key verification
   - Comprehensive logging

2. **Query Processing**
   - Invalid query detection
   - Timeout management (30s)
   - Graceful error handling

3. **UI Error Handling**
   - Empty input handling
   - Clear chat functionality
   - Error message display

## Deployment

1. **Local Deployment**
```bash
python run.py
```
- Runs on 0.0.0.0:7860
- Accessible via web browser
- Development mode support

2. **Production Considerations**
   - Environment configuration
   - Comprehensive logging
   - Error handling
   - Memory management
   - Security measures

## Maintenance and Updates

1. **Code Updates**
   - Version control
   - Dependency management
   - API compatibility

2. **Data Updates**
   - CSV file updates
   - Vector store rebuilding
   - System reinitialization

3. **Performance Monitoring**
   - Query response times
   - Memory usage
   - Error rates

## Future Enhancements

1. **Potential Improvements**
   - Multiple file support
   - Advanced query capabilities
   - Enhanced error handling
   - UI customization options

2. **Scalability Options**
   - Distributed processing
   - Caching mechanisms
   - Load balancing

3. **Feature Additions**
   - Data visualization
   - Export capabilities
   - Advanced analytics 