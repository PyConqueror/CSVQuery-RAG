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
  2. Initializes OpenAI embeddings
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
  - Uses RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
  - Implements ConversationalRetrievalChain for maintaining context
  - Configures ChatOpenAI with temperature=0 for consistent responses

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
  - Not out-of-scope topics
  - Relevant to CSV data context

##### Method: `query()`
```python
def query(self, question: str) -> str
```
- **Purpose**: Processes user queries and returns responses
- **Parameters**:
  - `question`: User's question about the data
- **Returns**: Generated response string
- **Process**:
  1. Input validation
  2. Query contextualization
  3. RAG chain execution
  4. Error handling

### 2. Frontend Interface (`frontend/app.py`)

#### Function: `process_query()`
```python
def process_query(message: str, history: list) -> str
```
- **Purpose**: Handles user input from Gradio interface
- **Parameters**:
  - `message`: Current user message
  - `history`: Chat history
- **Returns**: Generated response
- **Error Handling**: Graceful handling of empty inputs and exceptions

#### Gradio Interface Configuration

##### ChatInterface Setup
```python
demo = gr.ChatInterface(
    fn=process_query,
    title="CSVQuery-RAG",
    description="...",
    examples=[...],
    css=custom_css,
    theme=gr.themes.Soft(...)
)
```
- **Components**:
  - Custom chat processing function
  - Predefined example queries
  - Custom CSS styling
  - Soft color theme

##### Custom CSS Styling
```css
.message.user {
    /* User message styling */
}
.message.bot {
    /* Bot message styling */
}
```
- **Features**:
  - Responsive message bubbles
  - Different styles for user/bot messages
  - Custom border radius and padding
  - Shadow effects for depth

## Data Processing Pipeline

1. **Data Loading**
   - CSV file reading using pandas
   - Column information extraction
   - Data validation

2. **Text Processing**
   - Document creation from CSV rows
   - Text chunking for optimal retrieval
   - Metadata preservation

3. **Vector Store**
   - Document embedding using OpenAI
   - Chroma vector store implementation
   - Cosine similarity search

4. **Query Processing**
   - Query validation
   - Context addition
   - Response generation
   - Error handling

## System Requirements

### Software Dependencies
- Python 3.8+
- OpenAI API access
- Required Python packages:
  - langchain
  - gradio
  - pandas
  - chromadb
  - python-dotenv

### Environment Configuration
- `.env` file with OpenAI API key
- CSV data file in `backend/db/`
- Sufficient system memory for vector operations

## Performance Considerations

1. **Vector Store**
   - Optimized for similarity search
   - Configurable chunk size and overlap
   - Memory usage scales with data size

2. **Query Processing**
   - Timeout handling for long queries
   - Configurable response parameters
   - Context window management

3. **UI Performance**
   - Responsive chat interface
   - Efficient message handling
   - Browser compatibility

## Security Considerations

1. **API Key Management**
   - Environment variable usage
   - No hardcoded credentials
   - Secure key validation

2. **Input Validation**
   - Query sanitization
   - Error boundary implementation
   - Safe data handling

3. **Data Privacy**
   - Local vector store
   - Configurable data retention
   - No external data storage

## Error Handling

1. **System Initialization**
   - CSV file validation
   - API key verification
   - Environment setup checks

2. **Query Processing**
   - Invalid query detection
   - Timeout management
   - Exception handling

3. **UI Error Handling**
   - Input validation
   - Response formatting
   - Connection error handling

## Best Practices

1. **Code Organization**
   - Modular architecture
   - Clear separation of concerns
   - Consistent naming conventions

2. **Documentation**
   - Inline code comments
   - Function documentation
   - System architecture details

3. **Testing**
   - Input validation testing
   - Error handling verification
   - Response quality assurance

## Deployment

1. **Local Deployment**
```bash
python run.py
```
- Runs on localhost:7860
- Configurable host/port
- Development mode support

2. **Production Considerations**
   - Environment configuration
   - Error logging
   - Performance monitoring
   - Security hardening

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