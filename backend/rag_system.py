import os
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import logging
from typing import Union
from backend.system_prompts import get_combined_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

# Define the base directory for data
DATA_DIR = os.path.join(os.path.dirname(__file__), "db")

class RAGSystem:
    def __init__(self, csv_path):
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found at: {csv_path}")
                
            self.csv_path = csv_path
            logger.info(f"Initializing RAG system with CSV file: {csv_path}")
            
            # Initialize OpenAI embeddings with explicit error handling
            try:
                self.embeddings = OpenAIEmbeddings()
                logger.info("Successfully initialized OpenAI embeddings")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {str(e)}")
                raise
                
            self.initialize_system()
        except Exception as e:
            logger.error(f"Error in RAG system initialization: {str(e)}")
            raise

    def initialize_system(self):
        try:
            # Read and process CSV data
            logger.info("Reading CSV file...")
            self.df = pd.read_csv(self.csv_path)
            
            # Get column information for context
            self.columns_info = ", ".join(self.df.columns)
            logger.info(f"CSV columns: {self.columns_info}")
            
            # Convert DataFrame to text documents
            logger.info("Converting data to documents...")
            documents = []
            for idx, row in self.df.iterrows():
                doc = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
                documents.append(doc)

            # Enhanced text splitting with optimized parameters
            logger.info("Splitting text into chunks with optimized parameters...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,  # Smaller chunks for more precise retrieval
                chunk_overlap=50,  # Reduced overlap to minimize redundancy
                length_function=len,
                separators=[",", "\n", " "],  # Custom separators for CSV data
                keep_separator=True
            )
            texts = text_splitter.create_documents(documents)
            logger.info(f"Created {len(texts)} text chunks")

            # Enhanced vector store with metadata
            logger.info("Creating vector store with enhanced configuration...")
            # Add metadata to each document
            texts_with_metadata = []
            for idx, text in enumerate(texts):
                metadata = {
                    'chunk_id': idx,
                    'source': 'csv_data',
                    'columns': self.columns_info
                }
                texts_with_metadata.append({
                    'page_content': text,
                    'metadata': metadata
                })

            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                collection_metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,  # Increased accuracy during index construction
                    "hnsw:search_ef": 100  # Increased accuracy during search
                }
            )
            logger.info("Vector store created successfully")

            # Initialize conversation chain with system prompt
            logger.info("Initializing conversation chain...")
            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-4-turbo",
                request_timeout=60
            )
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer'
            )
            
            # Create a prompt template with system message
            system_prompt = get_combined_prompt(columns_info=self.columns_info)
            logger.info("System prompt configured for RAG")
            
            # Create a proper prompt template that includes both context and question
            # Note: StuffDocumentsChain requires a prompt template with 'context' variable
            prompt_template = f"{system_prompt}\n\nContext: {{context}}\n\nQuestion: {{question}}"
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            logger.info("Created prompt template with context and question variables")
            
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="mmr",  # use MMR for better diversity in results
                    search_kwargs={
                        "k": 10,  # Reduced k for more focused results
                        "fetch_k": 30,  # Fetch more candidates for MMR
                        "lambda_mult": 0.7  # Balance between relevance and diversity
                    }
                ),
                memory=memory,
                return_source_documents=False,
                chain_type="stuff",
                combine_docs_chain_kwargs={"prompt": prompt}
            )
            logger.info("Conversation chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error in system initialization: {str(e)}")
            raise

    def is_valid_query(self, question: str) -> bool:
        """
        Validate if the query is appropriate for the CSV data context.
        Returns False for obviously out-of-scope questions.
        """
        try:
            if not question or not isinstance(question, str):
                return False
                
            # List of keywords that might indicate out-of-scope questions
            out_of_scope_keywords = [
                "weather", "news", "current events", "stock market",
                "politics", "sports", "create", "make",
                "write code", "programming", "what is your name",
                "who are you", "tell me about yourself"
            ]
            
            question_lower = question.lower()
            
            # Check for out-of-scope keywords
            if any(keyword in question_lower for keyword in out_of_scope_keywords):
                logger.info(f"Query contains out-of-scope keywords: {question}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error in query validation: {str(e)}")
            return False

    def reset_memory(self):
        """Reset the conversation memory to start a fresh conversation"""
        try:
            self.qa_chain.memory.clear()
            logger.info("Conversation memory cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing conversation memory: {str(e)}")
            raise

    def query(self, input_data: Union[str, dict]) -> str:
        try:
            # Handle both string and dict input for backward compatibility
            if isinstance(input_data, str):
                question = input_data
                chat_history = []
            else:
                question = input_data['question']
                chat_history = input_data.get('chat_history', [])
                # Check if this is a reset request
                if question == "__reset_memory__":
                    self.reset_memory()
                    return "Conversation memory has been reset."
                
            # Input validation
            if not question or not isinstance(question, str):
                return "Please provide a valid question string."
                
            # First validate the query
            if not self.is_valid_query(question):
                return "I can only answer questions about the data in the CSV file. This question appears to be outside the scope of the available data."
                
            # We don't need to manually add column context since it's in the system prompt now
            # Just pass the question directly to use our configured system prompt
            contextualized_question = question
            
            logger.info(f"Processing query: {question}")
            
            # Process valid query with timeout handling
            try:
                response = self.qa_chain.invoke({"question": contextualized_question})
                logger.info("Query processed successfully")
                return response['answer']
            except Exception as e:
                if "timeout" in str(e).lower():
                    logger.error(f"Query timeout: {str(e)}")
                    return "The request timed out. Please try again with a simpler question."
                logger.error(f"Error processing query: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"

# Initialize the RAG system
def get_rag_system():
    try:
        csv_path = os.path.join(DATA_DIR, "MOCK_DATA.csv")
        logger.info(f"Attempting to initialize RAG system with CSV file: {csv_path}")
        return RAGSystem(csv_path)
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        raise Exception(f"Failed to initialize RAG system: {str(e)}") 