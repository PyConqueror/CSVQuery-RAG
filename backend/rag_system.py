import os
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import logging
from typing import Union

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

            # Split text into chunks
            logger.info("Splitting text into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=150,
                length_function=len,
            )
            texts = text_splitter.create_documents(documents)
            logger.info(f"Created {len(texts)} text chunks")

            # Create vector store
            logger.info("Creating vector store...")
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            logger.info("Vector store created successfully")

            # Initialize conversation chain
            logger.info("Initializing conversation chain...")
            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-4-turbo",
                request_timeout=30
            )
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer'
            )

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 100}
                ),
                memory=memory,
                return_source_documents=False,
                chain_type="stuff"
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
                
            # Add context about available columns
            contextualized_question = (
                f"Based on the CSV data with columns: {self.columns_info}\n"
                f"Please answer this question: {question}\n"
                "If the answer cannot be found in the data, say 'I cannot find this information in the provided data.'"
            )
            
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