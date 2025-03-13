import os
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

# Load environment variables
load_dotenv()

# Define the base directory for data
DATA_DIR = os.path.join(os.path.dirname(__file__), "db")

# System prompt template
SYSTEM_TEMPLATE = """You are a helpful assistant specifically designed to answer questions about the provided CSV data. Your role is to:

1. ONLY answer questions that are directly related to the data in the CSV file
2. If a question is not about the CSV data or requires external information, respond with: "I can only answer questions about the data in the CSV file."
3. Be precise and factual in your responses, using only the information available in the data
4. When appropriate, provide statistical insights about the data
5. If you're unsure if the data contains the information needed, respond with: "I cannot find this information in the provided data."

Remember:
- Do not make assumptions beyond the data
- Do not provide personal opinions
- Do not answer questions about topics outside the CSV data
- Always maintain data privacy and do not disclose sensitive information if present

Current Context: You are analyzing a CSV file containing {context}."""

class RAGSystem:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.embeddings = OpenAIEmbeddings()
        self.initialize_system()

    def initialize_system(self):
        # Read and process CSV data
        self.df = pd.read_csv(self.csv_path)
        
        # Get column information for context
        columns_info = ", ".join(self.df.columns)
        
        # Create system prompt with context
        system_prompt = SystemMessagePromptTemplate.from_template(
            SYSTEM_TEMPLATE.format(context=f"the following columns: {columns_info}")
        )
        
        # Create human message template
        human_prompt = HumanMessagePromptTemplate.from_template("{question}")
        
        # Create chat prompt
        chat_prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            human_prompt
        ])

        # Convert DataFrame to text documents
        documents = []
        for idx, row in self.df.iterrows():
            doc = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
            documents.append(doc)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.create_documents(documents)

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings
        )

        # Initialize conversation chain
        llm = ChatOpenAI(temperature=0)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": chat_prompt}
        )

    def is_valid_query(self, question: str) -> bool:
        """
        Validate if the query is appropriate for the CSV data context.
        Returns False for obviously out-of-scope questions.
        """
        # List of keywords that might indicate out-of-scope questions
        out_of_scope_keywords = [
            "weather", "news", "current events", "stock market",
            "politics", "sports", "create", "generate", "make",
            "write code", "programming", "what is your name",
            "who are you", "tell me about yourself"
        ]
        
        question_lower = question.lower()
        
        # Check for out-of-scope keywords
        if any(keyword in question_lower for keyword in out_of_scope_keywords):
            return False
            
        return True

    def query(self, question: str) -> str:
        try:
            # First validate the query
            if not self.is_valid_query(question):
                return "I can only answer questions about the data in the CSV file. This question appears to be outside the scope of the available data."
                
            # Process valid query
            response = self.qa_chain({"question": question})
            return response['answer']
        except Exception as e:
            return f"Error processing query: {str(e)}"

# Initialize the RAG system
def get_rag_system():
    csv_path = os.path.join(DATA_DIR, "MOCK_DATA.csv")
    return RAGSystem(csv_path) 