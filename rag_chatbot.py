from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_mistralai import ChatMistralAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

def continuous_chat(vectorstore):
    load_dotenv()
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable is not set.")
    
    llm = ChatMistralAI(mistral_api_key=api_key, model="mistral-large-latest")

    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

    template = """
    You are a helpful assistant who answers faq related to 
    • Specifications of the latest smartphones or laptops.
    • How customers can track their orders.
    • Details about the company's return policy.
    • Available payment methods for online purchases.
    • Warranty information for products.
    If you don't know the answer or the question is outside the provided context, politely say so
    and suggest they rephrase or ask something related to the documentation.
    
    Context: {context}
    
    Current conversation:
    {chat_history}
    
    Human: {question}
    
    Assistant:
    """
    prompt = PromptTemplate.from_template(template)
   
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True
    )

    print("Welcome to FAQ Chat Assistant! Type 'exit' to end the conversation.")
    while True:
        query = input("\nYour question: ")
        print("==" * 50)
        
        if query.lower() in ["exit", "quit", "bye"]:
            print("Thank you for using FAQ Chat Assistant. Goodbye!")
            break
        try:
            result = qa_chain({"question": query})
            print("\nAssistant:", result["answer"])
            print("==" * 50)
        except Exception as e:
            print(f"Error: {e}")
            print("Sorry, I encountered an error. Please try again.")
        
def load_vectorstore():
    load_dotenv()
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable is not set.")
    vectorstore = FAISS.load_local("faiss_index", MistralAIEmbeddings(mistral_api_key=api_key,model="mistral-embed"),allow_dangerous_deserialization=True)
    return vectorstore

if __name__ == "__main__":
    vectorstore = load_vectorstore()
    continuous_chat(vectorstore)