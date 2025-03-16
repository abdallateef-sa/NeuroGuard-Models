# __import__("pysqlite3")
# import sys

# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import PyPDF2
from langchain_together import ChatTogether
from langchain_core.runnables.history import RunnableWithMessageHistory


msgs = StreamlitChatMessageHistory(key="special_app_key")
llm = ChatTogether(model="meta-llama/Llama-3.3-70B-Instruct-Turbo",temperature=0.0, api_key="a0150ce38821b43f8333015f810d8bc98060c62c33f54494fd30e82dc2bc6c7f")



def read_db(filepath: str, embeddings_name):
    embeddings = HuggingFaceBgeEmbeddings(model_name=embeddings_name)
    vectordb = Chroma(persist_directory=filepath, embedding_function=embeddings)
    retreiver = vectordb.as_retriever()
    return retreiver


def read_system_prompt(filepath: str):

    with open(filepath, "r") as file:
        prompt_content = file.read()

    context = "{context}"

    system_prompt = f'("""\n{prompt_content.strip()}\n"""\n"{context}")'

    return system_prompt


def create_conversational_rag_chain(sys_prompt_dir, vdb_dir, llm, embeddings_name):
    retriever = read_db(vdb_dir, embeddings_name)

    contextualize_q_system_prompt = """Given a chat history and the latest user question
    which might reference context in the chat history,
    formulate a response which can be understood and clear
    without the chat history. Do NOT answer the question,
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    sys_prompt = read_system_prompt(sys_prompt_dir)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sys_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def extract_pdf_text(file_object):
    reader = PyPDF2.PdfReader(file_object)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return str(text)



def analysis_text(text: str):

    detailed_prompt = f"""
    You are a medical report analysis assistant. Your task is to:
    
    1. Carefully review the medical report text
    2. Identify key medical values and their significance
    3. Provide a clear, concise interpretation in a user-friendly format
    4. Follow this output structure strictly:
    
    Medical Value Interpretation Guide:
    - If a value is outside normal range, indicate:
      * The specific value
      * Whether it's high or low
      * Potential implications (in simple language)
    - Use clear, non-technical language
    - Avoid medical jargon
    - Provide actionable insights
    
    Example Output Format:
    ```
    Glucose Level: 228.69 mg/dL
    ⚠️ Status: High
    Interpretation: Your glucose level is elevated, which may indicate:
    - Potential pre-diabetic condition
    - Need for dietary adjustments
    - Recommend consulting your healthcare provider

    Recommendation: Schedule a follow-up blood test
    ```

    Report Text:
    {text}

    Your Analysis:
    """
    return detailed_prompt

def bot_func(rag_chain, user_input, session_id):
    for chunk in rag_chain.stream(
        {"input": user_input}, config={"configurable": {"session_id": session_id}}
    ):
        if answer_chunk := chunk.get("answer"):
            yield answer_chunk

def create_bot_for_selected_bot(name, embeddings, vdb_dir, sys_prompt_dir):
    """Create a bot for the selected configuration."""
    rag_chain = create_conversational_rag_chain(
        sys_prompt_dir, vdb_dir, llm, embeddings
    )
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,  
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
        max_tokens_limit=500,
        top_n=5
    )
    return conversational_rag_chain

