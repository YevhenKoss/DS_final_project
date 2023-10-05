import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# from langchain.llms import HuggingFaceHub
from translate import Translator
from htmlTemplates import css, bot_template, user_template
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback


size = 50000000
acc = 0  # Сюди треба передати параметри користувача. Якщо прєм. то =1, якщо базовий то =0.


def get_pdf_text(pdf_docs, selected_file_index):
    if selected_file_index >= 0 and selected_file_index < len(pdf_docs):
        pdf_reader = PdfReader(pdf_docs[selected_file_index])
        text = " ".join(page.extract_text() for page in pdf_reader.pages)
        return text
    else:
        return ""


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 1024})
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )


def handle_user_input(user_question, conversation_chain, chat_history):
    chat_history.clear()
    with get_openai_callback() as cb:
        response = conversation_chain({"question": user_question})
        chat_history.extend(reversed(response["chat_history"]))
        st.write(cb)
        for i, message in enumerate(chat_history):
            template = user_template if i % 2 != 0 else bot_template
            st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)




def clear_chat_history(chat_history):
    chat_history.clear()


def save_chat_history(chat_history, filename="chat_history.txt"):
    chat_history.clear()
    st.success("Chat history saved successfully")


def translate_text(text, target_language):
    translator = Translator(to_lang=target_language)
    translated_text = translator.translate(text)
    return translated_text


def check_file_size(pdf_docs):
    for pdf in pdf_docs:
        file_size = pdf.getbuffer().nbytes
        if not acc:
            if file_size >= size:
                st.warning("Yoy have to upload file less than 50 MB")
                pdf_docs = None
    return pdf_docs


def main():
    load_dotenv()
    st.set_page_config(page_title="LLMExplorer", page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("LLMExplorer :robot_face:")

    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        if st.session_state.conversation is None:
            st.warning("Please process your PDF documents first.")
        else:
            handle_user_input(
                user_question,
                st.session_state.conversation,
                st.session_state.chat_history,
            )

    with st.sidebar:

        st.subheader("Your documents")

        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        pdf_docs = check_file_size(pdf_docs)

        if pdf_docs:
            selected_file = st.selectbox(
                "Select a PDF file:", [pdf.name for pdf in pdf_docs]
            )
            selected_file_index = [pdf.name for pdf in pdf_docs].index(selected_file)
        else:
            selected_file_index = -1

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs, selected_file_index)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
            else:
                st.warning("Please upload PDF documents before processing.")

        if st.button("Clear Chat History"):
            clear_chat_history(st.session_state.chat_history)
            st.success("Chat history cleared successfully.")

        if st.button("Save Chat History"):
            if st.session_state.chat_history:
                save_chat_history(st.session_state.chat_history)

        st.subheader("Translation")
        text_to_translate = st.text_area("Enter text to translate:")
        target_language = st.selectbox(
            "Select target language:", ["en", "fr", "es", "de", "ru", "uk"]
        )
        if st.button("Translate"):
            if text_to_translate:
                translated_text = translate_text(text_to_translate, target_language)
                st.write(f"Translated Text ({target_language}): {translated_text}")


if __name__ == "__main__":
    main()