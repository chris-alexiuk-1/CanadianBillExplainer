from langchain.chat_models import ChatAnthropic
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import streamlit as st
from dotenv import load_dotenv
import PyPDF2

load_dotenv()


class LegalExpert:
    def __init__(self):
        self.system_prompt = self.get_system_prompt()

        self.user_prompt = HumanMessagePromptTemplate.from_template("{legal_question}")

        full_prompt_template = ChatPromptTemplate.from_messages(
            [self.system_prompt, self.user_prompt]
        )

        self.chat = ChatAnthropic()

        self.chain = LLMChain(llm=self.chat, prompt=full_prompt_template)

    def get_system_prompt(self):
        system_prompt = """
        You are a Canadian Legal Expert. 

        Under no circumstances do you give legal advice.
        
        You are adept at explaining the law in laymans terms, and you are able to provide context to legal questions.

        While you can add context outside of the provided context, please do not add any information that is not directly relevant to the question, or the provided context.

        You speak {language}.

        ### CONTEXT
        {context}

        ### END OF CONTEXT
        """

        return SystemMessagePromptTemplate.from_template(system_prompt)

    def run_chain(self, language, context, question):
        return self.chain.run(
            language=language, context=context, legal_question=question
        )


def retrieve_pdf_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# create a streamlit app
st.title("Canadian Legal Explainer (that does not give advice)")

if "LegalExpert" not in st.session_state:
    st.session_state.LegalExpert = LegalExpert()

# create a upload file widget for a pdf
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if a pdf file is uploaded
if pdf_file:
    # retrieve the text from the pdf
    if "context" not in st.session_state:
        st.session_state.context = retrieve_pdf_text(pdf_file)

# create a button that clears the context
if st.button("Clear context"):
    st.session_state.__delitem__("context")
    st.session_state.__delitem__("legal_response")

# if there's context, proceed
if "context" in st.session_state:
    # create a dropdown widget for the language
    language = st.selectbox("Language", ["English", "Français"])
    # create a text input widget for a question
    question = st.text_input("Ask a question")

    # create a button to run the model
    if st.button("Run"):
        # run the model
        legal_response = st.session_state.LegalExpert.run_chain(
            language=language, context=st.session_state.context, question=question
        )

        if "legal_response" not in st.session_state:
            st.session_state.legal_response = legal_response

        else:
            st.session_state.legal_response = legal_response

# display the response
if "legal_response" in st.session_state:
    st.write(st.session_state.legal_response)
