import streamlit as st
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms.palm import PaLM
from llama_index import ServiceContext
from llama_index.memory import ChatMemoryBuffer
import os

st.set_page_config(page_title="Chat with KK_Tutor, powered by KK", page_icon="👨‍🏫", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title(f"**KK_Tutor** 👨‍🏫")
st.info("You can ask anything about unit 1!", icon="📃")
os.environ['GOOGLE_API_KEY'] = st.secrets.api_key

# Initialize chat history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about chemistry unit 1!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the resources – hang tight! This should take 1-2 minutes."):
        docs = SimpleDirectoryReader(input_dir="./data", recursive=True).load_data()
        llm = PaLM(model="text-curie-001", temperature=0.5)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, chunk_size=800, chunk_overlap=20)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine

    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    st.session_state.chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    context_prompt=(
        "You are a chemistry tutor able to have normal interactions, as well as talk. "
        "The user is a student who will have doubts about the context. "
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
        "\nInstruction: Refrain from giving single word answers."
        "\nInstruction: If the given query is not within the context, do not answer! Say Sorry, the given question is out of context!"
    ),
    verbose=False,
)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.markdown(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
