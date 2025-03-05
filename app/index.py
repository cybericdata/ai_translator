import streamlit as st
from sidebar import display_sidebar
from chat import display_chat_interface

st.title("HIV & AIDS Educational Assistant - Generative AI")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

display_sidebar()
display_chat_interface()