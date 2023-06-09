import streamlit as st
from streamlit_chat import message
from run import chat_bot

st.markdown("<h1 style='text-align: center;'>Mini GPT: Shakespeare Text Generation</h1>", unsafe_allow_html=True)


# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

st.sidebar.title("Sidebar")
outtokens = st.sidebar.slider("Number of output tokens: ", min_value = 0, max_value = 2000, value = 500,label_visibility="visible")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

def generate_response(prompt, outtokens):

    message = chat_bot(prompt, outtokens)
    
    return message 


# container for chat history
response_container = st.container()
# container for text box
container = st.container()


with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = generate_response(user_input, outtokens)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))



#trigger test2

# def get_text():
#     input_text = st.text_input("You: ","Enter in a short prompt", key="input")
#     return input_text

# user_input = get_text()

# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []

# if 'past' not in st.session_state:
#     st.session_state['past'] = []

# if user_input:
#     output = generate_response(user_input)
#     # store the output 
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(output)

# if st.session_state['generated']:
    
#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         message(st.session_state["generated"][i], key=str(i))
#         message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
