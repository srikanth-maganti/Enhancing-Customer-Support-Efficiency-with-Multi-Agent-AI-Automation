__import__("pysqlite3")
import sys
sys.modules["sqlite3"]=sys.modules.pop("pysqlite3")
import os
from huggingface_hub import login

login(os.environ["HUGGINGFACEHUB_API_TOKEN"])

import streamlit as st
import torch
torch.classes.__path__ = []

from agents.summerization_agent import summarizer
from agents.action_extraction_agent import action_extracter
from agents.resolution_recommendation_agent import generate_resolution
import agents.task_routing_agent as task_routing_agent
import random
from gen_model import run
ticket = random.randint(1, 1000)


def detect_intent(message: str, model='llama3.2'):
    prompt = f"""
    You are a message intent classifier for a customer support system.

    Your task is to classify the user's message into one of the following categories:

    1. casual_chat – for greetings, general talk, or unrelated questions.
    2. incomplete_issue – if the user mentions having an issue or problem but doesn't clearly describe what the issue is.
    3. issue_report – only if the user clearly describes a specific problem, including what went wrong, what they were trying to do, and any error messages or behaviors observed.

    Respond with only one of the labels: casual_chat, incomplete_issue, or issue_report.

    Examples:
    - "Hi there!" → casual_chat
    - "I'm facing an issue" → incomplete_issue
    - "My app crashes when I try to open it" → issue_report

    User message: "{message}"
    """

    # response = ollama.chat(
    #     model=model,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    response = run(prompt)

    intent = response.strip().lower()
    return intent

# Satisfaction classifier
def detect_satisfaction(message: str, model='llama3.2'):
    prompt = f"""
    You are a sentiment classifier for customer support resolution feedback.

    Your job is to classify user satisfaction after providing a solution.

    Respond only with one label:
    - satisfied
    - unsatisfied

    Examples:
    - "Yes, it's fixed now." → satisfied
    - "Thanks, that helped!" → satisfied
    - "No, it's still not working." → unsatisfied
    - "That didn’t help me." → unsatisfied

    User feedback: "{message}"
    """
    # response = ollama.chat(
    #     model=model,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    response = run(prompt)
    return response.strip().lower()

# 🔧 Page Config
st.set_page_config(page_title="Customer Support Agent", page_icon="💬", layout="wide")

# Custom Chat UI
st.markdown("""
    <style>
        .stChatMessage {
            padding: 12px 18px;
            border-radius: 12px;
            margin: 5px 0;
            max-width: 80%;
            font-size: 16px;
        }
        .user {
            background-color: #DCF8C6;
            align-self: flex-end;
            color: black;
        }
        .assistant {
            background-color: #EAEAEA;
            align-self: flex-start;
            color: black;
        }
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            padding: 8px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Customer Support Agent")

# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_feedback" not in st.session_state:
    st.session_state.awaiting_feedback = False
if "generated_resolution" not in st.session_state:
    st.session_state.generated_resolution = ""
if "generated_action" not in st.session_state:
    st.session_state.generated_action = ""
if "generated_ticket" not in st.session_state:
    st.session_state.generated_ticket = ticket

# Display chat history
for msg in st.session_state.messages:
    role_class = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(msg["role"]):
        st.markdown(f'<div class="stChatMessage {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user"):
        st.markdown(f'<div class="stChatMessage user">{user_input}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):

        # CASE 1: Awaiting feedback
        if st.session_state.awaiting_feedback:
            satisfaction = detect_satisfaction(user_input)

            if satisfaction == "satisfied":
                bot_reply = "Awesome! Let me know if you need anything else."
                st.session_state.awaiting_feedback = False
            else:
                st.session_state.generated_ticket=st.session_state.generated_ticket + 1
                team,est_time=task_routing_agent.handle_message(
                    st.session_state.generated_ticket,
                    st.session_state.generated_action,
                    st.session_state.generated_resolution
                )
                if est_time==-1:
                    bot_reply =  f"Thanks for your feedback. I've escalated this to the {team} team.your problem will be resolved as soon as possible."
                else:

                    bot_reply = f"Thanks for your feedback. I've escalated this to the {team} team.your problem will be resolved in {est_time} minutes."
                st.session_state.awaiting_feedback = False

        # CASE 2: Normal conversation
        else:
            intent = detect_intent(user_input)

            if intent == "issue_report":
                summary = summarizer.handle_message(st.session_state.messages)
                action = action_extracter.handle_message(summary["summary"])
                resolution = generate_resolution(action["actions"])
                bot_reply = resolution

                # Store resolution context for follow-up
                st.session_state.generated_resolution = resolution
                st.session_state.generated_action = action["actions"]
                st.session_state.generated_ticket = ticket
                st.session_state.awaiting_feedback = True

                # Ask for feedback
                st.session_state.messages.append({"role": "assistant", "content": resolution})
                with st.chat_message("assistant"):
                    st.markdown(f'<div class="stChatMessage assistant">{resolution}</div>', unsafe_allow_html=True)

                follow_up = "Was this resolution helpful? (Yes/No)"
                st.session_state.messages.append({"role": "assistant", "content": follow_up})
                with st.chat_message("assistant"):
                    st.markdown(f'<div class="stChatMessage assistant">{follow_up}</div>', unsafe_allow_html=True)
                bot_reply = None  # Already handled display

            elif intent == "incomplete_issue":
                bot_reply = "Could you please describe the issue in more detail?"

            elif intent == "casual_chat":
                # bot_reply = ollama.chat(
                #     model="llama3.2",
                #     messages=[{"role": "user", "content": user_input}]
                # )["message"]["content"].strip()
                bot_reply = run(user_input).strip()

            else:
                bot_reply = "I'm not sure I understood that. Could you please clarify?"

        if bot_reply:
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant"):
                st.markdown(f'<div class="stChatMessage assistant">{bot_reply}</div>', unsafe_allow_html=True)
