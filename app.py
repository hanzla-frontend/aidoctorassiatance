
import os
import json
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------
# 1. CONFIGURATION
# ---------------------------------

load_dotenv()

st.set_page_config(
    page_title="AI Doctor Assistant",
    page_icon="🩺",
    layout="centered"
)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Please add GROQ_API_KEY to your .env file.")
    st.stop()

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-120b"


# ---------------------------------
# 2. AGENT TOOLS
# ---------------------------------

def health_education(topic):
    """Provide general educational information."""

    return {
        "topic": topic,
        "information": (
            f"Provide general educational information about {topic}. "
            "Explain its meaning, common causes, general symptoms, "
            "prevention and when professional medical advice is needed. "
            "Do not diagnose or prescribe medication."
        )
    }


def check_emergency_signs(symptoms):
    """Check for common emergency warning signs."""

    emergency_keywords = [
        "chest pain",
        "difficulty breathing",
        "can't breathe",
        "cannot breathe",
        "severe bleeding",
        "unconscious",
        "stroke",
        "seizure",
        "suicidal",
        "overdose",
        "severe allergic reaction"
    ]

    symptoms_lower = symptoms.lower()

    detected = [
        word for word in emergency_keywords
        if word in symptoms_lower
    ]

    return {
        "possible_warning_signs": detected,
        "urgent": bool(detected),
        "message": (
            "Potential emergency warning signs were detected. "
            "Seek emergency medical assistance immediately."
            if detected else
            "No listed warning phrase was detected. "
            "This does not rule out a medical emergency."
        )
    }


def ask_followup(question):
    """Help collect relevant information."""

    return {
        "suggested_question": (
            f"To better understand the concern, ask the user: {question}"
        )
    }


def create_doctor_summary(details):
    """Create a structured summary for a doctor visit."""

    return {
        "doctor_visit_summary": details,
        "instruction": (
            "Present this as a patient-provided summary, "
            "not a medical diagnosis."
        )
    }


# ---------------------------------
# 3. TOOL DEFINITIONS
# ---------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "health_education",
            "description": "Explain a general health topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_emergency_signs",
            "description": (
                "Check user-reported symptoms for common "
                "emergency warning signs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "string"
                    }
                },
                "required": ["symptoms"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_followup",
            "description": (
                "Suggest a relevant follow-up question "
                "to understand the user's concern."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string"
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_doctor_summary",
            "description": (
                "Prepare a summary of user-provided information "
                "for a healthcare professional."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "details": {
                        "type": "string"
                    }
                },
                "required": ["details"]
            }
        }
    }
]


# ---------------------------------
# 4. TOOL EXECUTOR
# ---------------------------------

def execute_tool(name, arguments):

    tool_functions = {
        "health_education": health_education,
        "check_emergency_signs": check_emergency_signs,
        "ask_followup": ask_followup,
        "create_doctor_summary": create_doctor_summary
    }

    if name not in tool_functions:
        return {"error": "Unknown tool"}

    return tool_functions[name](**arguments)


# ---------------------------------
# 5. AGENT LOOP
# ---------------------------------

SYSTEM_PROMPT = """
You are an AI Doctor Assistant.

Your purpose is health education, not medical diagnosis.

RULES:
1. Answer in simple, understandable English.
2. Explain general health topics.
3. Never claim to be a real doctor.
4. Never diagnose a medical condition.
5. Never prescribe medicines or dosages.
6. Do not tell users to stop prescribed medication.
7. Ask relevant follow-up questions when necessary.
8. Use available tools when useful.
9. Explain when professional medical care is appropriate.
10. If emergency symptoms are reported, advise immediate
    emergency medical assistance. Do not delay this advice
    while asking follow-up questions.
11. A negative emergency tool result does not mean the
    user is medically safe.
12. Protect user privacy and do not request unnecessary
    personal information.

This chatbot provides general health education only.
"""


def run_agent(user_message, chat_history):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    messages.extend(chat_history)

    messages.append({
        "role": "user",
        "content": user_message
    })

    # Agent can call tools in multiple steps
    for step in range(5):

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3
        )

        assistant_message = response.choices[0].message

        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        # No more tool calls: return final answer
        if not assistant_message.tool_calls:
            return assistant_message.content or (
                "I could not generate a response. Please try again."
            )

        # Execute selected tools
        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            result = execute_tool(
                tool_name,
                arguments
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(result)
            })

    return "Please try asking your question again."


# ---------------------------------
# 6. STREAMLIT USER INTERFACE
# ---------------------------------

st.title("🩺 AI Doctor Assistant")

st.markdown(
    """
    <p style="color:#64748b; font-size:16px;">
        Your AI-powered health education assistant.
    </p>
    <p style="color:#059669; font-size:13px;">
        Ask health questions • Learn medical terms •
        Explore healthy lifestyles
    </p>
    """,
    unsafe_allow_html=True
)

st.warning(
    "This assistant provides general health education only. "
    "It is not a doctor and cannot diagnose or treat diseases. "
    "For emergencies, contact local emergency services."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:

    st.header("About Assistant")

    st.write(
        "This chatbot uses Agentic AI and Python tools "
        "to provide health education."
    )

    st.subheader("Available Tools")

    st.write("📚 Health Education")
    st.write("🚨 Emergency Warning Signs")
    st.write("❓ Follow-up Questions")
    st.write("📝 Doctor Visit Summary")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
user_input = st.chat_input(
    "Ask your health-related question..."
)

if user_input:

    # Display user question
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("AI Agent is thinking..."):

            try:

                # Previous conversation
                history = st.session_state.messages.copy()

                answer = run_agent(
                    user_input,
                    history
                )

                st.markdown(answer)

                st.session_state.messages.extend([
                    {
                        "role": "user",
                        "content": user_input
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ])

            except Exception as error:

                st.error(
                    "Something went wrong. "
                    "Please check your API key and connection."
                )

                # Technical details for development
                st.caption(str(error))
