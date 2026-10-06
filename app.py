import os

# Disable telemetry / keep CrewAI storage in a writable location (set BEFORE importing crewai)
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_STORAGE_DIR", "/tmp/crewai")

import gradio as gr
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv

# Dummy function so ZeroGPU's startup check passes (the app doesn't use the GPU)
try:
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return None
except ImportError:
    pass  # running locally, not on HF

# ==============================
# Load API Key
# On Hugging Face: add GROQ_API_KEY under Settings -> Secrets
# Locally: put it in a .env file
# ==============================
load_dotenv()
MODEL_NAME = os.getenv("GROQ_MODEL", "groq/openai/gpt-oss-120b") 


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it in the Space: Settings → Variables and secrets → New secret."
        )
    return LLM(model=MODEL_NAME, temperature=0, api_key=api_key)


# ==============================
# Guardrail
# ==============================
TRAVEL_KEYWORDS = [
    "trip", "travel", "itinerary", "route", "budget",
    "visit", "tour", "plan", "days", "day",
]


def is_travel_query(user_input: str) -> bool:
    text = user_input.lower()
    return any(word in text for word in TRAVEL_KEYWORDS)


# ==============================
# Main travel planner
# ==============================
def run_travel_planner(user_input: str) -> str:
    try:
        if not is_travel_query(user_input):
            return "⚠️ I can only help with travel planning queries (trips, routes, itinerary, budget)."

        llm = get_llm()

        route_agent = Agent(
            role="Travel Route Finder",
            goal="Only provide travel routes",
            backstory="You ONLY answer travel-related queries. Reject others.",
            llm=llm,
        )
        cost_agent = Agent(
            role="Travel Cost Estimator",
            goal="Only estimate travel cost",
            backstory="You ONLY answer travel-related queries. Reject others.",
            llm=llm,
        )
        planner_agent = Agent(
            role="Travel Planner",
            goal="Only create travel itinerary",
            backstory="You ONLY answer travel-related queries. Reject others.",
            llm=llm,
        )

        route_task = Task(
            description=f"""
            User Query: {user_input}

            Instruction:
            - Answer ONLY if it is a travel-related query
            - Provide transport routes (bus/train/flight)
            - Do NOT answer unrelated questions
            """,
            expected_output="Travel routes only",
            agent=route_agent,
        )
        cost_task = Task(
            description=f"""
            User Query: {user_input}

            Instruction:
            - Provide cost estimation
            - Stay within travel context only
            """,
            expected_output="Cost breakdown only",
            agent=cost_agent,
        )
        planner_task = Task(
            description=f"""
            User Query: {user_input}

            Instruction:
            - Create day-wise itinerary
            - STRICTLY travel planning only
            - No extra explanations
            """,
            expected_output="Structured itinerary",
            agent=planner_agent,
        )

        crew = Crew(
            agents=[route_agent, cost_agent, planner_agent],
            tasks=[route_task, cost_task, planner_task],
            verbose=False,
        )
        result = crew.kickoff()
        return f"✈️ Travel Plan:\n\n{result}"

    except Exception as e:
        return f"❌ Error: {e}"


# ==============================
# Chat function
# ==============================
def chat(user_message, history):
    history = history or []
    if not user_message or not user_message.strip():
        return history, ""

    response = run_travel_planner(user_message)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    return history, ""  # second value clears the textbox


# ==============================
# UI
# ==============================
with gr.Blocks(title="AI Travel Planner") as demo:
    gr.Markdown("## ✈️ AI Travel Planner")
    gr.Markdown("Plan routes, cost & itinerary in seconds 🚀 (responses can take 30–60 seconds)")

    chatbot = gr.Chatbot(height=450)

    with gr.Row():
        txt = gr.Textbox(
            placeholder="Example: 2 day trip Chennai to Bangalore under 5000",
            scale=4,
            show_label=False,
        )
        btn = gr.Button("Send 🚀", scale=1)

    clear = gr.Button("Clear Chat ❌")

    btn.click(chat, inputs=[txt, chatbot], outputs=[chatbot, txt])
    txt.submit(chat, inputs=[txt, chatbot], outputs=[chatbot, txt])
    clear.click(lambda: [], None, chatbot)


if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft())
