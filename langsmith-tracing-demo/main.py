
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Match the endpoint used by test.py when no endpoint is configured.
if not os.getenv("LANGSMITH_ENDPOINT"):
    os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"

# Verify tracing configuration BEFORE running
required_keys = [
    "OPENAI_API_KEY",
    "LANGSMITH_API_KEY"
]

for key in required_keys:
    if not os.getenv(key):
        raise ValueError(f"Missing environment variable: {key}")

if os.getenv("LANGSMITH_TRACING", "").lower() != "true":
    raise ValueError("LangSmith tracing is OFF!")

print("OpenAI API key: Available")
print("LangSmith API key: Available")
print("LangSmith tracing: ENABLED")
print("LangSmith endpoint:", os.getenv("LANGSMITH_ENDPOINT"))
print("Project:", os.getenv("LANGSMITH_PROJECT"))

# Initialize LangChain model
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# Three test prompts
prompts = {
    "Prompt 1 - Simple": 
        "What is artificial intelligence? Answer in 2 sentences.",
"Prompt 2 - Technical":
        "Explain what is contact center "
        "Explain about webex contact center",

    "Prompt 3 - Detailed Analysis":
        "Explain How AI is used in Contact center "
        "Explain Webex Architecture "
        "SIP connectivity, call routing, IVR, agent integration, "
             
}

# Execute each prompt
for name, prompt in prompts.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    start_time = time.perf_counter()

    response = llm.invoke(prompt)

    end_time = time.perf_counter()

    latency = end_time - start_time

    usage = response.usage_metadata or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    print("\nResponse:")
    print(response.content)

    print("\nPerformance Metrics:")
    print("Input Tokens:", input_tokens)
    print("Output Tokens:", output_tokens)
    print("Total Tokens:", total_tokens)
    print("Latency:", round(latency, 2), "seconds")
