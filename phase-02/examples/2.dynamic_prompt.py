from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.runtime import Runtime
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny in {city}."

@dataclass
class Context:
    user_role: str = "user"   # "user" | "expert" | "beginner"

@dynamic_prompt
def my_dynamic_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    base = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base} Explain concepts simply and avoid jargon."
    return base

agent = create_agent(
    model="claude-haiku-4-5-20251001",
    tools=[get_weather],
    middleware=[my_dynamic_prompt],
    context_schema=Context
)

# Invoke with context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain async programming"}]},
    context=Context(user_role="expert")
)

print(result)