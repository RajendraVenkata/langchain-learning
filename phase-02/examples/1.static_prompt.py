from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny in {city}."

agent = create_agent(
    model="claude-haiku-4-5-20251001",   # or "gpt-5.4-mini", etc.
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Mumbai?"}]
})
print(result)