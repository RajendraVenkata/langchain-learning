from langchain.agents import create_agent
from langchain.tools import tool 

@tool
def check_weather(city: str) -> str :
    """Check the current weather of a city

        Args:
            city: name of the city (e.g Mumbay, Hyderabad)

        Returns: 
            A string describing the weather
    """

    return f"The weather in {city} is sunny 25C"

agent = create_agent(
    model = "claude-haiku-4-5-20251001",
    tools = [check_weather],
    system_prompt = "You are a helpful assistant. Answer the weather questions using check_weather tool."
)

result = agent.invoke({
    "messages":[{"role": "user", "content":"what is the weather in Mumbai?"}]
})

print(result)
