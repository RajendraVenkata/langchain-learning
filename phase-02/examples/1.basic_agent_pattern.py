from langchain.agents import create_agent
from langchain.tools import tool 

@tool 
def check_weather(city: str) -> str :
    """Check the weather for a city

        Args: 
            city: name of the city (example Mumbai)
        
        Returns:
            A string describing the weather

    """

    return f"the weather in {city} in sunny 25C"


agent = create_agent(
    model="claude-haiku-4-5-20251001",
    tools = [check_weather],
    system_prompt = "You are a helpful assistant. When the user asks for the weather you call the check_weather tool and answer"
)

result = agent.invoke({
    "messages":[{
        "role":"user", "content": "what is the weather in Mumbai"
    }]
})

print(result)