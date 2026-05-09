from langchain.agents import create_agent 
from langchain.tools import tool, BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain_core.messages import AIMessage
import json


@tool 
def search_web(query: str) -> str:
    """Search the web for informaton about a topic
        this tool searches the internet and returns relevant results.
    """

    return f"Result for the '{query}' : ... fount 10 matches..."


@tool 
def weather_check(city: str) -> str: 
    """Get the current weather of a city"""
    return f"Weather of a {city}: Sunny 25 C"

def calculate_sum(a: int, b: int) -> int: 
    """add two numbers together."""
    return a + b 


class MultiInputSchema(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


class CustomTool(BaseTool):
    name: str = "calculate_product"
    description: str = "multiply two numbers together"
    args_schema: Type[BaseModel] = MultiInputSchema
    def _run(self, a: int, b: int) -> str: 
        return a * b

agent = create_agent(
    model = "claude-haiku-4-5-20251001",
    tools = [search_web,weather_check,calculate_sum, CustomTool()],#
    system_prompt = "You are a helpful assistant. Use tools to answer questions accuretely."
)

result = agent.invoke({
    "messages":[{"role":"user", "content":"What's the weather in paris and what 5 + 3 and what is 5 * 3 "}]
})
last = result["messages"][-1]
if isinstance(last, AIMessage) and not last.tool_calls:
    print(last.content)
else:
    print("Agent didn't finish cleanly:", last)