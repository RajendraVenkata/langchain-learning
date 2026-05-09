from langchain.agents import create_agent 
from langchain.tools import tool 

@tool 
def calculate(expression: str) -> str: 
    """Evaluate a mathematical expression."""
    try: 
        result - eval(expression)
        return "result: {result}"
    except Exception as e: 
        return "Error {e}"

agent = create_agent(
    model = "claude-haiku-4-5-20251001",
    tools = [calculate],
    system_prompt="You are a helpful math tutor, when a user ask for calculations, use the calculate tool. Explain the steps clearly."
)

result = agent.invoke({
    "messages":[
        {"role":"user","content":"Calculate 2 + 2 * 3"}
    ]
})

print(result)