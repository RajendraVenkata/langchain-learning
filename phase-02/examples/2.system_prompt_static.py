from langchain.agents import create_agent 
from langchain.tools import tool 

@tool 
def calculate(expression: str) -> str :
    """
        Args: 
            expression: evaluate the methematical expession in string format.
        Result:
            Returns. the value/result of the expression  
    """

    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

agent = create_agent(
    model = "claude-haiku-4-5-20251001",
    tools = [calculate],
    system_prompt= "You are a helpful math tutor. when user asks for a calculation, use the calculator tool. Explain the steps Clearly."
)

result = agent.invoke({
    "messages": [{"role":"user","content":"Calculate 2 + 2 * 3"}]
})
print(result)