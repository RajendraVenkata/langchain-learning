from langchain.agents import create_agent 
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool 
from langchain.messages import ToolMessage 

@tool 
def execute_sql(query: str) -> str:
    """Execute a SQL query and return results."""
    if "DROP" in query.upper():
        raise ValueError("Drop statements are not allowed")
    if "SELECT" not in query.upper():
         raise ValueError("only select queries are allowed")
    return f"Results: {query[:50]}..."


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Catch tool execution errors and return friendly messages."""
    try:
        return handler(request)  # Execute the tool normally
    except ValueError as e:
        # Return error as a ToolMessage the agent can see
        return ToolMessage(
            content=f"Tool error: {str(e)}. Please check your input and try again.",
            tool_call_id=request.tool_call["id"]
        )
    except Exception as e:
        # Catch unexpected errors too
        return ToolMessage(
            content=f"Unexpected error: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model = "claude-haiku-4-5-20251001",
    tools = [execute_sql],
    moddleware = [handle_tool_errors],
    system_prompt = "You are a SQL assistant. Help user write safe SELECT queries."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Run DROP TABLE users"}]
})

print(result)