from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langchain_core.messages import AIMessage

# Step 1: Define custom state with user_name
class CustomState(AgentState):
    user_name: str  # Add a custom field to the agent state

# Step 2: Define a tool that accesses the state
@tool
def greet(runtime: ToolRuntime[None, CustomState]) -> str:
    """Greet the user by their name.
    
    This tool reads the user_name from the agent state.
    """
    user_name = runtime.state.get("user_name", "Guest")
    return f"Hello {user_name}! How can I help you today?"

@tool
def remind_user(reminder: str, runtime: ToolRuntime[None, CustomState]) -> str:
    """Set a reminder for the user."""
    user_name = runtime.state.get("user_name", "Guest")
    return f"Reminder set for {user_name}: {reminder}"

# Step 3: Create agent with custom state
agent = create_agent(
    model="claude-haiku-4-5-20251001",
    tools=[greet, remind_user],
    state_schema=CustomState,
    system_prompt="You are a helpful assistant. Use the greet tool when appropriate."
)

# Step 4: Invoke with state
result = agent.invoke({
    "messages": [{"role": "user", "content": "set reminder I need to wakeup at 5 AM"}],
    "user_name": "Alice"  # ← Supply the custom state field
})

#print(result)
# Output: "Hello Alice! How can I help you today?"

last = result["messages"][-1]
if isinstance(last, AIMessage) and not last.tool_calls:
    print(last.content)
else:
    print("Agent didn't finish cleanly:", last)