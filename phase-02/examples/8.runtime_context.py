from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool

# Step 1: Define context schema
@dataclass
class Context:
    user_id: str
    session_id: str
    is_premium: bool = False

# Step 2: Tool that uses context
@tool
def check_api_limit(runtime=None) -> str:
    """Check how many API calls the user has left."""
    # In a real tool, you'd use runtime.context to check limits per user
    return "API calls remaining: 100"

# Step 3: Create agent with context schema
agent = create_agent(
    model="claude-haiku-4-5-20251001",
    tools=[check_api_limit],
    context_schema=Context,
    system_prompt="You are an API assistant."
)

# Step 4: Invoke with context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "How many API calls do I have?"}]},
    context=Context(
        user_id="user_123",
        session_id="sess_abc",
        is_premium=True
    )
)

from langchain_core.messages import AIMessage
last = result["messages"][-1]
if isinstance(last, AIMessage) and not last.tool_calls:
    print(last.content)
else:
    print("Agent didn't finish cleanly:", last)