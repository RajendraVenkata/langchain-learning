from langchain.agents import AgentState, create_agent
from langchain.tools import tool

# v1 (Correct) — inherit from AgentState (which is a TypedDict)
class MyAgentState(AgentState):
    user_id: str
    subscription_tier: str = "free"  # Optional with default

@tool
def check_subscription(runtime=None) -> str:
    """Check the user's subscription status."""
    # In a real tool, you'd use runtime to access state
    return "Subscription: Premium"

agent = create_agent(
    model="claude-haiku-4-5-20251001",
    tools=[check_subscription],
    state_schema=MyAgentState,
    system_prompt="You are a subscription assistant."
)

# Invoke with state
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's my subscription?"}],
    "user_id": "user_123",
    "subscription_tier": "premium"
})

from langchain_core.messages import AIMessage
last = result["messages"][-1]
if isinstance(last, AIMessage) and not last.tool_calls:
    print(last.content)
else:
    print("Agent didn't finish cleanly:", last)