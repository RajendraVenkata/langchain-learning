from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool
from typing_extensions import NotRequired
from typing import Any


# Step 1: Define state that this middleware uses
class CallCounterState(AgentState):
    model_call_count: NotRequired[int]  # Optional field, no default


# Step 2: Create middleware with its own state
class CallCounterMiddleware(AgentMiddleware[CallCounterState]):
    state_schema = CallCounterState  # Associate state with this middleware

    def before_model(self, state: CallCounterState, runtime) -> dict[str, Any] | None:
        count = state.get("model_call_count", 0)
        print(f"Model call #{count + 1}")

        if count > 10:
            print("Too many model calls! Stopping.")
            return {"jump_to": "end"}  # Short-circuit
        return None

    def after_model(self, state: CallCounterState, runtime) -> dict[str, Any] | None:
        count = state.get("model_call_count", 0)
        return {"model_call_count": count + 1}  # Increment counter


# Step 3: Define some tools so the agent actually loops
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    fake_data = {
        "Paris": "Cloudy, 14°C",
        "Tokyo": "Sunny, 22°C",
        "New York": "Rainy, 10°C",
        "London": "Foggy, 8°C",
    }
    return fake_data.get(city, f"Weather data unavailable for {city}")


@tool
def get_time(city: str) -> str:
    """Get the current local time for a given city."""
    fake_data = {
        "Paris": "3:00 PM",
        "Tokyo": "11:00 PM",
        "New York": "9:00 AM",
        "London": "2:00 PM",
    }
    return fake_data.get(city, f"Time data unavailable for {city}")


# Step 4: Create agent with the middleware and tools
agent = create_agent(
    model="gpt-5-mini",
    tools=[get_weather, get_time],
    middleware=[CallCounterMiddleware()],
    system_prompt=(
        "You are a helpful travel assistant. "
        "When asked about multiple cities, look up information for each one "
        "using the available tools before giving a final answer."
    ),
)


# Step 5: Invoke with a query that requires multiple tool calls
if __name__ == "__main__":
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                "I'm planning a trip. Can you tell me the current weather "
                "and local time in Paris, Tokyo, and New York?"
            ),
        }]
    })

    print("\n--- Final Answer ---")
    print(result["messages"][-1].content)
    print(f"\nTotal model calls: {result.get('model_call_count', 0)}")