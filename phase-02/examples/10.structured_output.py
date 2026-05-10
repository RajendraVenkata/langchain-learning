from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.tools import tool
from pydantic import BaseModel

# Step 1: Define the output schema
class ReviewAnalysis(BaseModel):
    summary: str
    sentiment: str  # "positive", "negative", "neutral"
    rating: int     # 1-5 stars

# Step 2: Define a sample tool
@tool
def fetch_review(review_id: str) -> str:
    """Fetch a customer review by ID."""
    return "Great product! Works exactly as described. Very happy with the purchase. Highly recommend!"

# Step 3a: Create agent with ToolStrategy (works with any model)
agent_tool_strategy = create_agent(
    model="gpt-5-mini",
    tools=[fetch_review],
    response_format=ToolStrategy(ReviewAnalysis),
    system_prompt="You are a review analyst. Analyze reviews and provide structured output in JSON."
)

# Step 3b: Alternatively, use ProviderStrategy (faster, but model-specific)
agent_provider_strategy = create_agent(
    model="gpt-5-mini",
    tools=[fetch_review],
    response_format=ProviderStrategy(ReviewAnalysis),
    system_prompt="You are a review analyst."
)

# Step 4: Invoke
result = agent_provider_strategy.invoke({
    "messages": [{"role": "user", "content": "Analyze review #123"}]
})

from langchain_core.messages import AIMessage
last = result["messages"][-1]
if isinstance(last, AIMessage) and not last.tool_calls:
    print(last.content)
else:
    print("Agent didn't finish cleanly:", last)
# Output is now ALWAYS a dict:
# {
#     "summary": "Customer loved the product and recommends it",
#     "sentiment": "positive",
#     "rating": 5
# }

# Access structured fields
#print(f"Sentiment: {result['sentiment']}")
#print(f"Rating: {result['rating']}/5")