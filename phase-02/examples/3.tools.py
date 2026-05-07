from langchain.agents import create_agent
from langchain.tools import tool, BaseTool

# 1. @tool-decorated functions ✅
@tool
def get_stock_price(ticker: str) -> str:
    """Get the current price for a stock ticker."""
    return f"Price for {ticker}: $100"

# 2. Plain callables with type hints and docstring ✅
def get_news(topic: str) -> str:
    """Get recent news about a topic."""
    return f"News about {topic}"

# 3. BaseTool subclass instances ✅
class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "Does something custom"
    def _run(self, input: str) -> str:
        return f"Result: {input}"

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[
        get_stock_price,       # @tool decorated
        get_news,              # plain callable
        MyCustomTool(),        # BaseTool instance
    ]
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the latest new on MS"}]},
)

print(result)