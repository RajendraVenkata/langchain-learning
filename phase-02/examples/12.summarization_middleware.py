from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool

@tool
def answer_question(q: str) -> str:
    """Answer a question."""
    return f"Answer to '{q}': ..."

# Create agent with summarization
agent = create_agent(
    model="gpt-5.4",
    tools=[answer_question],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4",
            trigger=("tokens", 1000),   # tuple, not dict
            keep=("messages", 20),      # also a tuple — optional, this is the default-ish
        )
    ],
    system_prompt="You are a helpful Q&A assistant.",
)

# Simulate a long conversation (20 turns)
messages = []
for i in range(20):
    messages.append({"role": "user", "content": f"Question {i}: Tell me about topic {i}"})
    messages.append({"role": "assistant", "content": f"Answer {i}: Here's information about topic {i}"})

# Invoke with long history
result = agent.invoke({
    "messages": messages  # 40 messages total — lots of tokens!
})

from langchain_core.messages import AIMessage
last = result["messages"][-1]
if isinstance(last, AIMessage) and not last.tool_calls:
    print(last.content)
else:
    print("Agent didn't finish cleanly:", last)
# The middleware automatically summarised old messages
# The model received: "Summary of earlier conversation: ..." + recent messages