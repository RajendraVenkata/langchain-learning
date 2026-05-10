from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain_openai import ChatOpenAI
from typing import Callable

# Step 1: Define different models
cheap_model = ChatOpenAI(model="gpt-4o-mini")    # Fast, cheap
powerful_model = ChatOpenAI(model="gpt-4-turbo")  # Slow, expensive

# Step 2: Create middleware that decides which model to use
class DynamicModelMiddleware(AgentMiddleware):
    def __init__(self, messages_threshold: int = 10):
        self.messages_threshold = messages_threshold
    
    def wrap_model_call(
        self, 
        request: ModelRequest, 
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Choose model based on conversation length."""
        message_count = len(request.state.get("messages", []))
        
        if message_count > self.messages_threshold:
            print(f"Using powerful model (conversation has {message_count} messages)")
            selected_model = powerful_model
        else:
            print(f"Using cheap model (conversation has {message_count} messages)")
            selected_model = cheap_model
        
        # Override the model and execute
        return handler(request.override(model=selected_model))

# Step 3: Create agent with dynamic model middleware
agent = create_agent(
    model=cheap_model,  # Default
    tools=[],
    middleware=[DynamicModelMiddleware(messages_threshold=10)],
    system_prompt="You are a helpful assistant."
)

# Step 4: Test it
# First call — short conversation (uses cheap model)
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "What is 2+2?"}]
})
# Output: Using cheap model (conversation has 1 messages)

# After many turns — long conversation (uses powerful model)
messages = [
    {"role": "user", "content": f"Question {i}"}
    for i in range(15)
]
result2 = agent.invoke({"messages": messages})
# Output: Using powerful model (conversation has 15 messages)