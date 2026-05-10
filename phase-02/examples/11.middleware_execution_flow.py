from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from typing import Any

class Middleware1(AgentMiddleware):
    def before_model(self, state, runtime) -> dict[str, Any] | None:
        print("1: before_model")
        return None
    
    def after_model(self, state, runtime) -> dict[str, Any] | None:
        print("1: after_model")
        return None

class Middleware2(AgentMiddleware):
    def before_model(self, state, runtime) -> dict[str, Any] | None:
        print("2: before_model")
        return None
    
    def after_model(self, state, runtime) -> dict[str, Any] | None:
        print("2: after_model")
        return None

agent = create_agent(
    model="gpt-5-mini",
    tools=[],
    middleware=[Middleware1(), Middleware2()]
)

agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

# Output:
# 1: before_model
# 2: before_model
# [MODEL CALL]
# 2: after_model
# 1: after_model