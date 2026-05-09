from dataclasses import dataclass 
from langchain.agents import create_agent 
from langchain.tools import tool 
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass 
class Context:
    user_role: str = "user"

@tool 
def get_report(report_id: str) -> str:
    """Get a report by ID"""
    return f"{report_id}: this is the repor data."

@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    """Generate different prompts based on the user role"""
    user_role = request.runtime.context.user_role
    base_prompt = "You are a helpful assistant."

    if  user_role == 'admin':
        return f"{base_prompt} You have full access of all reports and data. Provide detailed insights."
    elif user_role == "viewer":
        return f"{base_prompt} You can read reports but cannot make changes. Be helpful but cautious."
    else:
        return base_prompt
    
agent = create_agent(
    model = "claude-haiku-4-5-20251001",
    tools = [get_report],
    middleware = [role_based_prompt],
    context_schema = Context 
)

result_admin = agent.invoke(
    {"messages": [{"role":"user", "content": "Show me report with report id 10"}]},
    context=Context(user_role="admin")
)

print("Admin result", result_admin)

result_viewer = agent.invoke({
    "messages":[{"role":"user", "content":"show me report with id 100"}]}, 
    context = Context(user_role="viewer")
)

