from langchain.agents import create_agent
from langchain.tools import tool 
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.tools import tool 
from dataclasses import dataclass 


@dataclass 
class Context:
    user_role: str = "user"

@tool 
def get_report(report_id: str) -> str:
    ""Get a report."""
    return f"Report {report_id}: confidential data..."

