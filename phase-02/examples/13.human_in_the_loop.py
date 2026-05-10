"""
Human-in-the-loop example with HumanInTheLoopMiddleware.

Run:
    export OPENAI_API_KEY=sk-...
    python 13.human_in_the_loop.py
"""

import json

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


# ---- Tools ----------------------------------------------------------------

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to someone."""
    return f"Email sent to {recipient} | subject={subject!r} | body={body!r}"


@tool
def delete_file(filename: str) -> str:
    """Delete a file (irreversible!)."""
    return f"File deleted: {filename}"


@tool
def read_file(filename: str) -> str:
    """Read a file (safe, no approval needed)."""
    return f"Contents of {filename}: <pretend file contents here>"


# ---- Agent ----------------------------------------------------------------

agent = create_agent(
    model="openai:gpt-4o",
    tools=[send_email, delete_file, read_file],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "Please review this email before sending:",
                },
                "delete_file": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "Confirm file deletion (this is irreversible):",
                },
                # read_file is omitted -> runs without approval
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
    system_prompt="You are a helpful assistant.",
    checkpointer=InMemorySaver(),  # required for interrupts
)


# ---- Schema-tolerant extraction ------------------------------------------

# Different langchain versions use different keys for the same data.
# These helpers find the right one whatever your version uses.

_NAME_KEYS = ("name", "action", "tool_name", "tool")
_ARGS_KEYS = ("arguments", "args", "tool_input", "input")


def _pick(d: dict, keys: tuple) -> object:
    for k in keys:
        if k in d:
            return d[k]
    return None


def normalize_action_requests(interrupt_value) -> list[dict]:
    """
    Return a list of {'name': str, 'args': dict, 'description': str|None}
    regardless of which schema the installed version uses.
    """
    # The interrupt value can be a dict with 'action_requests', or directly
    # a list of action_requests, or a single HumanInterrupt-shaped dict.
    raw_list = None
    if isinstance(interrupt_value, dict):
        if "action_requests" in interrupt_value:
            raw_list = interrupt_value["action_requests"]
        elif "action_request" in interrupt_value:        # legacy single
            raw_list = [interrupt_value["action_request"]]
        else:
            raw_list = [interrupt_value]
    elif isinstance(interrupt_value, list):
        raw_list = interrupt_value
    else:
        raw_list = [interrupt_value]

    out = []
    for item in raw_list:
        if not isinstance(item, dict):
            out.append({"name": str(item), "args": {}, "description": None})
            continue
        name = _pick(item, _NAME_KEYS) or "<unknown>"
        args = _pick(item, _ARGS_KEYS) or {}
        desc = item.get("description")
        out.append({"name": name, "args": args, "description": desc})
    return out


# ---- Helpers --------------------------------------------------------------

def prompt_for_decisions(interrupt_value):
    """Show pending tool calls to the user and collect approve/reject decisions."""
    actions = normalize_action_requests(interrupt_value)

    print("\n" + "=" * 60)
    print("Agent wants to perform the following action(s):")
    print("=" * 60)

    for i, a in enumerate(actions):
        print(f"\n[{i}] Tool: {a['name']}")
        try:
            pretty = json.dumps(a["args"], indent=2, default=str)
        except Exception:
            pretty = repr(a["args"])
        print(f"    Arguments: {pretty}")
        if a["description"]:
            print(f"    Note: {a['description']}")

    decisions = []
    for i, a in enumerate(actions):
        while True:
            choice = input(
                f"\nDecision for [{i}] {a['name']} -- (a)pprove / (r)eject: "
            ).strip().lower()
            if choice in ("a", "approve"):
                decisions.append({"type": "approve"})
                break
            if choice in ("r", "reject"):
                reason = input("  Rejection reason (optional): ").strip()
                decisions.append({
                    "type": "reject",
                    "message": reason or "User rejected the action.",
                })
                break
            print("  Please type 'a' or 'r'.")

    return decisions


def get_interrupts(result):
    """Return list of interrupts whether using v2 GraphOutput or legacy dict."""
    interrupts = getattr(result, "interrupts", None)
    if interrupts:
        return list(interrupts)
    if isinstance(result, dict) and result.get("__interrupt__"):
        return list(result["__interrupt__"])
    return []


def get_final_messages(result):
    """Pull the messages list off either return shape."""
    if hasattr(result, "value"):
        return result.value.get("messages", [])
    if isinstance(result, dict):
        return result.get("messages", [])
    return []


def run(user_message: str, thread_id: str = "demo-thread-1"):
    config = {"configurable": {"thread_id": thread_id}}

    invoke_kwargs = {"config": config}
    # Try v2 first; fall back gracefully if this version doesn't accept it.
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            version="v2",
            **invoke_kwargs,
        )
        invoke_kwargs["version"] = "v2"
    except TypeError:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            **invoke_kwargs,
        )

    first_dump = True
    while True:
        interrupts = get_interrupts(result)
        if not interrupts:
            break

        interrupt = interrupts[0]
        value = getattr(interrupt, "value", interrupt)

        # On first interrupt, dump the raw structure so the schema is visible.
        if first_dump:
            print("\n[debug] raw interrupt value:")
            try:
                print(json.dumps(value, indent=2, default=str)[:2000])
            except Exception:
                print(repr(value)[:2000])
            first_dump = False

        decisions = prompt_for_decisions(value)
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            **invoke_kwargs,
        )

    messages = get_final_messages(result)
    print("\n" + "=" * 60)
    print("Final response:")
    print("=" * 60)
    if messages:
        last = messages[-1]
        print(getattr(last, "content", last))
    else:
        print("(no messages returned)")


# ---- Entry point ----------------------------------------------------------

if __name__ == "__main__":
    run("Send an email to alice@example.com with subject 'Hello' and body 'Hi Alice!'")

    # Try these too:
    # run("Read the file /etc/hostname for me.", thread_id="demo-thread-2")
    # run("Delete the file /tmp/old_log.txt", thread_id="demo-thread-3")