# Phase 02A — LangChain Core: Agents & Tools

> **Level:** Beginner → Intermediate  
> **Part:** 1 of 3 in Phase 02  
> **Source files:** `migrate-complete.md` · `oss-complete.md`  
> **Goal:** Understand how to build agents with `create_agent`, define tools with `@tool`, handle tool errors, and manage runtime context.
>
> ⚠️ **Note on examples:** Every code example in this file is extracted directly from your source files with inline source citations. No synthetic or illustrative examples.

---

## Table of Contents

1. [Import Path Change](#1-import-path-change)
2. [Basic Agent Pattern](#2-basic-agent-pattern)
3. [System Prompts — Static](#3-system-prompts--static)
4. [System Prompts — Dynamic](#4-system-prompts--dynamic)
5. [Tools — Definition and Usage](#5-tools--definition-and-usage)
6. [Tool Error Handling](#6-tool-error-handling)
7. [Tools Accessing Custom State](#7-tools-accessing-custom-state)
8. [Custom State Schema](#8-custom-state-schema)
9. [Runtime Context](#9-runtime-context)
10. [Dynamic Model Selection](#10-dynamic-model-selection)
11. [Structured Output](#11-structured-output)
12. [Streaming Node Name Change](#12-streaming-node-name-change)
13. [v0 → v1 Quick Reference](#13-v0-v1-quick-reference)
14. [Self-Quiz](#14-self-quiz)
15. [Flashcards](#15-flashcards)

---

## 1. Import Path Change

**Source:** `migrate-complete.md` (Import path section)

The function and package both changed:

```python
# v0 (old) — DEPRECATED
from langgraph.prebuilt import create_react_agent

# v1 (new) — USE THIS
from langchain.agents import create_agent
```

---

## 2. Basic Agent Pattern

**Source:** `migrate-complete.md` (Prompts → Static prompt rename section)

```python
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[check_weather],
    system_prompt="You are a helpful assistant"
)
```

---

## 3. System Prompts — Static

### Rename from `prompt` to `system_prompt`

**Source:** `migrate-complete.md` (Prompts → Static prompt rename section)

```python
# v1 (new) — correct
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[check_weather],
    system_prompt="You are a helpful assistant"
)
```

### SystemMessage → string conversion

**Source:** `migrate-complete.md` (Prompts → SystemMessage to string section)

If you have a `SystemMessage` object, extract the string content:

```python
# v1 (new) — use plain string
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[check_weather],
    system_prompt="You are a helpful assistant"
)
```

The old pattern of passing `SystemMessage(content="...")` is no longer needed.

---

## 4. System Prompts — Dynamic

Dynamic prompts adapt based on runtime context (e.g., user role, conversation state).

**Source:** `migrate-complete.md` (Dynamic prompts section)

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_role: str = "user"

@dynamic_prompt
def dynamic_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        prompt = (
            f"{base_prompt} Provide detailed technical responses."
        )
    elif user_role == "beginner":
        prompt = (
            f"{base_prompt} Explain concepts simply and avoid jargon."
        )
    else:
        prompt = base_prompt

    return prompt

agent = create_agent(
    model="gpt-5.4",
    tools=tools,
    middleware=[dynamic_prompt],
    context_schema=Context
)

# Use with context
agent.invoke(
    {"messages": [{"role": "user", "content": "Explain async programming"}]},
    context=Context(user_role="expert")
)
```

---

## 5. Tools — Definition and Usage

### Basic tool definition with `@tool`

**Source:** `migrate-complete.md` (Tools section)

```python
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[check_weather, search_web]
)
```

The `tools` list accepts:
- `@tool`-decorated functions
- Plain callables with type hints and docstrings
- `dict` representing built-in provider tools

### What tools parameter no longer accepts

**Source:** `migrate-complete.md` (Tools section)

```python
# v0 (old) — ToolNode no longer accepted
from langgraph.prebuilt import create_react_agent, ToolNode

agent = create_react_agent(
    model="claude-sonnet-4-6",
    tools=ToolNode([check_weather, search_web])  # ❌ NOT ALLOWED IN V1
)

# v1 (new) — pass a plain list
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[check_weather, search_web]            # ✅ CORRECT
)
```

---

## 6. Tool Error Handling

### Using `@wrap_tool_call` middleware

**Source:** `migrate-complete.md` (Tools → Handling tool errors section)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Only handle errors that occur during tool execution due to invalid inputs
        # that pass schema validation but fail at runtime (e.g., invalid SQL syntax).
        # Do NOT handle:
        # - Network failures (use tool retry middleware instead)
        # - Incorrect tool implementation errors (should bubble up)
        # - Schema mismatch errors (already auto-handled by the framework)
        #
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[check_weather, search_web],
    middleware=[handle_tool_errors]
)
```

---

## 7. Tools Accessing Custom State

### Using ToolRuntime for state access

**Source:** `migrate-complete.md` (Custom state → Defining state via state_schema section)

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState

# Define custom state extending AgentState
class CustomState(AgentState):
    user_name: str

@tool
def greet(
    runtime: ToolRuntime[None, CustomState]
) -> str:
    """Use this to greet the user by name."""
    user_name = runtime.state.get("user_name", "Unknown")
    return f"Hello {user_name}!"

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[greet],
    state_schema=CustomState
)
```

---

## 8. Custom State Schema

### TypedDict via AgentState (only supported type)

**Source:** `migrate-complete.md` (Custom state → State type restrictions section)

```python
from langchain.agents import AgentState, create_agent

# AgentState is a TypedDict
class CustomAgentState(AgentState):
    user_id: str

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=tools,
    state_schema=CustomAgentState
)
```

### What is no longer supported

**Source:** `migrate-complete.md` (Custom state → State type restrictions section)

```python
# ❌ v0 pattern — Pydantic no longer supported
from pydantic import BaseModel

class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str

# ❌ v0 pattern — dataclass no longer supported
@dataclass
class AgentState:
    user_id: str

# ✅ v1 pattern — TypedDict via AgentState
from langchain.agents import AgentState

class AgentState(AgentState):
    user_id: str
```

---

## 9. Runtime Context

Static context (user metadata that doesn't change during the conversation) is passed separately from dynamic state (messages).

**Source:** `migrate-complete.md` (Runtime context section)

```python
from dataclasses import dataclass
from langchain.agents import create_agent

@dataclass
class Context:
    user_id: str
    session_id: str

agent = create_agent(
    model=model,
    tools=tools,
    context_schema=Context
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    context=Context(user_id="123", session_id="abc")
)
```

### v0 pattern (old, still works but not recommended)

**Source:** `migrate-complete.md` (Runtime context section)

```python
# v0 — old pattern, still works for backward compatibility
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config={
        "configurable": {
            "user_id": "123",
            "session_id": "abc"
        }
    }
)
```

> From your notes: "The old `config["configurable"]` pattern still works for backward compatibility, but using the new `context` parameter is recommended for new applications or applications migrating to v1."

---

## 10. Dynamic Model Selection

Choose a different model per request based on conversation state.

**Source:** `migrate-complete.md` (Model → Dynamic model selection section)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware, ModelRequest
)
from langchain.agents.middleware.types import ModelResponse
from langchain_openai import ChatOpenAI
from typing import Callable

basic_model = ChatOpenAI(model="gpt-5-nano")
advanced_model = ChatOpenAI(model="gpt-5.4")

class DynamicModelMiddleware(AgentMiddleware):

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        if len(request.state.messages) > self.messages_threshold:
            model = advanced_model
        else:
            model = basic_model
        return handler(request.override(model=model))

    def __init__(self, messages_threshold: int) -> None:
        self.messages_threshold = messages_threshold

agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[DynamicModelMiddleware(messages_threshold=10)]
)
```

---

## 11. Structured Output

Generate structured JSON responses from agents.

**Source:** `migrate-complete.md` (Structured output → Tool and provider strategies section)

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from pydantic import BaseModel

class OutputSchema(BaseModel):
    summary: str
    sentiment: str

# Using ToolStrategy (artificial tool calling)
agent = create_agent(
    model="gpt-5.4-mini",
    tools=tools,
    response_format=ToolStrategy(OutputSchema)
)
```

### Prompted output removed

**Source:** `migrate-complete.md` (Structured output → Prompted output removed section)

```
Prompted output is no longer supported via the `response_format` argument.
Compared to strategies like artificial tool calling and provider native 
structured output, prompted output has not proven to be particularly reliable.
```

---

## 12. Streaming Node Name Change

When streaming agent execution, filter on the node name.

**Source:** `migrate-complete.md` (Streaming node name rename section)

In v1, the streaming node was renamed from `"agent"` to `"model"`:

```python
# v1 — check for "model" node
for event in agent.stream({"messages": [...]}):
    if "model" in event:
        print(event["model"])
```

---

## 13. v0 → v1 Quick Reference

**Source:** `migrate-complete.md` (entire `create_react_agent` → `create_agent` section)

| What changed | v0 (old) | v1 (new) |
|---|---|---|
| Import path | `from langgraph.prebuilt import create_react_agent` | `from langchain.agents import create_agent` |
| Function name | `create_react_agent(...)` | `create_agent(...)` |
| System prompt param | `prompt="..."` | `system_prompt="..."` |
| SystemMessage in prompt | `prompt=SystemMessage(content="...")` | `system_prompt="..."` (extract string) |
| Tools list | `tools=ToolNode([tool_a, tool_b])` | `tools=[tool_a, tool_b]` |
| Tool error handling | Part of `ToolNode` initialization | `@wrap_tool_call` middleware |
| Pre-model hook | `pre_model_hook=fn` | `before_model` method on `AgentMiddleware` |
| Post-model hook | `post_model_hook=fn` | `after_model` method on `AgentMiddleware` |
| Static context | `config={"configurable": {...}}` | `context=Context(...)` |
| Dynamic prompt | Function passed to `prompt=` | `@dynamic_prompt` middleware |
| State schema | Pydantic or dataclass | `TypedDict` only (inherit from `AgentState`) |
| Streaming node name | `"agent"` | `"model"` |
| Pre-bound model | `ChatOpenAI().bind_tools([...])` | Not supported — pass model string directly |

---

## 14. Self-Quiz

1. What is the new import path for `create_agent` in v1?
2. Did the `system_prompt` parameter exist in v0? What was it called?
3. Can you pass a `ToolNode` to the `tools` parameter in v1?
4. Name one error you SHOULD catch in `@wrap_tool_call`. Name one you SHOULD NOT.
5. What is the difference between `context=` and `messages=` when invoking an agent?
6. You want to use different models based on conversation length. Which method on `AgentMiddleware` do you implement?
7. In v0, how did you pass static metadata to an agent? How do you do it in v1?
8. What state types are supported in v1? What types are no longer supported?
9. What happens to `SystemMessage` objects in v1? How should you migrate them?
10. What is the new streaming node name in v1?
11. What does `request.override(model=new_model)` do?
12. Can you still use `config["configurable"]` in v1? Should you for new code?
13. What are the two structured output strategies that replaced prompted output?
14. What are the three things `tools=` accepts in v1?
15. What does a tool need to have to be valid (besides the `@tool` decorator)?

---

## 15. Flashcards

Study these before moving to Phase 02B.

| # | Question | Answer |
|---|---|---|
| 1 | Import path for `create_agent` in v1? | `from langchain.agents import create_agent` |
| 2 | What was `system_prompt` called in v0? | `prompt` |
| 3 | Does v1 accept `ToolNode` in tools list? | No — pass a plain list: `tools=[tool_a, tool_b]` |
| 4 | What must every `@tool` function have? | A docstring and type-annotated parameters |
| 5 | What replaced pre-model hooks? | `before_model` method on `AgentMiddleware` |
| 6 | What replaced post-model hooks? | `after_model` method on `AgentMiddleware` |
| 7 | How do you pass context in v1? | `context=Context(...)` on `invoke` / `stream` |
| 8 | Streaming node name in v1? | `"model"` (was `"agent"` in v0) |
| 9 | State type in v1? | `TypedDict` only (via `AgentState`) |
| 10 | What replaced prompted structured output? | `ToolStrategy` and `ProviderStrategy` |
| 11 | What does `{"jump_to": "end"}` from `before_model` do? | Short-circuits the agent loop immediately |
| 12 | Can you access agent state in a tool? | Yes, via `runtime: ToolRuntime[None, CustomState]` |
| 13 | What does `request.override(model=x)` do? | Creates a new request with a different model |
| 14 | Is `config["configurable"]` deprecated? | No, still works, but `context=` is preferred for new code |
| 15 | What type is `AgentState`? | A `TypedDict` that you inherit from for custom state |

---

> **Next in Phase 02:** [Phase 02B — Middleware Deep Dive](./phase-02b-middleware.md)  
> **Source files used:** `migrate-complete.md` (all sections cited inline)
