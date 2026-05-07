# Phase 02A — LangChain Core: Agents & Tools

> **Level:** Beginner → Intermediate  
> **Part:** 1 of 3 in Phase 02  
> **Source files:** `migrate-complete.md` · `oss-complete.md`  
> **Goal:** Understand how to build agents with `create_agent`, define tools with `@tool`, handle tool errors, and manage runtime context. This is the most important file in Phase 02 — everything else builds on it.

---

## Table of Contents

1. [The Big Picture — What Is an Agent?](#1-the-big-picture)
2. [`create_agent` — The Core Function](#2-create_agent)
3. [Parameters Reference](#3-parameters-reference)
4. [Prompts — Static and Dynamic](#4-prompts)
5. [Tools — Defining and Passing Them](#5-tools)
6. [Tool Error Handling](#6-tool-error-handling)
7. [Runtime Context](#7-runtime-context)
8. [Structured Output](#8-structured-output)
9. [Streaming & the Node Name Change](#9-streaming)
10. [v0 → v1 Migration Quick Reference](#10-v0-v1-migration-quick-reference)
11. [Self-Quiz](#11-self-quiz)
12. [Flashcards](#12-flashcards)

---

## 1. The Big Picture

An **agent** in LangChain v1 is a loop that:
1. Receives a user message
2. Calls a language model
3. Checks if the model wants to call a tool
4. Calls that tool and feeds the result back to the model
5. Repeats until the model produces a final answer

In v1, `create_agent` is the single entry point for building this loop. It replaced `create_react_agent` from `langgraph.prebuilt`, which is now deprecated.

```
User message
    ↓
[model node]  ←── system_prompt, middleware
    ↓
Tool call? ──Yes──→ [tool executor] ──→ back to model
    ↓ No
Final response
```

> **Key insight:** `create_agent` runs *on top of* LangGraph. Under the hood, it creates a `StateGraph`. You do not need to write any graph code to use it — that comes in Phase 03.

---

## 2. `create_agent`

### The import path changed in v1

```python
# v0 (old) — DEPRECATED
from langgraph.prebuilt import create_react_agent

# v1 (new) — USE THIS
from langchain.agents import create_agent
```

### Minimal working example

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It is sunny in {city}."

agent = create_agent(
    model="claude-sonnet-4-6",   # or "gpt-5.4-mini", etc.
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Mumbai?"}]
})
print(result)
```

---

## 3. Parameters Reference

| Parameter | Type | Description |
|---|---|---|
| `model` | `str` or `BaseChatModel` | The language model to use. Pass a string (model name) or a model instance. **Do not pass pre-bound models.** |
| `tools` | `list` | List of tools. Accepts `@tool`-decorated functions, `BaseTool` instances, or callables with type hints and docstrings. **Does NOT accept `ToolNode`.** |
| `system_prompt` | `str` | Static system prompt string. Renamed from `prompt` in v0. |
| `middleware` | `list` | List of middleware objects. Replaces pre/post-model hooks from v0. |
| `context_schema` | `dataclass` | Schema for static runtime context. Replaces `config["configurable"]` from v0. |
| `state_schema` | `TypedDict subclass` | Custom state schema extending `AgentState`. Only `TypedDict` supported — no Pydantic or dataclasses. |
| `response_format` | `ToolStrategy` or `ProviderStrategy` | For structured output. Prompted output (`str, Schema`) is no longer supported. |

### What is no longer accepted

```python
# ❌ Pre-bound models — no longer supported
model_with_tools = ChatOpenAI().bind_tools([some_tool])
agent = create_agent(model_with_tools, tools=[])

# ✅ Use this instead
agent = create_agent("gpt-5.4-mini", tools=[some_tool])

# ❌ ToolNode — no longer accepted in tools list
from langgraph.prebuilt import ToolNode
agent = create_agent(model, tools=ToolNode([tool_a, tool_b]))

# ✅ Use this instead
agent = create_agent(model, tools=[tool_a, tool_b])

# ❌ Pydantic state schema — no longer supported
class MyState(BaseModel):
    user_id: str

# ✅ Use TypedDict via AgentState
from langchain.agents import AgentState
class MyState(AgentState):
    user_id: str
```

---

## 4. Prompts

### Static prompt (most common)

The `prompt` parameter was renamed to `system_prompt` in v1. Always pass a plain string.

```python
# v1 — correct
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant."   # ✅ plain string
)

# v0 — old pattern, do not use
from langchain.messages import SystemMessage
agent = create_react_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    prompt=SystemMessage(content="You are a helpful assistant.")  # ❌ deprecated
)
```

> **Rule:** If you have a `SystemMessage` object, extract its `.content` string and pass that to `system_prompt`.

### Dynamic prompt

Dynamic prompts adapt the system prompt at runtime based on context (e.g., user role, account tier). This is a **context engineering** pattern. Use the `@dynamic_prompt` decorator and pass it as middleware.

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_role: str = "user"   # "user" | "expert" | "beginner"

@dynamic_prompt
def my_dynamic_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    base = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base} Explain concepts simply and avoid jargon."
    return base

agent = create_agent(
    model="gpt-5.4",
    tools=tools,
    middleware=[my_dynamic_prompt],
    context_schema=Context
)

# Invoke with context
agent.invoke(
    {"messages": [{"role": "user", "content": "Explain async programming"}]},
    context=Context(user_role="expert")
)
```

**When to use dynamic prompts:**
- Different user tiers (free vs. pro)
- Different roles (admin vs. viewer)
- Personalisation based on user metadata
- A/B testing prompt variations

---

## 5. Tools

### Defining a tool with `@tool`

The simplest way to define a tool is to decorate a Python function with `@tool`. The function must have:
- A docstring (this becomes the tool's description sent to the model)
- Type-annotated parameters (these define the tool's input schema)

```python
from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    # your actual implementation here
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"
```

### What the `tools` parameter accepts

```python
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
```

### Tools accessing custom state

When a tool needs to read from the agent's state (e.g., to get `user_name`), use `ToolRuntime`:

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState

class CustomState(AgentState):
    user_name: str

@tool
def greet(runtime: ToolRuntime[None, CustomState]) -> str:
    """Greet the current user by name."""
    user_name = runtime.state.get("user_name", "Unknown")
    return f"Hello {user_name}!"

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[greet],
    state_schema=CustomState
)
```

---

## 6. Tool Error Handling

In v0, tool error handling was done by wrapping tools in a `ToolNode`. In v1, it is done via middleware using the `@wrap_tool_call` decorator.

### When to handle tool errors

Handle errors that occur during **runtime** — inputs that pass schema validation but fail at execution (e.g., invalid SQL syntax, a city name that doesn't match an API's format).

**Do NOT handle:**
- Network failures (use a retry middleware instead)
- Incorrect tool implementation bugs (let them bubble up for debugging)
- Schema mismatch errors (the framework auto-handles these)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Return a friendly error message instead of crashing."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[search_web, calculate],
    middleware=[handle_tool_errors]
)
```

---

## 7. Runtime Context

Agents receive two types of data at invocation time:

| Type | Examples | How to pass |
|---|---|---|
| **Dynamic state** | Message history, tool call results | Always via `messages` key in `invoke` |
| **Static context** | User ID, session ID, user role | Via `context=` parameter on `invoke`/`stream` |

### Passing static context (v1 pattern)

```python
from dataclasses import dataclass
from langchain.agents import create_agent

@dataclass
class Context:
    user_id: str
    session_id: str
    subscription_tier: str = "free"

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=tools,
    context_schema=Context
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    context=Context(user_id="u_123", session_id="s_abc", subscription_tier="pro")
)
```

### v0 vs v1 comparison

```python
# v0 — old pattern (still works but not recommended for new code)
result = agent.invoke(
    {"messages": [...]},
    config={
        "configurable": {
            "user_id": "u_123",
            "session_id": "s_abc"
        }
    }
)

# v1 — new pattern (preferred)
result = agent.invoke(
    {"messages": [...]},
    context=Context(user_id="u_123", session_id="s_abc")
)
```

> **Note from your notes:** The old `config["configurable"]` pattern still works for backward compatibility, but using the new `context` parameter is recommended for new applications.

---

## 8. Structured Output

When you need the agent to return a structured response (not just free-form text), use `response_format` with one of two strategies.

### Two strategies in v1

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from pydantic import BaseModel

class OutputSchema(BaseModel):
    summary: str
    sentiment: str   # "positive" | "negative" | "neutral"

# Option A: ToolStrategy — uses artificial tool calling
agent = create_agent(
    model="gpt-5.4-mini",
    tools=tools,
    response_format=ToolStrategy(OutputSchema)
)

# Option B: ProviderStrategy — uses provider-native structured output
agent = create_agent(
    model="gpt-5.4-mini",
    tools=tools,
    response_format=ProviderStrategy(OutputSchema)
)
```

### What was removed

**Prompted output** is no longer supported. The old pattern of passing `("please generate ...", OutputSchema)` as `response_format` has been removed because it proved unreliable compared to tool-based and provider-native strategies.

```python
# ❌ No longer supported — removed in v1
agent = create_react_agent(
    model="gpt-5.4-mini",
    tools=tools,
    response_format=("please generate the following schema:", OutputSchema)
)
```

### Where structured output is now generated

In v0, structured output was generated in a **separate node** after the main agent loop. In v1, it is generated **in the main loop**, which reduces both cost and latency.

---

## 9. Streaming

### The node name changed

When streaming events from agents, the model node was renamed from `"agent"` to `"model"`. If you have code that filters streaming events by node name, update it.

```python
# v0 — checking for "agent" node
for event in agent.stream({"messages": [...]}):
    if "agent" in event:          # ❌ won't match in v1
        print(event["agent"])

# v1 — checking for "model" node
for event in agent.stream({"messages": [...]}):
    if "model" in event:          # ✅ correct in v1
        print(event["model"])
```

### Streaming with context

```python
for event in agent.stream(
    {"messages": [{"role": "user", "content": "Analyse this data..."}]},
    context=Context(user_id="u_123")
):
    print(event)
```

---

## 10. v0 → v1 Migration Quick Reference

| What changed | v0 (old) | v1 (new) |
|---|---|---|
| Import path | `from langgraph.prebuilt import create_react_agent` | `from langchain.agents import create_agent` |
| Function name | `create_react_agent(...)` | `create_agent(...)` |
| System prompt param | `prompt="..."` | `system_prompt="..."` |
| SystemMessage in prompt | `prompt=SystemMessage(content="...")` | `system_prompt="..."` (extract string) |
| Tools list | `tools=ToolNode([tool_a, tool_b])` | `tools=[tool_a, tool_b]` |
| Tool error handling | `ToolNode(handle_tool_errors=fn)` | `@wrap_tool_call` middleware |
| Pre-model hook | `pre_model_hook=fn` | `before_model` method on `AgentMiddleware` |
| Post-model hook | `post_model_hook=fn` | `after_model` method on `AgentMiddleware` |
| Static context | `config={"configurable": {...}}` | `context=Context(...)` |
| Dynamic prompt | function passed to `prompt=` | `@dynamic_prompt` middleware |
| State schema | Pydantic or dataclass | `TypedDict` only (inherit from `AgentState`) |
| Streaming node name | `"agent"` | `"model"` |
| Pre-bound model | `ChatOpenAI().bind_tools([...])` | Not supported — pass model string directly |

---

## 11. Self-Quiz

1. What is the new import path for creating agents in v1?
2. What three types of objects does `tools=` accept? What does it no longer accept?
3. What is the difference between `system_prompt` and a dynamic prompt? When would you use each?
4. A tool receives a city name, calls a weather API, and the API returns a 404. Should you handle this in `@wrap_tool_call`? Why or why not?
5. What is the `context=` parameter for? What's the difference between `context` and `messages`?
6. You have an existing agent that uses `config={"configurable": {"user_id": "123"}}`. Is this broken in v1? What should you migrate to?
7. What two strategies replaced prompted output for structured responses?
8. Your streaming code filters on `event["agent"]`. What do you need to change for v1?
9. Can you pass a Pydantic model as `state_schema`? What should you use instead?
10. What happens to `ValidationNode` in v1? What replaces it?

---

## 12. Flashcards

| # | Question | Answer |
|---|---|---|
| 1 | New import for agent creation in v1? | `from langchain.agents import create_agent` |
| 2 | What was `create_react_agent` renamed to? | `create_agent` |
| 3 | What was the `prompt` parameter renamed to? | `system_prompt` |
| 4 | Does `tools=` accept `ToolNode` in v1? | No — pass a plain list: `tools=[tool_a, tool_b]` |
| 5 | What decorator turns a function into a tool? | `@tool` from `langchain.tools` |
| 6 | What must every tool function have? | A docstring (description) and type-annotated parameters |
| 7 | What replaced pre/post-model hooks? | Middleware with `before_model` / `after_model` methods |
| 8 | How do you pass static context in v1? | `context=Context(...)` on `invoke`/`stream` |
| 9 | What is the streaming node name in v1? | `"model"` (was `"agent"` in v0) |
| 10 | Are pre-bound models supported in `create_agent`? | No — pass model string or unbound model instance |
| 11 | What state schema types are supported in v1? | `TypedDict` only — via `AgentState` inheritance |
| 12 | What replaced prompted structured output? | `ToolStrategy` and `ProviderStrategy` |
| 13 | What is `@wrap_tool_call` for? | Middleware decorator for handling runtime tool execution errors |
| 14 | What errors should `@wrap_tool_call` NOT handle? | Network failures, implementation bugs, schema mismatch errors |
| 15 | Where is structured output generated in v1? | In the main agent loop (not a separate node) |

---

> **Next in Phase 02:** [Phase 02B — Middleware Deep Dive](./phase-02b-middleware.md)  
> **Source notes:** `migrate-complete.md` (all agent/tool patterns), `oss-complete.md` (tutorials list)
