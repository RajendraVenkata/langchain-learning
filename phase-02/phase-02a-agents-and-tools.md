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

Dynamic prompts adapt based on runtime context (e.g., user role, conversation state). Instead of using a fixed `system_prompt` string, you define a function that generates the prompt at runtime based on the current context.

**Source:** `migrate-complete.md` (Dynamic prompts section)

### Breaking down the example

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.runtime import Runtime
```

**What's being imported:**
- `dataclass` — Python decorator to define a simple data structure for context
- `create_agent` — the agent factory function (Phase 02A §2)
- `dynamic_prompt` — a decorator that marks a function as a dynamic prompt middleware
- `ModelRequest` — the request object passed to the dynamic prompt function, containing the current agent state and runtime

---

### Step 1: Define your context schema

```python
@dataclass
class Context:
    user_role: str = "user"
```

**What this does:**
- Defines a simple `dataclass` called `Context` with one field: `user_role`
- `user_role` defaults to `"user"` if not provided
- This is the **static context** you will pass at invocation time — data that doesn't change during the conversation
- Later, when you call `agent.invoke(...)`, you'll pass `context=Context(user_role="expert")` to tell the agent which role the current user is

---

### Step 2: Define the dynamic prompt function

```python
@dynamic_prompt
def dynamic_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    base_prompt = "You are a helpful assistant."
```

**What's happening:**
- `@dynamic_prompt` decorator marks this function as a middleware that generates prompts dynamically
- The function receives a `request: ModelRequest` — this object contains:
  - `request.state` — the current agent state (messages, custom fields)
  - `request.runtime.context` — the static context you passed at invocation (in this case, the `Context` object with `user_role`)
- `request.runtime.context.user_role` extracts the user role from the context

```python
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
```

**What's happening:**
- Branches on the `user_role` to customize the prompt
- **Expert:** adds instruction to provide detailed technical responses
- **Beginner:** adds instruction to explain simply and avoid jargon
- **Default (any other value):** uses the base prompt as-is
- Returns the final system prompt string that the model will see

---

### Step 3: Register the dynamic prompt as middleware

```python
agent = create_agent(
    model="gpt-5.4",
    tools=tools,
    middleware=[dynamic_prompt],
    context_schema=Context
)
```

**What's happening:**
- `middleware=[dynamic_prompt]` — registers your dynamic prompt function as a middleware (like in Phase 02B)
- `context_schema=Context` — tells the agent "I will pass you a `Context` object at invocation time with user role and other metadata"

---

### Step 4: Invoke with context

```python
agent.invoke(
    {"messages": [{"role": "user", "content": "Explain async programming"}]},
    context=Context(user_role="expert")
)
```

**What's happening:**
- `{"messages": [...]}` — the dynamic state (the conversation)
- `context=Context(user_role="expert")` — the static context for this call
  - When the agent is about to call the model, the `dynamic_prompt` function runs
  - It reads `request.runtime.context.user_role`, sees `"expert"`
  - It returns `"You are a helpful assistant. Provide detailed technical responses."`
  - The model receives this prompt and answers accordingly

### When to use dynamic prompts

**Use dynamic prompts when:**
- Different user roles (admin, viewer, guest) need different instructions
- User preferences or account tier affect the prompt
- Conversation state should influence the system prompt
- You want to A/B test different prompt variations

**Don't use dynamic prompts when:**
- The prompt never changes — just use `system_prompt="..."`
- You only need one version of the prompt

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

**What's happening:**
- `check_weather` and `search_web` are tools that the agent can call
- They are **not defined here** in this snippet (the full definitions are in your source with docstrings and type hints)
- The agent will look at the docstrings and type hints of these functions to understand:
  - **What they do** (from the docstring)
  - **What inputs they need** (from the type-annotated parameters)
  - **What they return** (from the return type annotation)

**The agent's job:** When the model decides it wants to call a tool, the agent:
1. Looks up the tool by name (e.g., `"check_weather"`)
2. Gets the actual input from the model
3. Calls the tool function with that input
4. Gets the result back
5. Feeds the result to the model as a `ToolMessage`

**What the `@tool` decorator does:**
- Converts a regular Python function into a LangChain tool
- Automatically extracts the docstring as the tool description
- Automatically extracts parameter type hints as the tool's input schema
- Makes the function compatible with agents

---

### What tools parameter accepts

The `tools=` parameter accepts three types of objects:

**1. Functions decorated with `@tool`:**
```python
from langchain.tools import tool

@tool
def check_weather(city: str) -> str:
    """Check the weather in a city. Returns a weather description."""
    # implementation
    return f"Weather in {city}: sunny"
```

**2. Plain callables with type hints and docstring:**
```python
def search_web(query: str) -> str:
    """Search the web for information. Returns search results."""
    # implementation
    return f"Results for {query}"

# Pass directly to tools list
agent = create_agent(
    model="...",
    tools=[search_web]  # Plain function, no @tool needed
)
```

**3. `BaseTool` subclass instances:**
```python
from langchain.tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "Does something custom"
    
    def _run(self, input: str) -> str:
        return f"Result: {input}"

agent = create_agent(
    model="...",
    tools=[MyCustomTool()]  # Instance of BaseTool
)
```

---

### What is NOT accepted anymore

**Source:** `migrate-complete.md` (Tools section)

```python
# ❌ v0 (old) — ToolNode no longer accepted
from langgraph.prebuilt import create_react_agent, ToolNode

agent = create_react_agent(
    model="claude-sonnet-4-6",
    tools=ToolNode([check_weather, search_web])
)
```

**Why this fails in v1:**
- `ToolNode` was a v0 pattern that bundled tools together with execution logic
- In v1, tool execution is handled by the agent framework automatically
- You just pass a plain list of tool functions — the framework does the rest

```python
# ✅ v1 (new) — pass a plain list
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[check_weather, search_web]
)
```

**The difference:**
- v0: Tools were wrapped in `ToolNode`, and you had to understand the execution model
- v1: Just list your tools — the agent handles everything

---

## 6. Tool Error Handling

Tool errors happen when a tool receives an input that passes schema validation but fails at runtime (e.g., an invalid SQL query, a city name that doesn't match an API's format, a malformed math expression).

**Source:** `migrate-complete.md` (Tools → Handling tool errors section)

### Using `@wrap_tool_call` middleware

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

### Breaking down the `@wrap_tool_call` decorator

**What `@wrap_tool_call` does:**
- Marks a function as a tool call wrapper middleware (see Phase 02B for full middleware details)
- The function receives a `request` (information about the tool call) and a `handler` (the actual tool execution)

**Inside the function:**
```python
try:
    return handler(request)
```
- `handler(request)` **executes the tool**
- If the tool runs successfully, its result is returned immediately

```python
except Exception as e:
```
- If the tool throws an **exception**, you catch it here
- `e` is the exception object

```python
    return ToolMessage(
        content=f"Tool error: Please check your input and try again. ({str(e)})",
        tool_call_id=request.tool_call["id"]
    )
```
- Instead of letting the exception crash the agent, you return a `ToolMessage`
- `ToolMessage` is a message that tells the agent: "The tool ran, but returned this error message"
- The agent sees the error message and can decide to:
  - Try again with different inputs
  - Give up and return a response to the user
  - Try a different tool
- `tool_call_id=request.tool_call["id"]` links the error message back to the specific tool call

---

### What you SHOULD catch

**Examples of runtime errors to catch:**
- Invalid SQL syntax (input passes as a string, but SQL parser rejects it)
- API 404 errors (the resource doesn't exist)
- Network timeouts from external services
- Malformed expressions in a calculator tool

**Why catch them:** These are **expected failure cases** where you want to give the model a chance to retry or try a different approach.

---

### What you SHOULD NOT catch

**1. Network failures:**
```python
# ❌ DON'T do this — network issues are separate concerns
except ConnectionError:
    return ToolMessage(content="Network error")
```
Use a separate **retry middleware** instead. The agent shouldn't decide retries — a dedicated middleware should.

**2. Incorrect tool implementation bugs:**
```python
# ❌ DON'T do this — bugs should bubble up for debugging
except IndexError:
    return ToolMessage(content="Something went wrong")
```
If your tool code has a bug, let it fail loudly so you can fix it.

**3. Schema mismatch errors:**
```python
# ❌ DON'T do this — the framework already prevents these
except TypeError:
    return ToolMessage(content="Wrong type")
```
The agent framework validates inputs against the tool's schema before calling the tool. Type errors shouldn't happen.

---

### When to register `@wrap_tool_call`

```python
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[check_weather, search_web],
    middleware=[handle_tool_errors]  # ← Register it as middleware
)
```

**Key difference from v0:**
- v0: Error handling was part of `ToolNode` initialization
- v1: Error handling is a middleware, composable with other middlewares (Phase 02B)

---

## 7. Tools Accessing Custom State

Tools sometimes need access to information from the agent's state (not just their own inputs). For example, a tool might need the current user's name or ID to personalise its response.

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

### Breaking down the example

**1. Define custom state:**
```python
class CustomState(AgentState):
    user_name: str
```
- Extends `AgentState` (which is a `TypedDict` with a `messages` field)
- Adds a new field `user_name` of type `str`
- This field will be available in the agent's state throughout execution

**2. Define a tool that uses the state:**
```python
@tool
def greet(
    runtime: ToolRuntime[None, CustomState]
) -> str:
```
- `runtime: ToolRuntime[None, CustomState]` — a special parameter that gives access to the agent's runtime
  - The `CustomState` type hint tells LangChain: "This tool uses custom state of type `CustomState`"
  - The `None` is a type parameter for tool-specific context (advanced, usually not used)

**3. Access the state inside the tool:**
```python
    user_name = runtime.state.get("user_name", "Unknown")
    return f"Hello {user_name}!"
```
- `runtime.state` is a dict-like object containing all the agent's current state
- `.get("user_name", "Unknown")` — safely reads the `user_name` field, defaulting to `"Unknown"` if not set
- Uses the state value to personalise the tool's response

**4. Register the custom state with the agent:**
```python
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[greet],
    state_schema=CustomState
)
```
- `state_schema=CustomState` — tells the agent "my tools might access fields from this `CustomState` type"

---

### How the state gets populated

The state comes from somewhere — either your code initialises it, or middleware populates it. Here's a minimal example of how `user_name` would get into the state:

```python
# At invocation time, you'd initialise the state with user_name
result = agent.invoke({
    "messages": [{"role": "user", "content": "Greet me"}],
    "user_name": "Alice"  # ← This initialises the custom state field
})

# When the greet tool runs:
# - runtime.state.get("user_name") returns "Alice"
# - The tool returns "Hello Alice!"
```

---

### When to use tool state access

**Use it when:**
- A tool needs user metadata (ID, name, account tier)
- A tool needs conversation history or other agent state
- A tool needs to know which sub-agent is running (in multi-agent setups)

**Don't use it when:**
- The tool only needs its own inputs
- The state is very large or changes frequently (consider middleware instead)

---

## 8. Custom State Schema

The agent's default state contains a `messages` field (the conversation history). You can extend this with additional fields for your application's needs.

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

### Breaking down the example

**What is `AgentState`?**
- `AgentState` is a `TypedDict` — a Python type hint for dict-like objects with known keys
- It already has a `messages` field containing the conversation history
- You inherit from it to add more fields specific to your application

```python
class CustomAgentState(AgentState):
    user_id: str
```
- Defines a new state type that extends `AgentState`
- Adds a required field `user_id` of type `str`
- The agent's state will now always have both:
  - `messages` (from `AgentState`)
  - `user_id` (your custom field)

**Why inherit from `AgentState` instead of defining your own `TypedDict`?**
- `AgentState` already includes the message handling that agents need
- Inheriting ensures you have the right structure

```python
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=tools,
    state_schema=CustomAgentState
)
```
- Tells the agent: "Use this state schema for this agent"
- The agent will now expect `user_id` to be present when invoked

---

### What is no longer supported

**Source:** `migrate-complete.md` (Custom state → State type restrictions section)

In v0, you could use Pydantic models or dataclasses for state. **This is no longer allowed in v1.**

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

# ✅ v1 pattern — TypedDict via AgentState ONLY
from langchain.agents import AgentState

class CustomAgentState(AgentState):
    user_id: str
```

### Why the change to TypedDict?

**TypedDict is better for this use case because:**
- It's **lightweight** — no runtime overhead (Pydantic validates, TypedDict doesn't)
- It's **type-hint focused** — you still get IDE autocomplete and type checking
- It **composes better** with the LangGraph state system (which uses `Annotated` dicts)
- **Validation moved to middleware** — if you need validation, implement it in `before_model` or `after_model` hooks (Phase 02B)

---

### How to initialise custom state

```python
# When invoking the agent, pass the state fields
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello"}],
    "user_id": "user_123"  # ← Custom state field
})
```

Or if you're building state programmatically:

```python
state = {
    "messages": [HumanMessage(content="Hello")],
    "user_id": "user_123"
}
result = agent.invoke(state)
```

---

## 9. Runtime Context

### Static vs Dynamic Data

When invoking an agent, you pass **two types of data:**

| Type | Example | Changes? | Where passed |
|---|---|---|---|
| **Dynamic state** | Message history, tool call results | Yes, changes per turn | In `{"messages": [...]}` dict |
| **Static context** | User ID, session ID, user role | No, stays same for entire session | Via `context=Context(...)` parameter |

**Static context is the same throughout the agent's lifetime** — it's metadata about the user or session that doesn't change as the conversation happens.

**Source:** `migrate-complete.md` (Runtime context section)

```python
from dataclasses import dataclass
from langchain.agents import create_agent

@dataclass
class Context:
    user_id: str
    session_id: str
```

### Breaking down the dataclass

```python
@dataclass
class Context:
    user_id: str
    session_id: str
```

**What this does:**
- `@dataclass` is a Python decorator that automatically generates `__init__`, `__repr__`, and other methods
- Defines a simple class with two fields: `user_id` and `session_id`
- This becomes the **schema** for static context you'll pass to the agent

**Why not TypedDict here?**
- `dataclass` is better for context because it's simpler and allows default values
- `TypedDict` is better for state because it integrates with LangGraph's state system
- Different tools for different jobs

---

### Registering context schema on the agent

```python
agent = create_agent(
    model=model,
    tools=tools,
    context_schema=Context
)
```

**What this does:**
- `context_schema=Context` tells the agent "I will pass you a `Context` object when invoking"
- The agent stores the schema for validation and type hints

---

### Passing context at invocation time

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    context=Context(user_id="123", session_id="abc")
)
```

**What's happening:**
- **First arg** `{"messages": [...]}` — the dynamic state for this turn
- **`context=Context(...)`** — the static context for the entire session
  - `user_id="123"` — this user's ID
  - `session_id="abc"` — this conversation session's ID
  - These values will be available inside tools and middleware via `request.runtime.context`

### Accessing context inside tools and middleware

In a tool (Phase 02A §7):
```python
@tool
def get_user_balance(runtime: ToolRuntime[None, CustomState]) -> str:
    """Get the user's account balance."""
    user_id = runtime.context.user_id  # Access context
    balance = db.query(user_id)
    return f"Your balance: ${balance}"
```

In middleware (Phase 02B):
```python
@dynamic_prompt
def my_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id  # Access context
    if is_premium_user(user_id):
        return "You have access to premium features."
    return "You are a free user."
```

---

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

**Why v0 is no longer recommended:**
- `config["configurable"]` is opaque — developers don't know what keys are expected
- `context=Context(...)` is explicit — the dataclass documents what context is needed
- Type checking works better with named context — IDE autocomplete shows available fields

**From your notes:** "The old `config["configurable"]` pattern still works for backward compatibility, but using the new `context` parameter is recommended for new applications or applications migrating to v1."

---

## 10. Dynamic Model Selection

Choose a different language model for each agent call based on runtime conditions (e.g., conversation length, task complexity, cost constraints).

**Source:** `migrate-complete.md` (Model → Dynamic model selection section)

### Implementing dynamic model selection

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware, ModelRequest
)
from langchain.agents.middleware.types import ModelResponse
from langchain_openai import ChatOpenAI
from typing import Callable
```

**What's being imported:**
- `AgentMiddleware` — the base class for custom middleware (Phase 02B)
- `ModelRequest` — the request object passed to `wrap_model_call`, containing agent state and the current model
- `ModelResponse` — the response type from the model
- `ChatOpenAI` — OpenAI chat model (can use Anthropic, etc.)
- `Callable` — type hint for a function that handles the actual model call

```python
basic_model = ChatOpenAI(model="gpt-5-nano")      # cheap, fast
advanced_model = ChatOpenAI(model="gpt-5.4")       # powerful, expensive

class DynamicModelMiddleware(AgentMiddleware):
```

**What this does:**
- Creates two model instances: one cheap (nano) and one powerful (gpt-5.4)
- Defines a custom middleware class to decide which one to use per call

```python
    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
```

**What's happening:**
- `wrap_model_call` is the middleware method that wraps model execution (see Phase 02B §9 for full reference)
- `request: ModelRequest` — contains the current state and the model object
- `handler: Callable` — the function that runs the model with the request (calls the LLM)

```python
        if len(request.state.messages) > self.messages_threshold:
            model = advanced_model
        else:
            model = basic_model
        return handler(request.override(model=model))
```

**What's happening:**
- Checks the number of messages in the conversation state
- If more than threshold → use the advanced model (better for complex conversations)
- If less → use the cheap model (good enough for simple queries)
- `request.override(model=model)` — creates a new request object with the chosen model
- `handler(modified_request)` — runs the model with the new request
- Returns the model's response

```python
    def __init__(self, messages_threshold: int) -> None:
        self.messages_threshold = messages_threshold
```

**What this does:**
- Constructor that stores the threshold value
- Example: threshold of 10 means "use advanced model after 10+ messages in history"

---

### Using dynamic model selection

```python
agent = create_agent(
    model=basic_model,   # default fallback
    tools=tools,
    middleware=[DynamicModelMiddleware(messages_threshold=10)]
)
```

**What's happening:**
- `model=basic_model` — starts with the cheap model by default
- `middleware=[DynamicModelMiddleware(...)]` — middleware decides per call whether to switch models
- `messages_threshold=10` — switch to advanced after 10+ messages

### How it works in practice

```
User asks simple question (5 messages total)
→ DynamicModelMiddleware checks: 5 < 10? Yes
→ Uses basic_model (gpt-5-nano)
→ Fast, cheap response

User asks follow-up after many turns (15 messages total)
→ DynamicModelMiddleware checks: 15 > 10? Yes
→ Switches to advanced_model (gpt-5.4)
→ Better reasoning, handles complexity, costs more
```

---

### When to use dynamic model selection

**Use it when:**
- Simple queries can be handled by a cheap model
- Complex queries need a powerful model
- You want cost optimization (pay for power only when needed)
- You're trying multiple models for A/B testing
- Different user tiers get different models (free tier = cheap, pro tier = advanced)

**Don't use it when:**
- One model handles all your use cases fine
- Model switching adds latency you can't afford

---

## 11. Structured Output

When you need the agent to return a response in a specific JSON structure (not free-form text), use structured output. This is useful for APIs, data extraction, or when downstream systems expect a specific format.

**Source:** `migrate-complete.md` (Structured output → Tool and provider strategies section)

### Defining the output schema

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from pydantic import BaseModel

class OutputSchema(BaseModel):
    summary: str
    sentiment: str
```

**What this does:**
- `OutputSchema` is a Pydantic model that defines the **exact structure** of the agent's response
- `summary: str` — the agent must return a string field called "summary"
- `sentiment: str` — the agent must return a string field called "sentiment"
- When the agent finishes, you get back a dict with these exact fields

### Using ToolStrategy (artificial tool calling)

```python
# Using ToolStrategy (artificial tool calling)
agent = create_agent(
    model="gpt-5.4-mini",
    tools=tools,
    response_format=ToolStrategy(OutputSchema)
)
```

**How ToolStrategy works:**
- Treats the output schema as a "fake tool" the model must call
- The model generates tool arguments that match your schema
- The framework extracts those arguments and returns them as structured data
- Works with any model, not just those with native structured output support

### Using ProviderStrategy (native structured output)

```python
# Using ProviderStrategy (provider-native structured output)
agent = create_agent(
    model="gpt-5.4-mini",
    tools=tools,
    response_format=ProviderStrategy(OutputSchema)
)
```

**How ProviderStrategy works:**
- Uses the model provider's native structured output capability (e.g., OpenAI's `response_format`, Claude's structured output)
- Only works with models that support it
- Usually faster and more reliable than ToolStrategy
- Requires provider support

### When to use each strategy

| Strategy | Pros | Cons |
|---|---|---|
| **ToolStrategy** | Works with any model | Slightly slower, uses a tool slot |
| **ProviderStrategy** | Faster, more reliable | Only works with models that support it |

**Recommendation:** Try ProviderStrategy first. If your model doesn't support it, fall back to ToolStrategy.

---

### What was removed: Prompted output

**Source:** `migrate-complete.md` (Structured output → Prompted output removed section)

In v0, you could use "prompted output" — just telling the model in the prompt to generate a specific format:

```python
# ❌ v0 pattern — no longer supported
agent = create_react_agent(
    model="gpt-5.4-mini",
    tools=tools,
    response_format=("please generate output like this: {...", OutputSchema)
)
```

**Why it was removed:**
- "Prompted output" is unreliable — the model sometimes ignores the instruction
- Tool-based and provider-native strategies are much more reliable
- Not worth the maintenance burden

**What to use instead:** ToolStrategy or ProviderStrategy (above)

---

### Using the structured output

Once you invoke the agent with structured output:

```python
result = agent.invoke({"messages": [{"role": "user", "content": "Analyze this article..."}]})
# result is now a dict like: {"summary": "...", "sentiment": "positive"}

print(result["summary"])    # Access structured fields directly
print(result["sentiment"])
```

---

### Key difference from v0

**In v0:**
- Structured output was generated in a **separate node** after the main agent loop
- Added latency because the agent had to run twice

**In v1:**
- Structured output is generated in the **main loop**
- Same cost and latency as a normal agent call

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
