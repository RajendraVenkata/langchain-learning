# Phase 02B — LangChain Core: Middleware Deep Dive

> **Level:** Intermediate  
> **Part:** 2 of 3 in Phase 02  
> **Source files:** `migrate-complete.md` · `langchain-middleware-complete.md`  
> **Prerequisite:** Complete Phase 02A first.
>
> ⚠️ **Note on examples:** Every code example in this file is extracted directly from your source files with inline source citations.

---

## Table of Contents

1. [Why Middleware?](#1-why-middleware)
2. [Middleware Execution Flow](#2-middleware-execution-flow)
3. [Built-in Middleware — Ready to Use](#3-built-in-middleware)
4. [Custom State via Middleware](#4-custom-state-via-middleware)
5. [Dynamic Model Selection](#5-dynamic-model-selection)
6. [Tool Error Handling via Middleware](#6-tool-error-handling-via-middleware)
7. [Dynamic Prompts via Middleware](#7-dynamic-prompts-via-middleware)
8. [Composing Multiple Middlewares](#8-composing-multiple-middlewares)
9. [Full Method Reference](#9-full-method-reference)
10. [Self-Quiz](#10-self-quiz)
11. [Flashcards](#11-flashcards)

---

## 1. Why Middleware?

In v0, you had `pre_model_hook` and `post_model_hook` — single functions that ran before and after the model. In v1, middleware replaces this with a class-based, composable system where each behaviour is a reusable unit.

**From `migrate-complete.md`:**

> Pre-model hooks are now implemented as middleware with the `before_model` method. This new pattern is more extensible — you can define multiple middlewares to run before the model is called, reusing common patterns across different agents.

---

## 2. Middleware Execution Flow

```
invoke(messages, context)
        ↓
  [all middleware].before_model() in list order
        ↓
       [MODEL CALL]
        ↓
  [all middleware].after_model() in reverse list order
        ↓
   tool execution (wrapped by each middleware's wrap_tool_call)
        ↓
   (loop back to before_model if more tool calls)
```

---

## 3. Built-in Middleware

### SummarizationMiddleware

**Source:** `migrate-complete.md` (Pre-model hook section)

Automatically summarises long conversation history when a token threshold is exceeded.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=tools,
    middleware=[
        SummarizationMiddleware(
            model="claude-sonnet-4-6",
            trigger={"tokens": 1000}
        )
    ]
)
```

### HumanInTheLoopMiddleware

**Source:** `migrate-complete.md` (Post-model hook section)

Pauses agent execution before specific tool calls and waits for human approval.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[read_email, send_email],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "description": "Please review this email before sending",
                    "allowed_decisions": ["approve", "reject"]
                }
            }
        )
    ]
)
```

---

## 4. Custom State via Middleware

### Method 1: Via `state_schema` on `create_agent`

**Source:** `migrate-complete.md` (Custom state → Defining state via state_schema section)

Use when custom state needs to be accessed by **tools**:

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

### Method 2: Via `state_schema` attribute on the middleware class

**Source:** `migrate-complete.md` (Custom state → Defining state via middleware section)

Use when custom state is managed **only by the middleware itself**:

```python
from langchain.agents.middleware import AgentState, AgentMiddleware
from typing_extensions import NotRequired
from typing import Any

class CustomState(AgentState):
    model_call_count: NotRequired[int]

class CallCounterMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        count = state.get("model_call_count", 0)
        if count > 10:
            return {"jump_to": "end"}
        return None

    def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        return {"model_call_count": state.get("model_call_count", 0) + 1}

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[...],
    middleware=[CallCounterMiddleware()]
)
```

---

## 5. Dynamic Model Selection

**Source:** `migrate-complete.md` (Model → Dynamic model selection section)

Choose different models based on runtime conditions (e.g., conversation length):

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

## 6. Tool Error Handling via Middleware

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

## 7. Dynamic Prompts via Middleware

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

## 8. Composing Multiple Middlewares

Stack multiple middlewares — each runs independently in sequence:

**Implied from `migrate-complete.md` (multiple middleware examples)**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[read_email, send_email, search_web],
    middleware=[
        # 1. Summarise when history gets long
        SummarizationMiddleware(
            model="claude-sonnet-4-6",
            trigger={"tokens": 2000}
        ),
        
        # 2. Require human approval before sending emails
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": {"allowed_decisions": ["approve", "reject"]}}
        ),
        
        # 3. Handle tool runtime errors
        handle_tool_errors,   # @wrap_tool_call decorated function
    ]
)
```

---

## 9. Full Method Reference

All methods available on `AgentMiddleware`:

| Method | When it runs | Return value | Purpose |
|---|---|---|---|
| `before_model(state, runtime)` | Before every model call | `None` (continue) or `dict` (state update) or `{"jump_to": "end"}` (short-circuit) | Input guardrails, summarisation, message trimming |
| `after_model(state, runtime)` | After every model call | `None` (continue) or `dict` (state update) | Output guardrails, HITL approval, logging |
| `wrap_model_call(request, handler)` | Around every model call | result of `handler(request)` or modified result | Dynamic model selection, model-level retry |
| `wrap_tool_call(request, handler)` | Around every tool execution | result of `handler(request)` or `ToolMessage` | Tool error handling, tool-level retry |

---

## 10. Self-Quiz

1. In what order do `before_model` hooks run when three middlewares are stacked?
2. In what order do `after_model` hooks run relative to the middleware list?
3. What return value from `before_model` immediately ends the agent loop?
4. What is the difference between `SummarizationMiddleware` and `HumanInTheLoopMiddleware`?
5. When should you define custom state via `state_schema` on `create_agent` vs. via the middleware class attribute?
6. What two parameters does `SummarizationMiddleware` require?
7. What three keys can be in the `interrupt_on` dict for `HumanInTheLoopMiddleware`?
8. What method on `AgentMiddleware` implements dynamic model selection?
9. What does `request.override(model=new_model)` do?
10. Can you have more than one middleware in a single agent?
11. Which middleware method wraps tool execution?
12. What does the `@dynamic_prompt` decorator do?
13. When `after_model` returns `{"jump_to": "end"}`, what happens?
14. What is the signature of `wrap_tool_call`?
15. How do you access the current conversation state inside a middleware method?

---

## 11. Flashcards

| # | Question | Answer |
|---|---|---|
| 1 | What replaced `pre_model_hook`? | `before_model` method on `AgentMiddleware` |
| 2 | What replaced `post_model_hook`? | `after_model` method on `AgentMiddleware` |
| 3 | What return value short-circuits the agent loop from `before_model`? | `{"jump_to": "end"}` |
| 4 | Order of `before_model` hooks in a 3-middleware stack? | List order: middleware[0] → middleware[1] → middleware[2] |
| 5 | Order of `after_model` hooks in a 3-middleware stack? | Reverse order: middleware[2] → middleware[1] → middleware[0] |
| 6 | Which built-in middleware handles conversation length? | `SummarizationMiddleware` |
| 7 | Which built-in middleware pauses for human approval? | `HumanInTheLoopMiddleware` |
| 8 | Which middleware method handles dynamic model selection? | `wrap_model_call` |
| 9 | Which middleware method wraps tool execution? | `wrap_tool_call` |
| 10 | When should you use `state_schema` on `create_agent`? | When tools need to access the custom state |
| 11 | When should you use `state_schema` on the middleware class? | When only the middleware manages that state |
| 12 | What type must custom state be? | `TypedDict` (via `AgentState` inheritance) |
| 13 | What does `NotRequired[int]` mean in state? | The field is optional with no required default |
| 14 | What is `@dynamic_prompt`? | Decorator for middleware that adapts the system prompt per call |
| 15 | Can you chain multiple middlewares? | Yes — they compose and run in sequence |

---

> **Next in Phase 02:** [Phase 02C — Messages, Chat Models & Tutorials](./phase-02c-messages-and-tutorials.md)  
> **Previous:** [Phase 02A — Agents & Tools](./phase-02a-agents-and-tools.md)  
> **Source files used:** `migrate-complete.md` (all middleware patterns cited inline)
