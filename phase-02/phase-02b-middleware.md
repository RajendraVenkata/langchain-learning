# Phase 02B — LangChain Core: Middleware Deep Dive

> **Level:** Intermediate  
> **Part:** 2 of 3 in Phase 02  
> **Source files:** `migrate-complete.md`  
> **Prerequisite:** Complete Phase 02A first — this file assumes you understand `create_agent` and tools.  
> **Goal:** Fully understand the middleware system — the primary extension point in LangChain v1. By the end of this file you should be able to write any custom middleware from scratch.

---

## Table of Contents

1. [Why Middleware? The Mental Model](#1-why-middleware)
2. [The Middleware Execution Flow](#2-execution-flow)
3. [Built-in Middleware — Ready to Use](#3-built-in-middleware)
4. [Custom Middleware — Building Your Own](#4-custom-middleware)
5. [Custom State via Middleware](#5-custom-state-via-middleware)
6. [Dynamic Model Selection via Middleware](#6-dynamic-model-selection)
7. [Composing Multiple Middlewares](#7-composing-multiple-middlewares)
8. [Middleware Method Reference](#8-method-reference)
9. [Self-Quiz](#9-self-quiz)
10. [Flashcards](#10-flashcards)

---

## 1. Why Middleware?

In v0, customising agent behaviour required hooking into the graph before or after the model node using `pre_model_hook` and `post_model_hook`. These were single functions, which meant you could only have one of each. To compose behaviours (e.g., summarise history AND redact PII), you had to write a single function that did everything.

**Middleware solves this** by making each behaviour a reusable, composable unit. You pass a list of middlewares to `create_agent` and they all run in sequence.

```
v0 architecture (hooks):
  [user message] → pre_model_hook (one function) → [model] → post_model_hook (one function) → [response]

v1 architecture (middleware):
  [user message] → MW1.before_model → MW2.before_model → [model] → MW2.after_model → MW1.after_model → [response]
```

> **Key insight:** Middleware is **not** a decorator pattern like Flask middleware. It is a class-based system where each middleware can intercept the request before AND after the model call, and can also wrap tool calls.

---

## 2. The Execution Flow

Understanding execution order is critical for reasoning about middleware behaviour.

```
invoke(messages, context)
        ↓
  middleware[0].before_model(state, runtime)
        ↓
  middleware[1].before_model(state, runtime)
        ↓
       [MODEL CALL]
        ↓
  middleware[1].after_model(state, runtime)
        ↓
  middleware[0].after_model(state, runtime)
        ↓
   tool execution (each wrapped by middleware[N].wrap_tool_call)
        ↓
   (loop back to before_model if more tool calls)
```

**Order matters:**
- `before_model` runs in **list order** (index 0 first)
- `after_model` runs in **reverse list order** (last index first — like a stack)
- This is the standard "onion" middleware pattern

---

## 3. Built-in Middleware

LangChain v1 ships with several ready-to-use middleware classes. Learn these before building custom ones.

### `SummarizationMiddleware`

**What it does:** Automatically summarises long conversation history when a token threshold is exceeded, keeping the context window manageable.

**When to use:** Any long-running conversational agent where the message history might exceed model limits.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=tools,
    middleware=[
        SummarizationMiddleware(
            model="claude-sonnet-4-6",   # model used to do the summarisation
            trigger={"tokens": 1000}      # summarise when history exceeds 1000 tokens
        )
    ]
)
```

**What replaced:** The old `pre_model_hook=custom_summarization_function` pattern in v0.

---

### `HumanInTheLoopMiddleware`

**What it does:** Pauses agent execution before specific tool calls and waits for human approval. The agent cannot proceed until a human approves or rejects.

**When to use:** Any agent that can take irreversible actions (sending emails, making purchases, deleting records, running SQL writes).

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[read_email, send_email, delete_record],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "description": "Please review this email before sending",
                    "allowed_decisions": ["approve", "reject"]
                },
                "delete_record": {
                    "description": "Confirm record deletion",
                    "allowed_decisions": ["approve", "reject", "modify"]
                }
            }
        )
    ]
)
```

**What replaced:** The old `post_model_hook=custom_human_in_the_loop_hook` pattern in v0.

**HITL deprecation table (v0 → v1):**

| v0 class | v1 replacement |
|---|---|
| `HumanInterruptConfig` | `langchain.agents.middleware.human_in_the_loop.InterruptOnConfig` |
| `ActionRequest` | `langchain.agents.middleware.human_in_the_loop.InterruptOnConfig` |
| `HumanInterrupt` | `langchain.agents.middleware.human_in_the_loop.HITLRequest` |

---

### `@dynamic_prompt` (decorator middleware)

**What it does:** Adapts the system prompt at runtime based on the current conversation state and context. Covered in Phase 02A — included here for completeness.

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    if user_role == "admin":
        return "You are a helpful assistant with full system access."
    return "You are a helpful assistant with read-only access."
```

---

### `@wrap_tool_call` (decorator middleware)

**What it does:** Wraps each tool execution to handle runtime errors. Covered in Phase 02A — included here for completeness.

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )
```

---

## 4. Custom Middleware

When built-in middlewares don't cover your needs, subclass `AgentMiddleware` and implement one or more of its methods.

### Base class structure

```python
from langchain.agents.middleware import AgentMiddleware
from typing import Any

class MyCustomMiddleware(AgentMiddleware):
    
    def before_model(self, state, runtime) -> dict[str, Any] | None:
        """Called before every model call.
        
        Return None to continue normally.
        Return {"jump_to": "end"} to short-circuit and end the agent loop.
        Return a state update dict to modify state before the model sees it.
        """
        return None
    
    def after_model(self, state, runtime) -> dict[str, Any] | None:
        """Called after every model call.
        
        Return None to continue normally.
        Return a state update dict to modify state after the model responds.
        """
        return None
    
    def wrap_tool_call(self, request, handler):
        """Wraps each individual tool call.
        
        Call handler(request) to execute the tool normally.
        You can modify the request before calling handler.
        You can catch exceptions from handler.
        """
        return handler(request)
```

### Example: PII Redaction middleware

This middleware scans user messages for patterns that look like email addresses and redacts them before the model sees them.

```python
import re
from langchain.agents.middleware import AgentMiddleware
from typing import Any

class PIIRedactionMiddleware(AgentMiddleware):
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    
    def before_model(self, state, runtime) -> dict[str, Any] | None:
        """Redact emails from the last user message before the model sees it."""
        messages = state.get("messages", [])
        if not messages:
            return None
        
        last = messages[-1]
        if hasattr(last, "content") and isinstance(last.content, str):
            redacted = self.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", last.content)
            if redacted != last.content:
                # Return a state update — replaces the last message with the redacted version
                updated = last.model_copy(update={"content": redacted})
                return {"messages": messages[:-1] + [updated]}
        return None

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=tools,
    middleware=[PIIRedactionMiddleware()]
)
```

### Example: Request logging middleware

```python
import time
from langchain.agents.middleware import AgentMiddleware
from typing import Any

class LoggingMiddleware(AgentMiddleware):
    
    def before_model(self, state, runtime) -> dict[str, Any] | None:
        self._start_time = time.time()
        message_count = len(state.get("messages", []))
        print(f"[LOG] Model call starting. Messages in state: {message_count}")
        return None
    
    def after_model(self, state, runtime) -> dict[str, Any] | None:
        elapsed = time.time() - self._start_time
        print(f"[LOG] Model call completed in {elapsed:.2f}s")
        return None

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=tools,
    middleware=[LoggingMiddleware()]
)
```

### Short-circuiting the agent loop

Return `{"jump_to": "end"}` from `before_model` to immediately end the agent loop without calling the model. Useful for rate limiting, safety checks, or early exits.

```python
class RateLimitMiddleware(AgentMiddleware):
    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.call_count = 0
    
    def before_model(self, state, runtime):
        self.call_count += 1
        if self.call_count > self.max_calls:
            print(f"[RATE LIMIT] Exceeded {self.max_calls} calls.")
            return {"jump_to": "end"}  # short-circuit
        return None
```

---

## 5. Custom State via Middleware

You can extend the default agent state with additional fields. There are two ways to do this.

### Method 1: Via `state_schema` on `create_agent`

Best when the custom state needs to be accessed by **tools**.

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState

class CustomState(AgentState):   # AgentState is a TypedDict
    user_name: str
    account_tier: str

@tool
def get_personalised_offer(runtime: ToolRuntime[None, CustomState]) -> str:
    """Get a personalised offer for the current user."""
    tier = runtime.state.get("account_tier", "free")
    name = runtime.state.get("user_name", "there")
    if tier == "pro":
        return f"Hi {name}! Here's your exclusive pro offer: 50% off."
    return f"Hi {name}! Upgrade to pro for exclusive offers."

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_personalised_offer],
    state_schema=CustomState
)
```

### Method 2: Via middleware `state_schema` attribute

Best when the custom state is managed **by the middleware itself** (not accessed by tools). This keeps state extensions conceptually scoped to the middleware that owns them.

```python
from langchain.agents.middleware import AgentState, AgentMiddleware
from typing_extensions import NotRequired
from typing import Any

class CustomState(AgentState):
    model_call_count: NotRequired[int]   # NotRequired = optional field with no default

class CallCounterMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState   # associate this state with this middleware
    
    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        count = state.get("model_call_count", 0)
        if count > 10:
            print("[WARN] Too many model calls. Ending agent loop.")
            return {"jump_to": "end"}
        return None
    
    def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        # Increment the counter after each model call
        return {"model_call_count": state.get("model_call_count", 0) + 1}

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[...],
    middleware=[CallCounterMiddleware()]
)
```

### Which method to use?

| Situation | Use |
|---|---|
| State is read/written by tools | `state_schema` on `create_agent` |
| State is only used internally by a middleware | `state_schema` attribute on the middleware class |
| Both | Prefer middleware-scoped state where possible — keeps concerns separated |

> **From your notes:** "Defining custom state via middleware is preferred because it allows you to keep state extensions conceptually scoped to the relevant middleware and tools."

### State type restrictions

`create_agent` only supports `TypedDict` for state schemas. Pydantic models and dataclasses are **not supported**.

```python
# ❌ Pydantic — no longer supported
from pydantic import BaseModel
class MyState(BaseModel):
    user_id: str

# ❌ dataclass — no longer supported
from dataclasses import dataclass
@dataclass
class MyState:
    user_id: str

# ✅ TypedDict via AgentState — this is the correct pattern
from langchain.agents import AgentState
class MyState(AgentState):
    user_id: str
```

> **If you need validation:** Handle it in `before_model` or `after_model` middleware hooks instead of using Pydantic validators.

---

## 6. Dynamic Model Selection

Dynamic model selection lets you choose a different language model for each call based on runtime conditions (e.g., conversation length, task complexity, cost).

In v0, this was done by passing a callable to the `model` parameter. In v1, it is done via the `wrap_model_call` method on a custom middleware.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain_openai import ChatOpenAI
from typing import Callable

basic_model = ChatOpenAI(model="gpt-5-nano")      # cheap and fast
advanced_model = ChatOpenAI(model="gpt-5.4")       # powerful but expensive

class DynamicModelMiddleware(AgentMiddleware):
    
    def __init__(self, messages_threshold: int) -> None:
        self.messages_threshold = messages_threshold
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        # Use the advanced model for longer conversations
        if len(request.state.messages) > self.messages_threshold:
            selected_model = advanced_model
        else:
            selected_model = basic_model
        
        # Override the model on the request before calling the handler
        return handler(request.override(model=selected_model))

agent = create_agent(
    model=basic_model,   # default fallback
    tools=tools,
    middleware=[DynamicModelMiddleware(messages_threshold=10)]
)
```

**Common use cases for dynamic model selection:**
- Use a cheap model for short conversations, premium model for long ones
- Route to a specialist model based on detected topic
- Apply cost constraints per user account tier

---

## 7. Composing Multiple Middlewares

The power of middleware is composition. You can stack multiple middlewares together, and each runs independently.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[read_email, send_email, search_web],
    middleware=[
        # 1. Log all model calls
        LoggingMiddleware(),
        
        # 2. Summarise when history gets long
        SummarizationMiddleware(
            model="claude-sonnet-4-6",
            trigger={"tokens": 2000}
        ),
        
        # 3. Require human approval before sending emails
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": {"allowed_decisions": ["approve", "reject"]}}
        ),
        
        # 4. Handle tool runtime errors
        handle_tool_errors,   # @wrap_tool_call decorated function
    ]
)
```

**Execution order for the above:**
1. `LoggingMiddleware.before_model` → `SummarizationMiddleware.before_model` → model call
2. `HumanInTheLoopMiddleware.after_model` → `SummarizationMiddleware.after_model` → `LoggingMiddleware.after_model`
3. Tool execution wrapped by `handle_tool_errors`

> **Tip:** Put logging/monitoring middleware first in the list so it captures everything. Put transformation middleware (summarisation, PII redaction) before the model call so it processes before the model sees the data.

---

## 8. Method Reference

Full reference of all methods available on `AgentMiddleware`:

| Method | When it runs | Return value | Common uses |
|---|---|---|---|
| `before_model(state, runtime)` | Before every model call | `None` (continue) or `dict` (state update) or `{"jump_to": "end"}` | Input guardrails, summarisation, message trimming, PII redaction |
| `after_model(state, runtime)` | After every model call | `None` (continue) or `dict` (state update) | Output guardrails, HITL approval, logging, counting |
| `wrap_model_call(request, handler)` | Around every model call | result of `handler(request)` or modified result | Dynamic model selection, model-level retry |
| `wrap_tool_call(request, handler)` | Around every tool execution | result of `handler(request)` or `ToolMessage` | Tool error handling, tool-level retry, tool call logging |

---

## 9. Self-Quiz

1. In what order do `before_model` hooks run when you have three middlewares? What about `after_model`?
2. What return value from `before_model` short-circuits the entire agent loop?
3. What is the difference between `SummarizationMiddleware` and `HumanInTheLoopMiddleware`? When would you use each?
4. You want to redact phone numbers from user messages before the model sees them. Which middleware method do you implement?
5. You want to log how long every model call takes. Which two methods do you implement?
6. When should you define custom state via `state_schema` on `create_agent` vs. via the `state_schema` attribute on the middleware class?
7. You want to use `gpt-5-nano` for conversations under 5 messages and `gpt-5.4` for longer ones. Which middleware method do you implement, and how?
8. Can you validate custom state fields using Pydantic validators inside a middleware state schema?
9. What is `NotRequired` used for in a custom state TypedDict?
10. Write the stub for a middleware class that counts tool calls and logs when a tool is called more than 3 times.

---

## 10. Flashcards

| # | Question | Answer |
|---|---|---|
| 1 | What replaced `pre_model_hook` in v1? | `before_model` method on `AgentMiddleware` |
| 2 | What replaced `post_model_hook` in v1? | `after_model` method on `AgentMiddleware` |
| 3 | What does `before_model` return to short-circuit the agent loop? | `{"jump_to": "end"}` |
| 4 | In what order do `after_model` hooks run relative to the middleware list? | Reverse order (last in list runs first — like a stack) |
| 5 | What built-in middleware handles conversation length? | `SummarizationMiddleware` |
| 6 | What built-in middleware pauses for human approval? | `HumanInTheLoopMiddleware` |
| 7 | Which middleware method is used for dynamic model selection? | `wrap_model_call` |
| 8 | Which middleware method wraps tool execution? | `wrap_tool_call` |
| 9 | Where should state that's only used by a middleware live? | As `state_schema` attribute on the middleware class |
| 10 | What type must custom state schemas be? | `TypedDict` (via `AgentState` inheritance) |
| 11 | What does `NotRequired[int]` mean in a state TypedDict? | The field is optional and has no required default |
| 12 | What does `request.override(model=new_model)` do? | Creates a new request object with a different model, used in `wrap_model_call` |
| 13 | Can you have more than one middleware in the list? | Yes — they compose and run in sequence |
| 14 | What should you put first in the middleware list? | Logging/monitoring middleware (so it captures everything) |
| 15 | What two things does middleware enable that single hooks couldn't? | Reusability across agents, and composition of multiple behaviours |

---

> **Next in Phase 02:** [Phase 02C — Messages, Chat Models & Tutorials](./phase-02c-messages-and-tutorials.md)  
> **Previous:** [Phase 02A — Agents & Tools](./phase-02a-agents-and-tools.md)  
> **Source notes:** `migrate-complete.md` (all middleware patterns and examples)
