# Phase 02B — LangChain Core: Middleware Deep Dive

> **Level:** Intermediate  
> **Part:** 2 of 3 in Phase 02  
> **Source files:** `migrate-complete.md` · `langchain-middleware-complete.md`  
> **Prerequisite:** Complete Phase 02A first.
>
> ⚠️ **Note on examples:** Every example shows explanation + sourced code + complete runnable code.

---

## Table of Contents

1. [Why Middleware?](#1-why-middleware)
2. [Middleware Execution Flow](#2-middleware-execution-flow)
3. [Built-in Middleware — SummarizationMiddleware](#3-summarizationmiddleware)
4. [Built-in Middleware — HumanInTheLoopMiddleware](#4-humaninthellopmiddleware)
5. [Built-in Middleware — Dynamic Prompts](#5-dynamic-prompts-via-middleware)
6. [Custom State via Middleware](#6-custom-state-via-middleware)
7. [Dynamic Model Selection (Review)](#7-dynamic-model-selection-review)
8. [Tool Error Handling (Review)](#8-tool-error-handling-review)
9. [Composing Multiple Middlewares](#9-composing-multiple-middlewares)
10. [Full Method Reference](#10-full-method-reference)
11. [Self-Quiz](#11-self-quiz)
12. [Flashcards](#12-flashcards)

---

## 1. Why Middleware?

In v0, you had only `pre_model_hook` and `post_model_hook` — single functions. In v1, **middleware** lets you compose multiple behaviours.

**Source:** `migrate-complete.md` (Pre/post model hook sections)

### Explanation

**v0 problem:**
```python
def my_hook(state):
    # Summarise messages AND redact PII AND log calls
    # All in ONE function — messy!
    pass

agent = create_react_agent(..., pre_model_hook=my_hook)
```

**v1 solution:**
```python
class SummarizationMiddleware(AgentMiddleware): ...
class PIIRedactionMiddleware(AgentMiddleware): ...
class LoggingMiddleware(AgentMiddleware): ...

agent = create_agent(
    ...,
    middleware=[
        SummarizationMiddleware(),
        PIIRedactionMiddleware(),
        LoggingMiddleware()
    ]
)
```

Each middleware is **reusable** and **testable** independently.

---

## 2. Middleware Execution Flow

Understanding execution order is critical.

```
invoke(messages, context)
        ↓
  middleware[0].before_model() → middleware[1].before_model()
        ↓
       [MODEL CALL]
        ↓
  middleware[1].after_model() → middleware[0].after_model() (reversed!)
        ↓
   tool execution (each wrapped by middleware.wrap_tool_call)
        ↓
   (loop back if more tool calls)
```

**Key insight:**
- `before_model` runs in **list order** (0 → 1 → 2)
- `after_model` runs in **reverse order** (2 → 1 → 0) — like a stack

### Complete Runnable Code

```python
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
```

---

## 3. SummarizationMiddleware

Automatically summarises conversation history when it exceeds a token threshold.

**Source:** `migrate-complete.md` (Pre-model hook section)

### Explanation

Long conversations use many tokens, filling the model's context window. This middleware:
1. **Monitors** message token count
2. **Summarises** old messages when threshold exceeded
3. **Keeps** recent messages (details matter more)
4. **Result**: Unlimited conversation length without context overflow

### Complete Runnable Code

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool

@tool
def answer_question(q: str) -> str:
    """Answer a question."""
    return f"Answer to '{q}': ..."

# Create agent with summarization
agent = create_agent(
    model="gpt-5.4",
    tools=[answer_question],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4",  # Model used for summarisation
            trigger={"tokens": 1000}  # Summarise when >1000 tokens
        )
    ],
    system_prompt="You are a helpful Q&A assistant."
)

# Simulate a long conversation (20 turns)
messages = []
for i in range(20):
    messages.append({"role": "user", "content": f"Question {i}: Tell me about topic {i}"})
    messages.append({"role": "assistant", "content": f"Answer {i}: Here's information about topic {i}"})

# Invoke with long history
result = agent.invoke({
    "messages": messages  # 40 messages total — lots of tokens!
})

print(result)
# The middleware automatically summarised old messages
# The model received: "Summary of earlier conversation: ..." + recent messages
```

### How It Works

```
Initial state: messages = [Q1, A1, Q2, A2, ..., Q20, A20]
Token count: 5000 tokens (way over 1000 threshold!)
    ↓
SummarizationMiddleware.before_model runs
    ↓
Calls summarisation model: "Summarise messages 1-35"
Gets back: "Summary: User and assistant discussed topics 1-18..."
    ↓
Replaces old messages: [SUMMARY_MESSAGE, Q19, A19, Q20, A20]
Token count: 800 tokens (under threshold!)
    ↓
Passes shortened state to model
```

---

## 4. HumanInTheLoopMiddleware

Pauses agent execution before specific tool calls, requiring human approval.

**Source:** `migrate-complete.md` (Post-model hook section)

### Explanation

For **irreversible actions** (send email, delete data, publish), you need human approval:

1. Agent decides to call `send_email`
2. Middleware **intercepts** the call
3. Shows human the email content
4. Human clicks "approve" or "reject"
5. If approved: email sends. If rejected: agent tries different approach.

### Complete Runnable Code

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool

# Tools that need approval
@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to someone."""
    return f"Email sent to {recipient}: {subject}"

@tool
def delete_file(filename: str) -> str:
    """Delete a file (irreversible!)."""
    return f"File deleted: {filename}"

@tool
def read_file(filename: str) -> str:
    """Read a file (safe, no approval needed)."""
    return f"Contents of {filename}: ..."

# Create agent with HITL
agent = create_agent(
    model="gpt-5.4",
    tools=[send_email, delete_file, read_file],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "description": "Please review this email before sending:",
                    "allowed_decisions": ["approve", "reject"]
                },
                "delete_file": {
                    "description": "Confirm file deletion (this is irreversible):",
                    "allowed_decisions": ["approve", "reject"]
                }
                # Note: read_file is NOT in interrupt_on, so it runs without approval
            }
        )
    ],
    system_prompt="You are a helpful assistant."
)

# Invoke
result = agent.invoke({
    "messages": [{"role": "user", "content": "Send an email to alice@example.com saying hello"}]
})

# What happens:
# 1. Agent decides to call send_email(...)
# 2. Middleware shows: "Please review this email before sending:"
# 3. Human sees email content and decides
# 4. If "approve": email_sends, agent continues
# 5. If "reject": email NOT sent, agent gets message "User rejected" and can try alternatives

print(result)
```

### When to Require Approval

| Operation | Require Approval? | Why |
|---|---|---|
| Send email | ✅ YES | Irreversible, affects user |
| Delete file | ✅ YES | Irreversible, data loss |
| Publish to social media | ✅ YES | Public, hard to undo |
| Write to database | ✅ YES | Irreversible change |
| Read file | ❌ NO | Read-only, user sees result |
| Search web | ❌ NO | Read-only, low risk |
| Summarise document | ❌ NO | Analysis only, no side effects |
| Calculate numbers | ❌ NO | Math operation, reversible |

---

## 5. Dynamic Prompts via Middleware

**This was covered in Phase 02A §4. Review that section for complete code.**

Brief recap:

```python
@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    if user_role == "admin":
        return "You have full access"
    return "You have limited access"

agent = create_agent(..., middleware=[role_based_prompt], context_schema=Context)
```

---

## 6. Custom State via Middleware

Define state that's managed by a specific middleware.

**Source:** `migrate-complete.md` (Custom state → Defining state via middleware section)

### Explanation

State can live in two places:
1. **Global state** (via `state_schema` on `create_agent`) — accessed by tools and middleware
2. **Middleware-scoped state** (via `state_schema` on middleware class) — only used by that middleware

Use #2 when the state is **only relevant to one middleware**.

### Complete Runnable Code

```python
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware
from typing_extensions import NotRequired
from typing import Any

# Step 1: Define state that this middleware uses
class CallCounterState(AgentState):
    model_call_count: NotRequired[int]  # Optional field, no default

# Step 2: Create middleware with its own state
class CallCounterMiddleware(AgentMiddleware[CallCounterState]):
    state_schema = CallCounterState  # ← Associate state with this middleware
    
    def before_model(self, state: CallCounterState, runtime) -> dict[str, Any] | None:
        count = state.get("model_call_count", 0)
        print(f"Model call #{count + 1}")
        
        if count > 10:
            print("Too many model calls! Stopping.")
            return {"jump_to": "end"}  # Short-circuit
        return None
    
    def after_model(self, state: CallCounterState, runtime) -> dict[str, Any] | None:
        count = state.get("model_call_count", 0)
        return {"model_call_count": count + 1}  # Increment counter

# Step 3: Create agent with the middleware
agent = create_agent(
    model="gpt-5-mini",
    tools=[],
    middleware=[CallCounterMiddleware()],
    system_prompt="You are helpful."
)

# Step 4: Invoke
result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

# Output:
# Model call #1
# [Agent executes]
# Model call #2
# [Agent executes]
# ... up to Model call #11
# Too many model calls! Stopping.
```

### When to Use Middleware-Scoped State

**Use middleware-scoped state when:**
- The state field is only used by one middleware
- You want to keep concerns separated
- The state is internal bookkeeping (call counts, cache, etc.)

**Use global state when:**
- Tools need access to the field
- Multiple middlewares need the same field
- It's user-facing metadata (user_id, tier, etc.)

---

## 7. Dynamic Model Selection (Review)

**This was covered in Phase 02A §10.** Middleware wraps the model call and chooses which model to use.

```python
class DynamicModelMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        if len(request.state.messages) > 10:
            return handler(request.override(model=powerful_model))
        return handler(request.override(model=cheap_model))
```

---

## 8. Tool Error Handling (Review)

**This was covered in Phase 02A §6.** Use `@wrap_tool_call` to catch runtime errors.

```python
@wrap_tool_call
def handle_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(content=f"Error: {e}", tool_call_id=request.tool_call["id"])
```

---

## 9. Composing Multiple Middlewares

Stack middlewares together. Each runs independently.

### Complete Runnable Code

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    AgentMiddleware
)
from typing import Any

# Custom logging middleware
class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state, runtime) -> dict[str, Any] | None:
        print(f"[LOG] About to call model with {len(state.get('messages', []))} messages")
        return None
    
    def after_model(self, state, runtime) -> dict[str, Any] | None:
        print("[LOG] Model call completed")
        return None

# Stack them all
agent = create_agent(
    model="gpt-5.4",
    tools=[send_email, delete_file],
    middleware=[
        # 1. Log everything
        LoggingMiddleware(),
        
        # 2. Summarise if conversation gets long
        SummarizationMiddleware(
            model="gpt-5.4",
            trigger={"tokens": 2000}
        ),
        
        # 3. Require approval for risky operations
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {"description": "Review email:", "allowed_decisions": ["approve", "reject"]},
                "delete_file": {"description": "Confirm deletion:", "allowed_decisions": ["approve", "reject"]}
            }
        )
    ],
    system_prompt="You are a helpful assistant."
)

# When agent runs:
# 1. LoggingMiddleware.before_model → "About to call model..."
# 2. SummarizationMiddleware.before_model → Summarises if needed
# 3. Model runs
# 4. SummarizationMiddleware.after_model
# 5. LoggingMiddleware.after_model → "Model call completed"
# 6. Tool execution with HumanInTheLoopMiddleware checks
```

### Execution Order Diagram

```
invoke()
  ↓
LoggingMiddleware.before_model() ──→ logs start
  ↓
SummarizationMiddleware.before_model() ──→ summarises if needed
  ↓
[MODEL CALL]
  ↓
SummarizationMiddleware.after_model() ──→ no-op (just passes through)
  ↓
LoggingMiddleware.after_model() ──→ logs completion
  ↓
Tool execution (e.g., send_email)
  ↓
HumanInTheLoopMiddleware.wrap_tool_call() ──→ pauses for approval
  ↓
Result returned
```

---

## 10. Full Method Reference

All available methods on `AgentMiddleware`:

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ModelResponse
from typing import Callable, Any

class YourMiddleware(AgentMiddleware):
    # Called before every model invocation
    def before_model(
        self,
        state: dict[str, Any],
        runtime
    ) -> dict[str, Any] | None:
        """
        Return None to continue normally.
        Return dict to update state before model.
        Return {"jump_to": "end"} to short-circuit.
        """
        pass
    
    # Called after every model response
    def after_model(
        self,
        state: dict[str, Any],
        runtime
    ) -> dict[str, Any] | None:
        """
        Return None to continue normally.
        Return dict to update state after model.
        """
        pass
    
    # Wraps every tool execution
    def wrap_tool_call(
        self,
        request,  # Contains tool call info
        handler: Callable  # Executes the tool
    ):
        """
        Call handler(request) to execute tool normally.
        Catch exceptions, modify request, log events, etc.
        """
        return handler(request)
    
    # Wraps every model call (advanced)
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """
        Call handler(request) to invoke model normally.
        Use request.override(model=new_model) to change model.
        """
        return handler(request)
```

---

## 11. Self-Quiz

1. What is the execution order of `before_model` hooks with 3 middlewares?
2. What is the execution order of `after_model` hooks with 3 middlewares?
3. What return value from `before_model` short-circuits the agent?
4. When should you use `state_schema` on middleware vs on `create_agent`?
5. What is `NotRequired[int]` in a state TypedDict?
6. How does `SummarizationMiddleware` prevent context overflow?
7. What are the two main parameters of `HumanInTheLoopMiddleware`?
8. Can you have more than one middleware?
9. What does `request.override(model=x)` do?
10. What is the difference between middleware-scoped state and global state?

---

## 12. Flashcards

| # | Question | Answer |
|---|---|---|
| 1 | Execution order of `before_model` with 3 middlewares? | List order: 0 → 1 → 2 |
| 2 | Execution order of `after_model` with 3 middlewares? | Reverse order: 2 → 1 → 0 |
| 3 | What return value short-circuits the agent? | `{"jump_to": "end"}` from `before_model` |
| 4 | When to use middleware-scoped state? | When only that middleware uses it |
| 5 | When to use global state? | When tools or multiple middlewares need it |
| 6 | What does `SummarizationMiddleware` do? | Summarises conversation when token threshold exceeded |
| 7 | What does `HumanInTheLoopMiddleware` do? | Pauses for approval before risky tool calls |
| 8 | What does `LoggingMiddleware` do? | Example: logs model call events |
| 9 | What is `NotRequired[T]`? | Optional field with no required default value |
| 10 | Can you stack multiple middlewares? | Yes — they run in sequence, each independently |
| 11 | What method wraps tool execution? | `wrap_tool_call(request, handler)` |
| 12 | What method wraps model execution? | `wrap_model_call(request, handler)` |
| 13 | What does `handler(request)` do? | Executes the wrapped operation (tool or model call) |
| 14 | How to modify state in `after_model`? | Return `{"field": new_value}` dict |
| 15 | What is the base class for custom middleware? | `AgentMiddleware` |

---

> **Next in Phase 02:** [Phase 02C — Messages, Chat Models & Tutorials](./phase-02c-messages-and-tutorials.md)  
> **Previous:** [Phase 02A — Agents & Tools](./phase-02a-agents-and-tools.md)  
> **Source files used:** `migrate-complete.md`
