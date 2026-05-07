# Phase 02A — LangChain Core: Agents & Tools

> **Level:** Beginner → Intermediate  
> **Part:** 1 of 3 in Phase 02  
> **Source files:** `migrate-complete.md` · `oss-complete.md`  
> **Goal:** Understand how to build agents with `create_agent`, define tools with `@tool`, handle tool errors, and manage runtime context.
>
> ⚠️ **Note on examples:** Every code example is sourced from your files with citations. Each section shows: (1) concept explanation, (2) sourced code snippet breakdown, (3) complete runnable code.

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

The function and its location both changed in v1.

**Source:** `migrate-complete.md` (Import path section)

```python
# v0 (old) — DEPRECATED
from langgraph.prebuilt import create_react_agent

# v1 (new) — USE THIS
from langchain.agents import create_agent
```

**Explanation:**
- In v0, agents came from `langgraph.prebuilt` module
- In v1, they moved to `langchain.agents` — the main package
- The function was also renamed: `create_react_agent` → `create_agent` (shorter, clearer)

---

## 2. Basic Agent Pattern

An agent is a loop that repeatedly calls a model and tools until it produces a final answer.

**Source:** `migrate-complete.md` (Prompts → Static prompt section)

### Explanation

When you create an agent, you specify:
1. **`model`** — which LLM to use
2. **`tools`** — what functions the agent can call
3. **`system_prompt`** — instructions for the model

The agent does the rest: it calls the model, detects tool calls, executes tools, and loops until done.

### Complete Runnable Code

```python
from langchain.agents import create_agent
from langchain.tools import tool

# Step 1: Define a tool
@tool
def check_weather(city: str) -> str:
    """Check the current weather for a city.
    
    Args:
        city: The name of the city (e.g., 'Mumbai', 'San Francisco')
    
    Returns:
        A string describing the weather
    """
    # In a real app, this would call a weather API
    return f"The weather in {city} is sunny, 25°C."

# Step 2: Create the agent
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[check_weather],
    system_prompt="You are a helpful weather assistant. Answer weather questions using the check_weather tool."
)

# Step 3: Invoke the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Mumbai?"}]
})

print(result)
# Output: The agent will call check_weather("Mumbai"), get the result, and respond to the user
```

---

## 3. System Prompts — Static

A static system prompt is a fixed string that tells the model how to behave.

**Source:** `migrate-complete.md` (Prompts → Static prompt section)

### Explanation

The `system_prompt` parameter (renamed from `prompt` in v0) is where you define the model's role and instructions:
- "You are a helpful assistant"
- "You are a SQL expert who writes safe queries"
- "You are a data analyst who explains findings clearly"

The model sees this system prompt at the start of every conversation.

### What changed from v0

In v0, you could pass a `SystemMessage` object. In v1, just use a plain string.

### Complete Runnable Code

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

# v1 (correct) — use plain string
agent = create_agent(
    model="gpt-5-mini",
    tools=[calculate],
    system_prompt="You are a helpful math tutor. When users ask for calculations, use the calculate tool. Explain the steps clearly."
)

# Invoke it
result = agent.invoke({
    "messages": [{"role": "user", "content": "Calculate 2 + 2 * 3"}]
})

print(result)
# The agent receives the system_prompt and answers: "2 + 2 * 3 = 8 (because multiplication happens first)"
```

---

## 4. System Prompts — Dynamic

Dynamic prompts adapt based on runtime context (e.g., user role, conversation state).

**Source:** `migrate-complete.md` (Dynamic prompts section)

### Explanation

Sometimes you need **different instructions** depending on **who's using the agent** or **what's happening in the conversation**:
- Admin users get: "You have full access to all data"
- Regular users get: "You can only see data you own"
- After many questions: "You have complex context, provide detailed responses"

Instead of hardcoding the prompt, use a **dynamic prompt function** that generates it at runtime.

### The Mental Model

```
User calls agent with context=Context(user_role="admin")
    ↓
Dynamic prompt function runs
    ↓
Reads request.runtime.context.user_role → "admin"
    ↓
Generates: "You are a helpful assistant with full system access"
    ↓
Model receives this generated prompt + user message
    ↓
Responds as admin
```

### Complete Runnable Code

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.tools import tool

# Step 1: Define context schema
@dataclass
class Context:
    user_role: str = "user"  # Default role

# Step 2: Define a sample tool
@tool
def get_report(report_id: str) -> str:
    """Get a report by ID."""
    return f"Report {report_id}: Confidential data..."

# Step 3: Define dynamic prompt function
@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    """Generate different prompts based on user role."""
    user_role = request.runtime.context.user_role
    base_prompt = "You are a helpful assistant."
    
    if user_role == "admin":
        return f"{base_prompt} You have full access to all reports and data. Provide detailed insights."
    elif user_role == "viewer":
        return f"{base_prompt} You can read reports but cannot make changes. Be helpful but cautious."
    else:
        return base_prompt

# Step 4: Create agent with dynamic prompt middleware
agent = create_agent(
    model="gpt-5.4",
    tools=[get_report],
    middleware=[role_based_prompt],
    context_schema=Context
)

# Step 5: Invoke with different contexts
# As admin
result_admin = agent.invoke(
    {"messages": [{"role": "user", "content": "Show me all reports"}]},
    context=Context(user_role="admin")
)
print("Admin result:", result_admin)

# As viewer
result_viewer = agent.invoke(
    {"messages": [{"role": "user", "content": "Show me all reports"}]},
    context=Context(user_role="viewer")
)
print("Viewer result:", result_viewer)
# The agent gives different responses based on the dynamically generated prompt
```

### Step-by-Step Breakdown

**1. Context Schema** — stores the info that determines the prompt
```python
@dataclass
class Context:
    user_role: str = "user"
```

**2. Dynamic Prompt Function** — generates the prompt based on context
```python
@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role  # Extract role
    # Return different prompt for each role
    if user_role == "admin":
        return "...admin prompt..."
```

**3. Register the Middleware** — tell the agent to use this dynamic prompt
```python
agent = create_agent(
    ...,
    middleware=[role_based_prompt],
    context_schema=Context
)
```

**4. Pass Context at Invocation** — specify the user's role
```python
agent.invoke(..., context=Context(user_role="admin"))
```

---

## 5. Tools — Definition and Usage

Tools are functions the agent can call. The agent looks at the tool's docstring and type hints to understand what it does.

**Source:** `migrate-complete.md` (Tools section)

### Explanation

When you pass tools to `create_agent`, the framework:
1. **Reads the docstring** — this becomes the tool's description (tells the model what it does)
2. **Reads the type hints** — these define the input schema (what parameters are needed)
3. **Makes them available** — the model can call them by name

The framework automatically executes tools and returns results to the model.

### Complete Runnable Code

```python
from langchain.agents import create_agent
from langchain.tools import tool

# Option 1: Using @tool decorator (recommended)
@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic.
    
    This tool searches the internet and returns relevant results.
    """
    # Real implementation would call an API
    return f"Results for '{query}': ...found 10 matches..."

@tool
def check_weather(city: str) -> str:
    """Get the current weather in a city."""
    return f"Weather in {city}: Sunny, 25°C"

# Option 2: Plain function with type hints and docstring
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Create agent with all three tools
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[
        search_web,         # @tool decorated
        check_weather,      # @tool decorated
        calculate_sum       # Plain function
    ],
    system_prompt="You are a helpful assistant. Use tools to answer questions accurately."
)

# Invoke
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Paris and what's 5+3?"}]
})

print(result)
# The agent will:
# 1. Call check_weather("Paris") → gets weather
# 2. Call calculate_sum(5, 3) → gets 8
# 3. Responds with both pieces of information
```

### What Tools Parameter Accepts

```python
# 1. @tool-decorated functions ✅
@tool
def my_tool() -> str:
    """Do something."""
    return "result"

# 2. Plain callables with type hints ✅
def plain_function(x: int) -> str:
    """Do something."""
    return f"Result: {x}"

# 3. BaseTool subclass ✅
from langchain.tools import BaseTool

class CustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "Does something"
    def _run(self, input: str) -> str:
        return f"Result: {input}"

agent = create_agent(
    model="...",
    tools=[my_tool, plain_function, CustomTool()]
)
```

### What is NO Longer Accepted

```python
# ❌ v0 pattern — ToolNode not allowed
from langgraph.prebuilt import ToolNode
agent = create_agent(
    model="...",
    tools=ToolNode([tool1, tool2])  # ERROR in v1
)

# ✅ v1 pattern — just use a list
agent = create_agent(
    model="...",
    tools=[tool1, tool2]  # ✅ Correct
)
```

---

## 6. Tool Error Handling

When tools fail at runtime (invalid input, API error, etc.), you can catch those errors and return custom messages to the agent instead of crashing.

**Source:** `migrate-complete.md` (Tools → Handling tool errors section)

### Explanation

Tool errors are **different from schema errors**:
- **Schema errors** — input doesn't match the type hint. Framework prevents these automatically.
- **Runtime errors** — input is valid, but the tool fails when executed (e.g., invalid SQL, API returns 404, network timeout).

Use `@wrap_tool_call` middleware to handle runtime errors gracefully.

### What Errors to Catch vs. Not Catch

| Error Type | Catch? | Why |
|---|---|---|
| Invalid SQL syntax (input passed validation but SQL parser fails) | ✅ YES | Agent can retry with better syntax |
| Missing API key (input is valid but auth fails) | ✅ YES | Agent can handle gracefully |
| Division by zero in calculator | ✅ YES | Agent can understand the constraint |
| Network timeout | ❌ NO | Use retry middleware instead |
| Bug in your tool code (KeyError, AttributeError) | ❌ NO | Let it crash so you can fix the bug |
| Schema validation error | ❌ NO | Framework already handles these |

### Complete Runnable Code

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
from langchain.messages import ToolMessage

# Tool that can fail at runtime
@tool
def execute_sql(query: str) -> str:
    """Execute a SQL query and return results."""
    # Simulate a SQL execution that might fail
    if "DROP" in query.upper():
        raise ValueError("DROP statements are not allowed")
    if "SELECT" not in query.upper():
        raise ValueError("Only SELECT queries are allowed")
    return f"Results: {query[:50]}..."

# Error handling middleware
@wrap_tool_call
def handle_tool_errors(request, handler):
    """Catch tool execution errors and return friendly messages."""
    try:
        return handler(request)  # Execute the tool normally
    except ValueError as e:
        # Return error as a ToolMessage the agent can see
        return ToolMessage(
            content=f"Tool error: {str(e)}. Please check your input and try again.",
            tool_call_id=request.tool_call["id"]
        )
    except Exception as e:
        # Catch unexpected errors too
        return ToolMessage(
            content=f"Unexpected error: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )

# Create agent with error handling
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[execute_sql],
    middleware=[handle_tool_errors],
    system_prompt="You are a SQL assistant. Help users write safe SELECT queries."
)

# Test it
result = agent.invoke({
    "messages": [{"role": "user", "content": "Run DROP TABLE users"}]
})

print(result)
# The agent receives: "Tool error: DROP statements are not allowed"
# Instead of the program crashing, the agent can explain why and offer alternatives
```

### Without Error Handling (What Happens)

```python
# If you DON'T use error handling:
agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[execute_sql]
    # No middleware
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Run DROP TABLE users"}]
})
# The tool raises ValueError("DROP statements...")
# The program CRASHES — error is not caught
# No response to the user
```

---

## 7. Tools Accessing Custom State

Tools sometimes need information from the agent's state (not just their function parameters). For example, a greeting tool needs the user's name.

**Source:** `migrate-complete.md` (Custom state → Defining state via state_schema section)

### Explanation

The agent has a **state** — a dict containing:
- `messages` — the conversation history (always included)
- Custom fields — anything you add (e.g., `user_name`, `user_id`, `account_tier`)

To access this state **inside a tool**, add a special `runtime: ToolRuntime` parameter.

### Complete Runnable Code

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState

# Step 1: Define custom state with user_name
class CustomState(AgentState):
    user_name: str  # Add a custom field to the agent state

# Step 2: Define a tool that accesses the state
@tool
def greet(runtime: ToolRuntime[None, CustomState]) -> str:
    """Greet the user by their name.
    
    This tool reads the user_name from the agent state.
    """
    user_name = runtime.state.get("user_name", "Guest")
    return f"Hello {user_name}! How can I help you today?"

@tool
def remind_user(reminder: str, runtime: ToolRuntime[None, CustomState]) -> str:
    """Set a reminder for the user."""
    user_name = runtime.state.get("user_name", "Guest")
    return f"Reminder set for {user_name}: {reminder}"

# Step 3: Create agent with custom state
agent = create_agent(
    model="gpt-5.4",
    tools=[greet, remind_user],
    state_schema=CustomState,
    system_prompt="You are a helpful assistant. Use the greet tool when appropriate."
)

# Step 4: Invoke with state
result = agent.invoke({
    "messages": [{"role": "user", "content": "Greet me"}],
    "user_name": "Alice"  # ← Supply the custom state field
})

print(result)
# Output: "Hello Alice! How can I help you today?"
```

### How State Flows Through the System

```
Invoke agent:
  {
    "messages": [...],
    "user_name": "Alice"  ← Custom state field
  }
    ↓
Agent runs
    ↓
Model decides to call greet()
    ↓
Tool receives: runtime.state = {"messages": [...], "user_name": "Alice"}
    ↓
Tool reads: runtime.state.get("user_name") → "Alice"
    ↓
Tool returns: "Hello Alice!"
```

---

## 8. Custom State Schema

The agent's default state contains `messages`. You can extend it with additional fields for your application.

**Source:** `migrate-complete.md` (Custom state section)

### Explanation

Use `AgentState` (a `TypedDict`) as the base for custom state. **In v1, only `TypedDict` is supported** — Pydantic models and dataclasses are no longer allowed.

### Complete Runnable Code

```python
from langchain.agents import AgentState, create_agent
from langchain.tools import tool

# v1 (Correct) — inherit from AgentState (which is a TypedDict)
class MyAgentState(AgentState):
    user_id: str
    subscription_tier: str = "free"  # Optional with default

@tool
def check_subscription(runtime=None) -> str:
    """Check the user's subscription status."""
    # In a real tool, you'd use runtime to access state
    return "Subscription: Premium"

agent = create_agent(
    model="gpt-5-mini",
    tools=[check_subscription],
    state_schema=MyAgentState,
    system_prompt="You are a subscription assistant."
)

# Invoke with state
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's my subscription?"}],
    "user_id": "user_123",
    "subscription_tier": "premium"
})

print(result)
```

### What is NO Longer Supported

```python
# ❌ v0 — Pydantic not supported
from pydantic import BaseModel

class MyState(BaseModel):
    user_id: str

# ❌ v0 — dataclass not supported
from dataclasses import dataclass

@dataclass
class MyState:
    user_id: str

# ✅ v1 — TypedDict via AgentState
from langchain.agents import AgentState

class MyState(AgentState):
    user_id: str
```

### Why the Change?

- **Lightweight** — TypedDict has no runtime overhead
- **Better composition** — integrates with LangGraph's state system
- **Type safety** — still get IDE autocomplete and type checking
- **Validation in middleware** — if you need validation, put it in `before_model` hooks (Phase 02B)

---

## 9. Runtime Context

Static context (user metadata that doesn't change) is passed separately from dynamic state (messages).

**Source:** `migrate-complete.md` (Runtime context section)

### Explanation

```
Dynamic State (changes per turn)
├─ messages          → Conversation history
└─ custom fields     → User name, account tier, etc.

Static Context (same for entire session)
├─ user_id          → Who's using this
├─ session_id       → Which conversation
└─ API_key          → Credentials for external calls
```

**Context is for metadata that doesn't change.** State is for everything that does.

### Complete Runnable Code

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool

# Step 1: Define context schema
@dataclass
class Context:
    user_id: str
    session_id: str
    is_premium: bool = False

# Step 2: Tool that uses context
@tool
def check_api_limit(runtime=None) -> str:
    """Check how many API calls the user has left."""
    # In a real tool, you'd use runtime.context to check limits per user
    return "API calls remaining: 100"

# Step 3: Create agent with context schema
agent = create_agent(
    model="gpt-5.4",
    tools=[check_api_limit],
    context_schema=Context,
    system_prompt="You are an API assistant."
)

# Step 4: Invoke with context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "How many API calls do I have?"}]},
    context=Context(
        user_id="user_123",
        session_id="sess_abc",
        is_premium=True
    )
)

print(result)
# The agent has access to context and can provide personalized responses
```

### v0 Pattern (Still Works, But Not Recommended)

```python
# v0 — old pattern
result = agent.invoke(
    {"messages": [...]},
    config={
        "configurable": {
            "user_id": "user_123",
            "session_id": "sess_abc"
        }
    }
)

# v1 — new pattern (preferred)
result = agent.invoke(
    {"messages": [...]},
    context=Context(user_id="user_123", session_id="sess_abc")
)
```

**Why v1 is better:**
- Explicit — you can see what context fields are needed
- Type-safe — IDE autocomplete shows available fields
- Clearer intent — `context=` clearly means "static metadata"

---

## 10. Dynamic Model Selection

Choose a different language model per call based on runtime conditions (e.g., conversation length, task complexity).

**Source:** `migrate-complete.md` (Model → Dynamic model selection section)

### Explanation

Sometimes you want:
- **Cheap model** for simple, short conversations
- **Powerful model** for complex, long conversations
- **Different models** based on user account tier

Use middleware to decide which model to use per call.

### Complete Runnable Code

```python
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain_openai import ChatOpenAI
from typing import Callable

# Step 1: Define different models
cheap_model = ChatOpenAI(model="gpt-4o-mini")    # Fast, cheap
powerful_model = ChatOpenAI(model="gpt-4-turbo")  # Slow, expensive

# Step 2: Create middleware that decides which model to use
class DynamicModelMiddleware(AgentMiddleware):
    def __init__(self, messages_threshold: int = 10):
        self.messages_threshold = messages_threshold
    
    def wrap_model_call(
        self, 
        request: ModelRequest, 
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Choose model based on conversation length."""
        message_count = len(request.state.get("messages", []))
        
        if message_count > self.messages_threshold:
            print(f"Using powerful model (conversation has {message_count} messages)")
            selected_model = powerful_model
        else:
            print(f"Using cheap model (conversation has {message_count} messages)")
            selected_model = cheap_model
        
        # Override the model and execute
        return handler(request.override(model=selected_model))

# Step 3: Create agent with dynamic model middleware
agent = create_agent(
    model=cheap_model,  # Default
    tools=[],
    middleware=[DynamicModelMiddleware(messages_threshold=10)],
    system_prompt="You are a helpful assistant."
)

# Step 4: Test it
# First call — short conversation (uses cheap model)
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "What is 2+2?"}]
})
# Output: Using cheap model (conversation has 1 messages)

# After many turns — long conversation (uses powerful model)
messages = [
    {"role": "user", "content": f"Question {i}"}
    for i in range(15)
]
result2 = agent.invoke({"messages": messages})
# Output: Using powerful model (conversation has 15 messages)
```

### How It Works

```
User starts conversation
    ↓
Agent calls model
    ↓
DynamicModelMiddleware.wrap_model_call runs
    ↓
Checks message count: 1 message < 10 threshold
    ↓
Selects cheap_model
    ↓
Runs model with cheap_model
    ↓
Returns response

[User adds many more messages]
    ↓
Next model call
    ↓
Checks message count: 15 messages > 10 threshold
    ↓
Selects powerful_model
    ↓
Runs model with powerful_model (better reasoning)
```

---

## 11. Structured Output

Generate JSON responses with a specific schema instead of free-form text.

**Source:** `migrate-complete.md` (Structured output section)

### Explanation

Instead of asking the agent to write free-form responses, you can specify a JSON schema it must follow:

```python
class OutputSchema(BaseModel):
    summary: str
    sentiment: str  # "positive", "negative", "neutral"
```

Now the agent always returns `{"summary": "...", "sentiment": "..."}` — never random text.

### Complete Runnable Code

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.tools import tool
from pydantic import BaseModel

# Step 1: Define the output schema
class ReviewAnalysis(BaseModel):
    summary: str
    sentiment: str  # "positive", "negative", "neutral"
    rating: int     # 1-5 stars

# Step 2: Define a sample tool
@tool
def fetch_review(review_id: str) -> str:
    """Fetch a customer review by ID."""
    return "Great product! Works exactly as described. Very happy with the purchase. Highly recommend!"

# Step 3a: Create agent with ToolStrategy (works with any model)
agent_tool_strategy = create_agent(
    model="gpt-5-mini",
    tools=[fetch_review],
    response_format=ToolStrategy(ReviewAnalysis),
    system_prompt="You are a review analyst. Analyze reviews and provide structured output."
)

# Step 3b: Alternatively, use ProviderStrategy (faster, but model-specific)
agent_provider_strategy = create_agent(
    model="gpt-5-mini",
    tools=[fetch_review],
    response_format=ProviderStrategy(ReviewAnalysis),
    system_prompt="You are a review analyst."
)

# Step 4: Invoke
result = agent_tool_strategy.invoke({
    "messages": [{"role": "user", "content": "Analyze review #123"}]
})

print(result)
# Output is now ALWAYS a dict:
# {
#     "summary": "Customer loved the product and recommends it",
#     "sentiment": "positive",
#     "rating": 5
# }

# Access structured fields
print(f"Sentiment: {result['sentiment']}")
print(f"Rating: {result['rating']}/5")
```

### ToolStrategy vs ProviderStrategy

| Strategy | Pros | Cons |
|---|---|---|
| **ToolStrategy** | Works with any model | Uses one tool slot, slightly slower |
| **ProviderStrategy** | Faster, more reliable | Only works if model supports it |

**Recommendation:** Try ProviderStrategy first. If it fails, fall back to ToolStrategy.

### What Was Removed: Prompted Output

In v0, you could just ask the model to generate structured output in the prompt. This was unreliable:

```python
# ❌ v0 — no longer supported
agent = create_react_agent(
    model="gpt-5-mini",
    response_format=("Generate JSON like: {...}", OutputSchema)
)
```

**Why removed:** Models ignore prompts sometimes. The new strategies (tool-based and provider-native) are more reliable.

---

## 12. Streaming Node Name Change

When streaming events from agents, filter by node name.

**Source:** `migrate-complete.md` (Streaming node name rename section)

In v1, the model node was renamed from `"agent"` to `"model"`.

### Complete Runnable Code

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def hello() -> str:
    """Say hello."""
    return "Hello!"

agent = create_agent(
    model="gpt-5-mini",
    tools=[hello],
    system_prompt="You are friendly."
)

# Streaming with event filtering
for event in agent.stream({"messages": [{"role": "user", "content": "Say hello"}]}):
    # v1 — check for "model" node
    if "model" in event:
        print("Model response:", event["model"])
    # If you check for "agent", it won't match in v1
    # if "agent" in event:  # ❌ Won't work in v1
    #     print(event["agent"])
```

### Before and After

```python
# v0 — check for "agent" node
for event in agent.stream(...):
    if "agent" in event:  # ✅ Works in v0
        print(event["agent"])

# v1 — check for "model" node
for event in agent.stream(...):
    if "model" in event:  # ✅ Works in v1
        print(event["model"])
```

---

## 13. v0 → v1 Quick Reference

| What changed | v0 (old) | v1 (new) |
|---|---|---|
| Import path | `from langgraph.prebuilt import create_react_agent` | `from langchain.agents import create_agent` |
| Function name | `create_react_agent(...)` | `create_agent(...)` |
| System prompt param | `prompt="..."` | `system_prompt="..."` |
| SystemMessage in prompt | `prompt=SystemMessage(content="...")` | `system_prompt="..."` (extract string) |
| Tools list | `tools=ToolNode([tool_a, tool_b])` | `tools=[tool_a, tool_b]` |
| Tool error handling | Part of `ToolNode` | `@wrap_tool_call` middleware |
| Static context | `config={"configurable": {...}}` | `context=Context(...)` |
| State schema | Pydantic or dataclass | `TypedDict` only (via `AgentState`) |
| Streaming node name | `"agent"` | `"model"` |
| Structured output | Prompted output | `ToolStrategy` / `ProviderStrategy` |

---

## 14. Self-Quiz

Test yourself before moving to Phase 02B:

1. What is the new import path for `create_agent` in v1?
2. What three types does `tools=` accept in v1?
3. Explain the difference between static `system_prompt` and dynamic prompts.
4. When should you use `@wrap_tool_call` vs. let an error crash the program?
5. What is `runtime.state` and how do you use it in a tool?
6. Can you use Pydantic models for `state_schema` in v1?
7. What is the difference between `messages` and `context` when invoking an agent?
8. How does `DynamicModelMiddleware` decide which model to use?
9. What are `ToolStrategy` and `ProviderStrategy` used for?
10. What streaming node name should you check for in v1?
11. Why was prompted output removed for structured responses?
12. What does `request.override(model=new_model)` do?
13. When would you use a dynamic prompt instead of a static one?
14. What is the purpose of `@dynamic_prompt` decorator?
15. How do you pass context when invoking an agent?

---

## 15. Flashcards

Study these before Phase 02B:

| # | Question | Answer |
|---|---|---|
| 1 | Import path for `create_agent` in v1? | `from langchain.agents import create_agent` |
| 2 | Old name of `system_prompt`? | `prompt` |
| 3 | Three types `tools=` accepts? | `@tool` functions, plain callables, `BaseTool` instances |
| 4 | What decorator turns a function into a tool? | `@tool` |
| 5 | What replaced `pre_model_hook`? | `@dynamic_prompt` middleware or custom `AgentMiddleware` |
| 6 | What replaced `ToolNode`? | Just pass a plain list: `tools=[tool_a, tool_b]` |
| 7 | How to pass context in v1? | `context=Context(...)` on `invoke` |
| 8 | What is `runtime.state`? | The agent's state dict (includes `messages` + custom fields) |
| 9 | Can you use Pydantic for state_schema? | No — only `TypedDict` via `AgentState` |
| 10 | Streaming node name in v1? | `"model"` |
| 11 | What does `@wrap_tool_call` do? | Wraps tool execution to catch runtime errors |
| 12 | When NOT to catch tool errors? | Network failures, implementation bugs, schema errors |
| 13 | What are `ToolStrategy` / `ProviderStrategy`? | Two ways to enforce structured output (JSON schema) |
| 14 | Why use dynamic prompts? | To adapt instructions based on user role, conversation state, etc. |
| 15 | What does `request.override(model=x)` do? | Creates a modified request with a different model |

---

> **Next in Phase 02:** [Phase 02B — Middleware Deep Dive](./phase-02b-middleware.md)  
> **Source files used:** `migrate-complete.md` (all sections)
