# Buffer 4 — Memory, Context Engineering, Guardrails, Middleware

> RAW extraction material for synthesis. DENSE, FAITHFUL, COMPLETE. Real API names + code preserved.
> Sources (LangChain Python OSS docs):
> - 09-short-term-memory.md
> - 10-long-term-memory.md
> - 14-context-engineering.md
> - 15-guardrails.md
> - middleware/01-overview.md, 02-built-in.md, 03-custom.md

---

## 0. THE UNIFYING FRAME — Context Engineering

**This is the conceptual spine of the entire cluster.** Memory (short + long term), middleware, and guardrails are ALL mechanisms that shape *what the model sees and does at each step*. They are not separate features; they are instruments of context engineering.

**Definition (verbatim from docs):** "**Context engineering** is providing the right information and tools in the right format so the LLM can accomplish a task. This is the number one job of AI Engineers."

**Why agents fail** (the core motivation): When agents fail, the LLM call took the wrong action. LLMs fail for one of two reasons:
1. The underlying LLM is not capable enough
2. The "right" context was not passed to the LLM

"More often than not — it's actually the second reason." The lack of "right" context is the number one blocker for reliable agents. LangChain's agent abstractions are "uniquely designed to facilitate context engineering."

**The mechanism that makes it practical = MIDDLEWARE.** Quote: "LangChain middleware is the mechanism under the hood that makes context engineering practical." Middleware lets you (a) **update context** and (b) **jump to a different step** in the agent lifecycle. So: *Context engineering is the goal; middleware is the means; memory and guardrails are specific applications.*

---

# TOPIC 1: SHORT-TERM MEMORY

## 1.1 Purpose
**Problem solved:** An agent must remember previous interactions *within a single thread/conversation* (e.g., the user said "my name is Bob" earlier). Without it, each turn is stateless.

**WHY it matters:** Conversation history is the most common form of short-term memory. Long conversations challenge LLMs: a full history may not fit the context window → context loss/errors. Even when it fits, LLMs perform poorly over long contexts — they get "distracted" by stale/off-topic content, suffer slower responses and higher cost. So short-term memory must be actively *managed* (trim/delete/summarize), not just accumulated.

**Key concept — thread:** "A thread organizes multiple interactions in a session, similar to the way email groups messages in a single conversation."

**Scope contrast:** Short-term = single thread. Cross-thread/session recall = long-term memory (Topic 2).

## 1.2 Building blocks (every API named)
- **`checkpointer`** — the parameter passed to `create_agent` that enables thread-level persistence. Short-term memory is managed as part of the agent's **state** (graph state). State persisted to a DB (or memory) via the checkpointer; thread resumable any time. "Short-term memory updates when the agent is invoked or a step (like a tool call) is completed, and the state is read at the start of each step."
- **`InMemorySaver`** — from `langgraph.checkpoint.memory`; in-memory checkpointer for dev/prototyping.
- **`PostgresSaver`** — from `langgraph.checkpoint.postgres`; production checkpointer. `PostgresSaver.from_conn_string(DB_URI)` context manager; `.setup()` auto-creates tables. Install: `pip install langgraph-checkpoint-postgres`. Other backends: SQLite, Postgres, Azure Cosmos DB (see LangGraph persistence "checkpointer libraries").
- **`thread_id`** — passed via config `{"configurable": {"thread_id": "1"}}`. Identifies which thread/conversation to read/write. SAME thread_id across `invoke` calls → memory persists.
- **`AgentState`** — default state schema; manages conversation history via a `messages` key. From `langchain.agents`.
- **`state_schema`** — `create_agent` parameter to pass a *custom* state schema (subclass `AgentState`) adding fields like `user_id`, `preferences`. Also an `AgentMiddleware` attribute.
- **`messages` key** + **`add_messages` reducer** — default `AgentState` uses the `add_messages` reducer; required for `RemoveMessage` to work.
- **`RemoveMessage`** — from `langchain.messages`; deletes messages from state. `RemoveMessage(id=m.id)` removes one; `RemoveMessage(id=REMOVE_ALL_MESSAGES)` removes all.
- **`REMOVE_ALL_MESSAGES`** — from `langgraph.graph.message`; sentinel id to clear entire history.
- **`SummarizationMiddleware`** — built-in middleware to summarize+replace old messages. From `langchain.agents.middleware`.
- **Message-management hooks/decorators:** `@before_model`, `@after_model` (from `langchain.agents.middleware`) — used to trim/delete.
- **`ToolRuntime`** — typed `runtime` param a tool receives; `runtime.state` (read state), `runtime.context` (runtime context), `runtime.store` (long-term store), `runtime.tool_call_id`. Hidden from the tool signature (model doesn't see it).
- **`Command`** — from `langgraph.types`; tools return `Command(update={...})` to write state (and append `ToolMessage`).
- **`dynamic_prompt` / `ModelRequest`** — from `langchain.agents.middleware`; build dynamic system prompts from state/context.
- **`Runtime`** — from `langgraph.runtime`; passed to node-style hooks.
- **`RunnableConfig`** — from `langchain_core.runnables`; type for the config dict carrying `thread_id`.

## 1.3 Annotated code (verbatim, most important)

### (A) Minimal short-term memory via checkpointer — THE canonical example
```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


def get_user_info() -> str:
    """Look up information about the current user."""
    return "No user profile on file."


agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),
)

thread_config = {"configurable": {"thread_id": "1"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    thread_config,
)["messages"][-1].content

print(response)  # "Hi Bob! Nice to see you here. How are you doing?"

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    thread_config,
)["messages"][-1].content

print(response)  # "You are Bob!"
```
**Per-block WHY:** `checkpointer=InMemorySaver()` is the ONLY thing needed to turn on memory. The SAME `thread_config` (same `thread_id`) on the second `invoke` is what makes the agent recall "Bob" — without it, the second call would be a fresh thread. The agent's `messages` state is loaded at step start, updated at step end, persisted by the checkpointer.

### (B) Production checkpointer (Postgres)
```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgreSQL
    agent = create_agent(
        "gpt-5.5",
        tools=[get_user_info],
        checkpointer=checkpointer,
    )
```
**WHY:** Same API surface as InMemorySaver — swapping persistence backend requires only changing the checkpointer. `.setup()` creates tables; context manager scopes the connection.

### (C) Custom state schema (extend AgentState)
```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver


class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    "gpt-5.5",
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}})
```
**WHY:** Short-term memory isn't only `messages` — you can persist arbitrary structured fields per thread by subclassing `AgentState` and passing `state_schema`. Custom fields are passed in `invoke`.

### (D) Trim messages via `@before_model` (transient-ish, runs each model call)
```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    "gpt-5.5",
    tools=[...],
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)
```
**WHY:** `@before_model` fires before every model call. Returning a dict with a `messages` key updates state via the reducer. The trick: emit `RemoveMessage(id=REMOVE_ALL_MESSAGES)` to clear, then re-add the kept subset (`first_msg` + last few). Keeps the system/first message + recent turns so the model still sees relevant context within the window.

### (E) Summarize via `SummarizationMiddleware` (PERSISTENT replacement)
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig


checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-5.5",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=checkpointer,
)
```
**WHY:** Trimming/deleting *loses* information. Summarization condenses older messages into a summary message (separate LLM call) and *permanently* replaces them in state — recent messages stay intact. Future turns see the summary, not originals. `trigger=("tokens", 4000)` fires when token count ≥ 4000; `keep=("messages", 20)` preserves last 20 messages. Note `model` for summary can be cheaper (`gpt-5.4-mini`).

### (F) Read short-term state in a tool (`ToolRuntime.state`)
```python
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime


class CustomState(AgentState):
    user_id: str

@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """Look up user info."""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_user_info],
    state_schema=CustomState,
)
```
**WHY:** Tools read state via `runtime.state`. The `runtime` param is hidden from the model (not in the tool's declared signature/schema), so it doesn't pollute the tool spec.

### (G) Write short-term state from a tool (`Command(update=...)`)
```python
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel


class CustomState(AgentState):
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """Look up and update user info."""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })
```
**WHY:** Tools persist intermediate results to state by returning `Command(update={...})`. Note `ToolRuntime[CustomContext, CustomState]` parametrizes both context and state types. Must include a `ToolMessage` with the `tool_call_id` so the message history stays valid (assistant tool-call → tool-result pairing).

## 1.4 Advanced concepts (context-window management)
Four common patterns when conversation exceeds the window:
1. **Trim messages** — remove first/last N before calling the LLM (`@before_model`, `RemoveMessage`).
2. **Delete messages** — permanently delete from LangGraph state (`RemoveMessage(id=m.id)` / `REMOVE_ALL_MESSAGES`). Often via `@after_model`.
3. **Summarize messages** — replace old with an LLM-generated summary (`SummarizationMiddleware`). Preserves info that trimming loses.
4. **Custom strategies** — e.g., message filtering.

**Trimming strategy detail:** Count tokens; truncate when approaching the limit. LangChain's trim-messages utility lets you specify tokens to keep + a `strategy` (e.g., keep last `max_tokens`) for boundary handling.

**Delete validity WARNING (gotcha):** After deleting, ensure the resulting history is valid for the provider:
- Some providers expect history to START with a `user` message.
- Most providers require an `assistant` message with tool calls to be FOLLOWED by corresponding `tool` result messages.

**Access points for state (5 ways):** (1) Tools via `runtime`, (2) `dynamic_prompt` middleware, (3) `@before_model`, (4) `@after_model`, (5) `wrap_model_call`. Transient vs persistent: model-context edits via `wrap_model_call` are transient (single call); life-cycle hooks (`before_model`/`after_model`) persist to state.

## 1.5 Cross-framework interaction points
- Short-term memory ↔ LangGraph persistence: the `checkpointer` (`InMemorySaver`/`PostgresSaver`) comes FROM `langgraph.checkpoint.*`; LangChain agents store their `messages`/state in LangGraph's checkpointed graph state.
- Short-term memory ↔ middleware: trimming/deleting/summarizing are implemented as middleware hooks (`@before_model`, `@after_model`, `SummarizationMiddleware`).
- Short-term memory ↔ tools: tools read state (`runtime.state`) and write it (`Command(update=...)`).
- Short-term memory ↔ `RemoveMessage`/reducers: deletion requires the `add_messages` reducer on the `messages` state key.

## 1.6 Gotchas / version notes
- Without a matching `thread_id`, each `invoke` is effectively a new conversation (no recall).
- `InMemorySaver` loses data on process restart — use a DB-backed checkpointer in prod.
- HITL and several limit middlewares REQUIRE a checkpointer.
- After manual deletion, you can produce an invalid history (see WARNING above).

---

# TOPIC 2: LONG-TERM MEMORY

## 2.1 Purpose
**Problem solved:** Recall information ACROSS different conversations/sessions/threads (user preferences, extracted insights, historical data). Unlike short-term memory (scoped to one thread), long-term memory persists across threads and is recallable any time.

**WHY:** Personalization and continuity — a user's name, language, preferences learned in thread A should be available in thread B.

**Foundation:** Built on **LangGraph stores**; data saved as JSON documents organized by **namespace** and **key**.

## 2.2 Building blocks (every API named)
- **`store`** — `create_agent` parameter. Pass a store to enable long-term memory.
- **`BaseStore`** — base class (the store interface; `runtime.store` is typed against it).
- **`InMemoryStore`** — from `langgraph.store.memory`; in-memory dict store for dev. Constructor accepts `index=IndexConfig(...)` for semantic search.
- **`PostgresStore`** — from `langgraph.store.postgres`; production store. `PostgresStore.from_conn_string(DB_URI)` context manager; `.setup()` creates tables. Install: `pip install langgraph-checkpoint-postgres`.
- **`namespace`** — a tuple acting like a folder, e.g. `("users",)` or `(user_id, application_context)`. Groups related data; often includes user/org IDs. Enables hierarchical organization; cross-namespace search via content filters.
- **`key`** — distinct id within a namespace (like a file name), e.g. `"user_123"`, `"a-memory"`.
- **`.put(namespace, key, value)`** — write a JSON document.
- **`.get(namespace, key)`** — retrieve. Returns a value object exposing `.value` (the dict) and metadata. (Docs note: "returns StoreValue object with value and metadata".)
- **`.search(namespace, filter=..., query=...)`** — search within a namespace; `filter` does content-equality filtering; `query` does vector-similarity ranking (requires embeddings/index). E.g. `store.search(namespace, filter={"my-key": "my-value"}, query="language preferences")`.
- **`IndexConfig`** — from `langgraph.store.base`; configures semantic search: `IndexConfig(embed=embed_fn, dims=2)`.
- **`embed`** — an embedding function `Callable[[Sequence[str]], list[list[float]]]` (or a LangChain embeddings object) used by the index for semantic search.
- **`runtime.store`** — how a tool accesses the store at runtime (same store passed to `create_agent`). Tools assert `runtime.store is not None`.
- **`context_schema` / `Context` dataclass** — carries `user_id` etc. into `runtime.context` so tools/prompts know whose memory to read/write.
- **`UserInfo` (TypedDict)** — example pattern: define a typed schema for what the LLM should save.

## 2.3 Annotated code (verbatim)

### (A) Memory storage with namespaces, keys, semantic search
```python
from collections.abc import Sequence

from langgraph.store.base import IndexConfig
from langgraph.store.memory import InMemoryStore


def embed(texts: Sequence[str]) -> list[list[float]]:
    # Replace with an actual embedding function or LangChain embeddings object
    return [[1.0, 2.0] for _ in texts]


# InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production use.
store = InMemoryStore(index=IndexConfig(embed=embed, dims=2))
user_id = "my-user"
application_context = "chitchat"
namespace = (user_id, application_context)
store.put(
    namespace,
    "a-memory",
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
        "my-key": "my-value",
    },
)
# get the "memory" by ID
item = store.get(namespace, "a-memory")
# search for "memories" within this namespace, filtering on content equivalence, sorted by vector similarity
items = store.search(
    namespace, filter={"my-key": "my-value"}, query="language preferences"
)
```
**Per-block WHY:** `IndexConfig(embed=embed, dims=2)` turns on semantic search — `embed` converts text to vectors, `dims` is the vector dimensionality. `namespace = (user_id, application_context)` scopes memory hierarchically per user + context. `.put` stores an arbitrary JSON dict. `.get` is exact lookup by key; `.search` combines a content `filter` (exact match on `my-key`) with a semantic `query` (ranked by vector similarity). NOTE: embeddings are what enable `query`-based semantic recall.

### (B) Read long-term memory in a tool
```python
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.runnables import Runnable
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str


store = InMemoryStore()

store.put(
    ("users",),          # Namespace
    "user_123",          # Key (user ID)
    {"name": "John Smith", "language": "English"},
)


@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Look up user info."""
    assert runtime.store is not None
    user_id = runtime.context.user_id
    user_info = runtime.store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"


agent: Runnable = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[get_user_info],
    store=store,                # enables tool access to store
    context_schema=Context,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    context=Context(user_id="user_123"),
)
```
**WHY:** `store=store` on `create_agent` makes the SAME store available inside tools via `runtime.store`. `runtime.context.user_id` (from `context_schema=Context` + `context=` at invoke) tells the tool whose memory to fetch. `user_info.value` is the stored dict. This is the read half of long-term memory.

### (C) Write long-term memory from a tool
```python
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.runnables import Runnable
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

store = InMemoryStore()


@dataclass
class Context:
    user_id: str


class UserInfo(TypedDict):
    name: str


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save user info."""
    assert runtime.store is not None
    store = runtime.store
    user_id = runtime.context.user_id
    store.put(("users",), user_id, dict(user_info))   # (namespace, key, data)
    return "Successfully saved user info."


agent: Runnable = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[save_user_info],
    store=store,
    context_schema=Context,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    context=Context(user_id="user_123"),
)

# You can access the store directly to get the value
item = store.get(("users",), "user_123")
```
**WHY:** The `UserInfo` TypedDict gives the LLM a structured schema for what to extract/save. The tool calls `store.put(("users",), user_id, dict(user_info))` to persist. After the run, the host code can read directly with `store.get`. This is how an agent "learns" facts that survive across threads.

## 2.4 Advanced concepts
- **Memory types (conceptual):** semantic (facts), episodic (events), procedural (how-to). (Doc points to memory conceptual guide.)
- **Hierarchical organization via namespaces:** put user/org IDs + context labels in the namespace tuple; search across namespaces with content filters.
- **Semantic search w/ embeddings:** requires `IndexConfig(embed=..., dims=...)` on the store; then `.search(..., query=...)` ranks by vector similarity. Without an index, `query` semantic ranking isn't available (only `filter`). NOTE: per project conventions, embeddings are deferred in LangLearn — but the API path is `IndexConfig` + `embed`.
- **Direct store access:** the host program can call `store.get/.put/.search` outside of tools (the store object is shared).

## 2.5 Cross-framework interaction points
- Long-term memory ↔ LangGraph persistence: the `store`/`BaseStore`/`InMemoryStore`/`PostgresStore` come FROM `langgraph.store.*`; long-term memory IS a LangGraph memory store.
- Long-term memory ↔ embeddings/retrieval: semantic `.search(query=...)` depends on an embeddings function via `IndexConfig`; this is the bridge to vector retrieval/RAG-style recall.
- Long-term memory ↔ tools: tools read/write via `runtime.store` (`.get`/`.put`/`.search`).
- Long-term memory ↔ runtime context: `context_schema`/`runtime.context.user_id` scopes which namespace/key to use.
- Long-term memory ↔ context engineering: it is the "Store" data source (cross-conversation scope) in the context-engineering data-source table.
- Long-term memory ↔ Deep Agents filesystem: `FilesystemMiddleware` + `CompositeBackend`/`StoreBackend` routes `/memories/` paths to a `StoreBackend` for persistent cross-thread files.

## 2.6 Gotchas / version notes
- `InMemoryStore` is ephemeral; use `PostgresStore` (or other DB store) in prod.
- `.get` returns an object; use `.value` for the dict (and check truthiness for "not found").
- Semantic `query` needs an index/embeddings; otherwise rely on `filter`.
- The `embed` stub in docs returns constant vectors — replace with a real embedding model.

---

# TOPIC 3: CONTEXT ENGINEERING (the unifying frame, full detail)

## 3.1 Purpose
See Section 0. The job: pass the *right* information + tools in the *right* format. It is "the number one job of AI Engineers" and the number-one reliability lever.

**The agent loop (2 steps):**
1. **Model call** — call the LLM with a prompt + available tools; returns a response OR a request to execute tools.
2. **Tool execution** — run the requested tools; return tool results.
Loop continues until the LLM decides to finish.

To build reliable agents you control what happens at each step AND between steps.

## 3.2 Building blocks — THE THREE CONTEXT TYPES + THREE DATA SOURCES

### Context types (what you control) — table verbatim
| Context Type | What You Control | Transient or Persistent |
| --- | --- | --- |
| **Model Context** | What goes into model calls (instructions, message history, tools, response format) | Transient |
| **Tool Context** | What tools can access and produce (reads/writes to state, store, runtime context) | Persistent |
| **Life-cycle Context** | What happens between model and tool calls (summarization, guardrails, logging, etc.) | Persistent |

- **Transient context:** what the LLM sees for a single call; modify messages/tools/prompts without changing saved state.
- **Persistent context:** what gets saved in state across turns; life-cycle hooks and tool writes modify this permanently.

### Data sources (what the agent reads/writes) — table verbatim
| Data Source | Also Known As | Scope | Examples |
| --- | --- | --- | --- |
| **Runtime Context** | Static configuration | Conversation-scoped | User ID, API keys, DB connections, permissions, env settings |
| **State** | Short-term memory | Conversation-scoped | Current messages, uploaded files, auth status, tool results |
| **Store** | Long-term memory | Cross-conversation | User preferences, extracted insights, memories, historical data |

### Model context sub-levers (5)
1. **System Prompt** — base instructions; can be dynamic per state/store/runtime context (`@dynamic_prompt`, `ModelRequest`).
2. **Messages** — full conversation history sent to the LLM; inject/modify via `wrap_model_call` + `request.override(messages=...)`.
3. **Tools** — which tools are available; filter dynamically via `request.override(tools=...)`.
4. **Model** — which model+config to call; swap via `request.override(model=...)` (`init_chat_model`).
5. **Response Format** — output schema; set via `request.override(response_format=Schema)` (Pydantic `BaseModel`).

All five can draw from State, Store, or Runtime Context.

### Key APIs
- `@dynamic_prompt`, `ModelRequest` (has `.messages`, `.state`, `.runtime`, `.system_message`, `.tools`, `.override(...)`).
- `wrap_model_call`, `ModelResponse`, `ExtendedModelResponse`.
- `request.override(...)` — transient per-call override of messages/tools/model/response_format/system_message.
- `request.runtime.context`, `request.runtime.store`, `request.state`.
- `Command(update={...})` — persistent state write (from tools or via `ExtendedModelResponse`).
- `@tool(parse_docstring=True)` — tool definition where name/description/arg descriptions guide the model.
- `init_chat_model` — instantiate models once for dynamic selection.

## 3.3 Annotated code (verbatim, most important levers)

### (A) Dynamic system prompt from STORE (long-term memory → instructions)
```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id

    # Read from Store: get user preferences
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)

    base = "You are a helpful assistant."

    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\nUser prefers {style} responses."

    return base

agent = create_agent(
    model="gpt-5.4",
    tools=[...],
    middleware=[store_aware_prompt],
    context_schema=Context,
    store=InMemoryStore()
)
```
**WHY:** Shows the trifecta — `@dynamic_prompt` returns the system prompt string per call; it reads `runtime.context.user_id` (Runtime Context) and `runtime.store` (Store/long-term memory) to personalize instructions. This is "the right instructions for the current state."

### (B) Inject messages transiently from STATE via `wrap_model_call`
```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def inject_file_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Inject context about files user has uploaded this session."""
    uploaded_files = request.state.get("uploaded_files", [])

    if uploaded_files:
        file_descriptions = []
        for file in uploaded_files:
            file_descriptions.append(
                f"- {file['name']} ({file['type']}): {file['summary']}"
            )

        file_context = f"""Files you have access to in this conversation:
{chr(10).join(file_descriptions)}

Reference these files when answering questions."""

        messages = [
            *request.messages,
            {"role": "user", "content": file_context},
        ]
        request = request.override(messages=messages)

    return handler(request)

agent = create_agent(
    model="gpt-5.4",
    tools=[...],
    middleware=[inject_file_context]
)
```
**WHY:** `wrap_model_call` receives the `request` + a `handler` (the actual model call). You modify the request (`request.override(messages=...)`) and then call `handler(request)`. This is TRANSIENT — the injected `file_context` is seen by the model for this call only; it is NOT saved to state. Note the tip: "models pay more attention to final messages" → append context at the end.

### (C) Dynamic tool selection from STATE
```python
@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on conversation State."""
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state["messages"])

    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)
```
**WHY:** Too many tools overwhelm the model and increase errors; too few limit capability. Filter `request.tools` by `t.name` based on auth/state, then `override`. Tools must be registered up front; this filters the visible subset.

### (D) Dynamic model selection from STATE
```python
from langchain.chat_models import init_chat_model

large_model = init_chat_model("claude-sonnet-4-6")
standard_model = init_chat_model("gpt-5.4")
efficient_model = init_chat_model("gpt-5.4-mini")

@wrap_model_call
def state_based_model(request, handler):
    message_count = len(request.messages)
    if message_count > 20:
        model = large_model      # larger context window
    elif message_count > 10:
        model = standard_model
    else:
        model = efficient_model  # cheap for short convos
    request = request.override(model=model)
    return handler(request)
```
**WHY:** Match model to task — long conversation → big-context model; short → cheap model. Instantiate models ONCE outside the middleware (not per call).

### (E) Tool writes to STATE vs STORE
```python
# STATE (session-scoped) via Command:
from langgraph.types import Command

@tool
def authenticate_user(password: str, runtime: ToolRuntime) -> Command:
    """Authenticate user and update State."""
    if password == "correct":
        return Command(update={"authenticated": True})
    else:
        return Command(update={"authenticated": False})

# STORE (cross-session):
@tool
def save_preference(preference_key: str, preference_value: str,
                    runtime: ToolRuntime[Context]) -> str:
    """Save user preference to Store."""
    user_id = runtime.context.user_id
    store = runtime.store
    existing_prefs = store.get(("preferences",), user_id)
    prefs = existing_prefs.value if existing_prefs else {}
    prefs[preference_key] = preference_value
    store.put(("preferences",), user_id, prefs)
    return f"Saved preference: {preference_key} = {preference_value}"
```
**WHY:** Contrast persistence scopes. `Command(update=...)` writes STATE (lives within the thread). `store.put(...)` writes STORE (survives across threads). Same `ToolRuntime`, different destination.

### (F) Life-cycle context = Summarization (persistent)
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-5.4",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger={"tokens": 4000},
            keep={"messages": 20},
        ),
    ],
)
```
**WHY:** Life-cycle context = what happens BETWEEN model/tool steps. Summarization, unlike transient trimming, PERSISTENTLY updates state: (1) summarizes older messages via a separate LLM call, (2) replaces them with a summary message in state permanently, (3) keeps recent messages intact.

## 3.4 Advanced concepts
- **Transient vs persistent (critical mental model):** Model-context changes via `wrap_model_call`/`override` are transient (per-call, not saved). Life-cycle hooks (`before_model`/`after_model`/`wrap_tool_call`) and tool writes are persistent (modify state/store). For persistent updates from the model layer, return `ExtendedModelResponse` with a `Command`.
- **The four/three context-engineering levers:** The doc frames it as Model Context (transient), Tool Context (persistent), Life-cycle Context (persistent) — each drawing on Runtime Context, State, Store. (The "four levers" framing in synthesis = system prompt, tools, model, response format inside Model Context; plus the cross-cutting life-cycle lever.)
- **Best practices:** (1) Start simple (static prompts/tools), add dynamics only when needed; (2) test incrementally (one feature at a time); (3) monitor model calls/tokens/latency; (4) use built-in middleware (`SummarizationMiddleware`, `LLMToolSelectorMiddleware`); (5) document your context strategy; (6) understand transient vs persistent.

## 3.5 Cross-framework interaction points
- Context engineering ↔ middleware: middleware "is the mechanism that makes context engineering practical"; every lever above is a middleware hook.
- Context engineering ↔ memory: State = short-term memory; Store = long-term memory; both are context data sources.
- Context engineering ↔ tools: Tool Context reads/writes State/Store/Runtime Context.
- Context engineering ↔ guardrails: guardrails are a Life-cycle Context concern (validation between steps).
- Context engineering ↔ LangGraph: middleware hooks run inside the compiled LangGraph that `create_agent` returns.

## 3.6 Gotchas / version notes
- Forgetting transient vs persistent → surprise that `wrap_model_call` edits don't persist.
- Instantiate models once (outside middleware) for dynamic selection — avoid per-call init.
- Response format: agent runs the model/tool loop until done, THEN coerces the final response into the schema.

---

# TOPIC 4: GUARDRAILS

## 4.1 Purpose
**Problem solved:** Build safe, compliant agents by validating/filtering content at key execution points — detect sensitive info, enforce policies, validate outputs, prevent unsafe behavior BEFORE harm.

**Common use cases:** preventing PII leakage; detecting/blocking prompt injection; blocking harmful content; enforcing business/compliance rules; validating output quality/accuracy.

**WHY / mechanism:** Guardrails are implemented USING middleware — intercept execution before the agent starts, after it completes, or around model/tool calls.

**Two complementary approaches:**
- **Deterministic guardrails** — rule-based (regex, keyword match, explicit checks). Fast, predictable, cheap; may miss nuance.
- **Model-based guardrails** — LLMs/classifiers with semantic understanding. Catch subtle issues; slower, costlier.

## 4.2 Building blocks (every API named)
- **`PIIMiddleware`** (built-in) — from `langchain.agents.middleware`. Detect/handle PII.
  - Strategies: `redact` (`[REDACTED_{PII_TYPE}]`), `mask` (`****-****-****-1234`), `hash` (deterministic hash), `block` (raise exception).
  - Built-in PII types: `email`, `credit_card` (Luhn validated), `ip`, `mac_address`, `url`.
  - Params: `pii_type` (required), `strategy` (default `"redact"`), `detector` (custom fn or regex; default built-in), `apply_to_input` (default `True`), `apply_to_output` (default `False`), `apply_to_tool_results` (default `False`).
  - Custom detectors: regex string, compiled regex (`re.compile`), or function `detector(content: str) -> list[dict]` returning `{"text","start","end"}`.
- **`HumanInTheLoopMiddleware`** (built-in) — from `langchain.agents.middleware`. Pause for human approval of tool calls. Requires a checkpointer. Config `interrupt_on={tool_name: True|False|{...}}`. Resumed via `Command(resume={"decisions": [{"type": "approve"}]})`.
- **`AgentMiddleware`** — base class for custom guardrail middleware (class syntax).
- **`AgentState`** — state type in hooks.
- **`hook_config(can_jump_to=[...])`** — decorator/method config to declare jump targets (e.g., `["end"]`).
- **`@before_agent` / `@after_agent`** — decorators for session-level guardrails (once per invocation).
- **`before_agent` / `after_agent`** — class-method hooks.
- **`jump_to`** — return key (`"end"`, `"tools"`, `"model"`) to redirect control flow.
- **`init_chat_model`** — to build a separate safety-evaluation model.
- **`AIMessage`** — to inject a refusal message / type-check the last message.
- **`Runtime`** — from `langgraph.runtime`.

## 4.3 Annotated code (verbatim)

### (A) Built-in PII guardrail (layered strategies)
```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware


agent = create_agent(
    model="gpt-5.4",
    tools=[customer_service_tool, email_tool],
    middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)
```
**WHY:** Each `PIIMiddleware` handles ONE PII type with ONE strategy — composes freely in the list. `apply_to_input=True` checks user messages before the model; the custom `detector` regex defines a non-built-in type (`api_key`); `strategy="block"` raises on detection.

### (B) Custom deterministic guardrail — `before_agent` blocks banned keywords (class + decorator)
```python
# Class syntax
from typing import Any
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime

class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block requests containing banned keywords."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        first_message = state["messages"][0]
        if first_message.type != "human":
            return None
        content = first_message.content.lower()
        for keyword in self.banned_keywords:
            if keyword in content:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                    }],
                    "jump_to": "end"
                }
        return None
```
Decorator equivalent:
```python
from langchain.agents.middleware import before_agent, AgentState, hook_config

@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    ...
    return {"messages": [...], "jump_to": "end"}
```
**WHY:** `before_agent` runs ONCE at invocation start — ideal for session-level checks (auth, rate limit, banned content) before any processing. `@hook_config(can_jump_to=["end"])` declares the hook may short-circuit; returning `{"jump_to": "end"}` + a refusal message ends execution immediately, never calling the model.

### (C) Custom model-based guardrail — `after_agent` LLM safety check
```python
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from typing import Any

class SafetyGuardrailMiddleware(AgentMiddleware):
    """Model-based guardrail: Use an LLM to evaluate response safety."""

    def __init__(self):
        super().__init__()
        self.safety_model = init_chat_model("gpt-5.4-mini")

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None
        safety_prompt = f"""Evaluate if this response is safe and appropriate.
        Respond with only 'SAFE' or 'UNSAFE'.

        Response: {last_message.content}"""
        result = self.safety_model.invoke([{"role": "user", "content": safety_prompt}])
        if "UNSAFE" in result.content:
            last_message.content = "I cannot provide that response. Please rephrase your request."
        return None
```
**WHY:** `after_agent` runs ONCE after completion — final compliance scan on the full response. A SEPARATE cheaper model (`gpt-5.4-mini`) judges safety; if "UNSAFE", the last message content is overwritten with a refusal. Model-based catches semantic issues regex can't.

### (D) Stacking layered guardrails (order = defense in depth)
```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-5.4",
    tools=[search_tool, send_email_tool],
    middleware=[
        # Layer 1: Deterministic input filter (before agent)
        ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),
        # Layer 2: PII protection (before and after model)
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),
        # Layer 3: Human approval for sensitive tools
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),
        # Layer 4: Model-based safety check (after agent)
        SafetyGuardrailMiddleware(),
    ],
)
```
**WHY:** Guardrails "execute in order, allowing you to build layered protection." Input filtering → PII redaction (in+out) → human approval on dangerous tools → final model-based safety check. Each layer is independent and composable.

### (E) HITL guardrail (approval flow)
```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

agent = create_agent(
    model="gpt-5.4",
    tools=[search_tool, send_email_tool, delete_database_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,
                "delete_database": True,
                "search": False,        # auto-approve safe ops
            }
        ),
    ],
    checkpointer=InMemorySaver(),   # required for interrupts
)

config = {"configurable": {"thread_id": "some_id"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to the team"}]},
    config=config
)
# Resume after human approves:
result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)
```
**WHY:** HITL pauses (interrupts) before sensitive tools. REQUIRES a checkpointer + thread_id to persist state across the pause. Resume by invoking with `Command(resume={"decisions": [...]})` on the SAME thread_id.

## 4.4 Advanced concepts
- **Writing a guardrail AS middleware:** the entire point — guardrails are not a separate subsystem; they are middleware hooks. Choose the hook by scope: `before_agent` (once, input/session checks), `after_agent` (once, final output check), `before_model`/`after_model` (per model call), `wrap_tool_call` (around tools), or built-ins (`PIIMiddleware` around model in/out, `HumanInTheLoopMiddleware` around tools).
- **Deterministic vs model-based trade-off:** rules = fast/cheap/predictable but blunt; LLM-judge = nuanced but slow/costly. Layer both.
- **Short-circuiting:** `jump_to: "end"` + refusal message stops execution before harm; declare with `can_jump_to`.
- **PII on streamed output:** with `apply_to_output=True`, `PIIMiddleware` ALSO redacts streamed wire output (text deltas, tool-call args, tool outputs, state snapshots) via a registered stream transformer — requires `langchain>=1.3.2`.

## 4.5 Cross-framework interaction points
- Guardrails ↔ middleware: guardrails ARE middleware; built-ins (`PIIMiddleware`, `HumanInTheLoopMiddleware`) + custom hooks.
- Guardrails ↔ HITL/human-in-the-loop: `HumanInTheLoopMiddleware` is the canonical high-stakes guardrail.
- Guardrails ↔ context engineering: guardrails are a Life-cycle Context concern (validation between steps).
- Guardrails ↔ checkpointer: HITL needs a checkpointer to persist across interrupts.
- Guardrails ↔ short-term memory: `@after_model` guardrail can `RemoveMessage` sensitive content (see ST-memory example validate_response).
- Guardrails ↔ event streaming: PII redaction of wire output via stream transformers.

## 4.6 Gotchas / version notes
- HITL silently won't work without a checkpointer.
- `PIIMiddleware` `apply_to_output` default is `False` — must opt in to scan AI messages; `apply_to_tool_results` also defaults `False`.
- `credit_card` uses Luhn validation (reduces false positives).
- Streamed-output PII redaction requires `langchain>=1.3.2`.
- Persistent shell sessions (`ShellToolMiddleware`) do NOT currently support interrupts (HITL).

---

# TOPIC 5: MIDDLEWARE — OVERVIEW

## 5.1 Purpose
**Problem solved + WHY:** Middleware "provides a way to more tightly control what happens inside the agent." It is the implementation substrate for context engineering, memory management, and guardrails.

**The composition idea (CENTRAL):** Each middleware handles ONE concern and they compose freely by being added to the `middleware=[...]` list. Before-hooks run first→last, after-hooks last→first, wrap-hooks nest. This means cross-cutting concerns (logging, retries, PII, summarization, HITL) can be mixed and matched independently.

**Useful for:** tracking behavior (logging/analytics/debugging); transforming prompts, tool selection, output formatting; adding retries/fallbacks/early-termination; applying rate limits, guardrails, PII detection.

## 5.2 Building blocks (overview-level)
- Pass middleware via `create_agent(..., middleware=[...])`.
- The agent loop: model call → tool execution → finish (when no more tool calls).
- Middleware exposes hooks **before and after each step** (see custom doc for the full hook list).
- **Not a separate runtime:** hooks run INSIDE the compiled LangGraph that `create_agent` returns. You can drop the whole agent (middleware included) into a larger `StateGraph` as a node/subgraph and every hook still runs.
- `HumanInTheLoopMiddleware` matches each tool by `.name` (Python `@tool` name = function name).

## 5.3 Annotated code (verbatim)
### (A) Adding middleware
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-5.4",
    tools=[...],
    middleware=[
        SummarizationMiddleware(...),
        HumanInTheLoopMiddleware(...)
    ],
)
```
**WHY:** Middleware is just a list — composition is declarative. Order matters (see execution order).

### (B) Agent-with-middleware as a node in a larger StateGraph
```python
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.graph import START, StateGraph

email_agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[read_email, send_email],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_email": True})],
)

graph = (
    StateGraph(AgentState)
    .add_node("classify", classify_node)
    .add_node("email_agent", email_agent)
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route)
    .compile()
)
```
**WHY:** Middleware travels WITH the agent node. The HITL interrupt, summarization, PII redaction, retries, custom hooks all keep working when the agent is embedded in a bigger graph (classify → route → agent). Reach for this when topology exceeds "loop until done."

## 5.4 Advanced concepts
- Middleware ≠ separate runtime; it compiles into the LangGraph graph.
- Subgraph composition supports checkpointer scoping (per-invocation vs per-thread) — see "Use subgraphs."

## 5.5 Cross-framework interaction points
- Middleware ↔ LangGraph: hooks run inside the compiled LangGraph; agent+middleware can be a subgraph/node.
- Middleware ↔ agent loop: hooks fire before/after each loop step (see hook-order diagram).
- Middleware ↔ providers: provider-specific middleware exists for Anthropic (prompt caching, bash tool, text editor, memory, file search), AWS (Bedrock prompt caching), OpenAI (content moderation).

## 5.6 Gotchas / version notes
- HITL tool-name keys must match the tool `.name` exactly (Python: function name).

---

# TOPIC 6: MIDDLEWARE — BUILT-IN (exhaustive)

## 6.1 Purpose
Production-ready, configurable middleware for common concerns so you don't reinvent them. Provider-agnostic ones work with any LLM; provider-specific ones optimize for a vendor.

## 6.2 Building blocks — EXHAUSTIVE LIST OF BUILT-IN MIDDLEWARE
All importable from `langchain.agents.middleware` unless noted (Deep Agents ones from `deepagents.middleware.*`).

**Provider-agnostic (table verbatim):**
| Middleware | Description |
| --- | --- |
| **Summarization** | Auto-summarize conversation history near token limits. |
| **Human-in-the-loop** | Pause execution for human approval of tool calls. |
| **Model call limit** | Limit number of model calls to prevent excessive cost. |
| **Tool call limit** | Limit tool execution counts. |
| **Model fallback** | Fallback to alternative models when primary fails. |
| **PII detection** | Detect/handle PII. |
| **To-do list** | Task planning/tracking (`write_todos` tool). |
| **LLM tool selector** | Use an LLM to pre-select relevant tools. |
| **Tool retry** | Retry failed tool calls with exponential backoff. |
| **Model retry** | Retry failed model calls with exponential backoff. |
| **LLM tool emulator** | Emulate tool execution via an LLM (testing). |
| **Context editing** | Trim/clear older tool uses to manage context. |
| **Shell tool** | Persistent shell session for command execution. |
| **File search** | Glob + Grep search tools over a filesystem. |
| **Filesystem** | Filesystem for context + long-term memories (Deep Agents). |
| **Subagent** | Spawn subagents (Deep Agents). |

**Class names + key params:**
- **`SummarizationMiddleware`** — `model` (str|BaseChatModel, required), `trigger` (ContextSize tuple / TriggerClause dict / list; thresholds: `fraction` 0-1, `tokens` int, `messages` int; single tuple = one threshold, dict = AND of thresholds, list = OR), `keep` (ContextSize, default `("messages", 20)`; one of fraction/tokens/messages), `token_counter` (fn, default char-based), `summary_prompt` (template w/ `{messages}`), `trim_tokens_to_summarize` (default 4000). Deprecated: `summary_prefix`, `max_tokens_before_summary`, `messages_to_keep`. Types: `ContextSize`, `TriggerClause`. `fraction` needs model profile data (`langchain>=1.1`) or a custom `profile`.
- **`HumanInTheLoopMiddleware`** — `interrupt_on={tool_name: True | False | {"allowed_decisions": ["approve","edit","reject"]}}`. Requires checkpointer.
- **`ModelCallLimitMiddleware`** — `thread_limit` (across all runs in a thread), `run_limit` (per invocation), `exit_behavior` (`"end"` graceful | `"error"` raise; default `"end"`). Thread limiting needs checkpointer.
- **`ToolCallLimitMiddleware`** — `tool_name` (optional; omit = all tools globally), `thread_limit`, `run_limit` (at least one required), `exit_behavior` (`"continue"` default = block exceeded calls w/ error msgs, model continues | `"error"` raise `ToolCallLimitExceededError` | `"end"` stop w/ ToolMessage+AI msg, single-tool only else `NotImplementedError`).
- **`ModelFallbackMiddleware`** — `first_model` (required), `*additional_models` (tried in order). E.g. `ModelFallbackMiddleware("gpt-5.4-mini", "claude-3-5-sonnet-20241022")`.
- **`PIIMiddleware`** — (see Guardrails §4.2).
- **`TodoListMiddleware`** — auto-provides `write_todos` tool + planning system prompt. Params: `system_prompt`, `tool_description`.
- **`LLMToolSelectorMiddleware`** — `model` (default = agent's model), `system_prompt`, `max_tools` (int), `always_include` (list[str]; don't count against max). Uses structured output to pick relevant tools before main model. For 10+ tools.
- **`ToolRetryMiddleware`** — `max_retries` (default 2), `tools` (list, default all), `retry_on` (tuple of exc types or callable; default `(Exception,)`), `on_failure` (`"return_message"` default | `"raise"` | callable; doc full example also shows `"continue"`), `backoff_factor` (default 2.0; 0.0 = constant), `initial_delay` (1.0), `max_delay` (60.0), `jitter` (True, ±25%). Delay = `initial_delay * (backoff_factor ** retry_number)`.
- **`ModelRetryMiddleware`** — `max_retries` (2), `retry_on` (default `(Exception,)`), `on_failure` (`"continue"` default = return AIMessage w/ error | `"error"` raise | callable), `backoff_factor` (2.0), `initial_delay` (1.0), `max_delay` (60.0), `jitter` (True).
- **`LLMToolEmulator`** — `tools` (list[str|BaseTool]; `None`=emulate ALL, `[]`=none, list=only those), `model` (default agent's model). For testing without real tool execution.
- **`ContextEditingMiddleware`** + **`ClearToolUsesEdit`** — `edits` (list[ContextEdit], default `[ClearToolUsesEdit()]`), `token_count_method` (`"approximate"` default | `"model"`). `ClearToolUsesEdit`: `trigger` (token count, default 100000), `clear_at_least` (default 0), `keep` (recent tool results to preserve, default 3), `clear_tool_inputs` (default False), `exclude_tools` (default ()), `placeholder` (default `"[cleared]"`).
- **`ShellToolMiddleware`** + **`HostExecutionPolicy`** / **`DockerExecutionPolicy`** / **`CodexSandboxExecutionPolicy`** + **`RedactionRule`** — `workspace_root`, `startup_commands`, `shutdown_commands`, `execution_policy` (default HostExecutionPolicy), `redaction_rules`, `tool_description`, `shell_command` (default `/bin/bash`), `env`. Security: pick policy by isolation needs; redaction is post-execution (does NOT prevent exfiltration on host policy). Does NOT support interrupts (HITL).
- **`FilesystemFileSearchMiddleware`** — `root_path` (required), `use_ripgrep` (default True), `max_file_size_mb` (default 10). Adds `glob_search` + `grep_search` tools (Grep modes: `files_with_matches`, `content`, `count`).
- **`FilesystemMiddleware`** (Deep Agents) — tools `ls`, `read_file`, `write_file`, `edit_file`. Params: `backend` (default `StateBackend`), `system_prompt`, `custom_tool_descriptions`. Short-term (state) vs long-term: use `CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})` to route `/memories/` to a persistent `StoreBackend` (survives across threads).
- **`SubAgentMiddleware`** (Deep Agents) + **`CompiledSubAgent`** — `default_model`, `default_tools`, `subagents=[{name, description, system_prompt, tools, model, middleware}]` or `CompiledSubAgent(name, description, runnable=compiled_graph)`. A `general-purpose` subagent is always available for context isolation.

**Provider-specific:** Anthropic (prompt caching, bash tool, text editor, memory, file search), AWS (Bedrock prompt caching), OpenAI (content moderation).

## 6.3 Annotated code (verbatim, key built-ins)

### (A) Summarization with combined AND/OR triggers
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

# Single condition: trigger if tokens >= 4000
agent = create_agent(
    model="gpt-5.4",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)

# OR logic: trigger if tokens >= 3000 OR messages >= 6
agent2 = create_agent(
    model="gpt-5.4",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=[("tokens", 3000), ("messages", 6)],
            keep=("messages", 20),
        ),
    ],
)

# AND logic: trigger only when tokens >= 4000 AND messages >= 10
agent3 = create_agent(
    model="gpt-5.4",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger={"tokens": 4000, "messages": 10},
            keep=("messages", 20),
        ),
    ],
)

# Fractional limits (uses model profile data)
agent5 = create_agent(
    model="gpt-5.4",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger=("fraction", 0.8),
            keep=("fraction", 0.3),
        ),
    ],
)
```
**WHY:** Demonstrates the trigger algebra: tuple = one threshold; dict = AND; list = OR; list-of-dicts = OR-of-ANDs. `fraction` (of model context) needs profile data.

### (B) HITL with per-tool decision config
```python
agent = create_agent(
    model="gpt-5.4",
    tools=[your_read_email_tool, your_send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "your_send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "your_read_email_tool": False,
            }
        ),
    ],
)
```
**WHY:** `True` = require approval; `False` = auto-approve; dict = configure allowed decisions (approve/edit/reject). Read is auto-approved, send requires human decision.

### (C) Context editing (clear old tool outputs)
```python
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit

agent = create_agent(
    model="gpt-5.4",
    tools=[],
    middleware=[
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(trigger=100000, keep=3)],
        ),
    ],
)
```
**WHY:** When tokens exceed `trigger`, clear older tool outputs but `keep` the most recent 3; replace cleared content with a `[cleared]` placeholder. Distinct from summarization: it targets verbose TOOL results specifically.

### (D) Deep Agents filesystem with persistent /memories route
```python
from deepagents.middleware import FilesystemMiddleware
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
agent = create_agent(
    model="claude-sonnet-4-6",
    store=store,
    middleware=[
        FilesystemMiddleware(
            backend=CompositeBackend(
                default=StateBackend(),
                routes={"/memories/": StoreBackend()}
            ),
        ),
    ],
)
```
**WHY:** Files under `/memories/` go to `StoreBackend` (long-term, cross-thread); everything else stays in `StateBackend` (ephemeral). Directly links the filesystem abstraction to the short-term/long-term memory split.

## 6.4 Advanced concepts
- **Context management built-ins:** Summarization (replace w/ summary), ContextEditing (clear old tool outputs), LLMToolSelector (reduce tool count) — three different strategies for keeping the window small.
- **Resilience built-ins:** ModelFallback, ToolRetry, ModelRetry, ModelCallLimit, ToolCallLimit.
- **Multimodal caveat:** Summarization is TEXT-oriented — it does NOT compress image/audio/video; older multimodal messages become text summaries; for image-heavy apps store media externally and pass URLs/refs.
- **Subagent context isolation:** delegating to a subagent keeps the supervisor's context clean; `general-purpose` subagent always present.

## 6.5 Cross-framework interaction points
- Built-in middleware ↔ short-term memory: `SummarizationMiddleware`, `ContextEditingMiddleware` manage the `messages` state/window.
- Built-in middleware ↔ long-term memory: Deep Agents `FilesystemMiddleware` + `StoreBackend` persist to the LangGraph store.
- Built-in middleware ↔ checkpointer: `HumanInTheLoopMiddleware`, `ModelCallLimitMiddleware` (thread limit), `ToolCallLimitMiddleware` (thread limit) need a checkpointer.
- Built-in middleware ↔ guardrails: `PIIMiddleware` + `HumanInTheLoopMiddleware` ARE the built-in guardrails.
- Built-in middleware ↔ tools: `TodoListMiddleware`/`ShellToolMiddleware`/`FilesystemFileSearchMiddleware`/`FilesystemMiddleware`/`SubAgentMiddleware` REGISTER new tools via the `tools` class attribute.
- Built-in middleware ↔ event streaming: `PIIMiddleware` registers a stream transformer for wire-output redaction.

## 6.6 Gotchas / version notes
- `fraction` triggers/keep need model profile data (`langchain>=1.1`) or a manual `profile`.
- Summarization deprecated params: `summary_prefix` → `summary_prompt`; `max_tokens_before_summary` → `trigger`; `messages_to_keep` → `keep`.
- ToolCallLimit `exit_behavior="end"` only works for a single tool.
- ShellToolMiddleware redaction is post-exec (no exfiltration prevention on host policy); no HITL support.
- `LLMToolEmulator(tools=None)` emulates ALL tools (could surprise).

---

# TOPIC 7: MIDDLEWARE — CUSTOM

## 7.1 Purpose
Build your own middleware by implementing hooks that run at specific points in the agent execution flow. "Keep middleware focused — each should do one thing well" (the compose-freely philosophy).

## 7.2 Building blocks — ALL HOOKS, DECORATORS, CLASS

### Two hook styles
**Node-style hooks** (run sequentially at execution points; for logging/validation/state updates):
| Hook | When it runs |
| --- | --- |
| `before_agent` | Before agent starts (once per invocation) |
| `before_model` | Before each model call |
| `after_model` | After each model response |
| `after_agent` | After agent completes (once per invocation) |

**Wrap-style hooks** (run AROUND each call; you control if handler is called 0/1/N times — short-circuit/normal/retry; for retries/caching/transformation):
| Hook | When it runs |
| --- | --- |
| `wrap_model_call` | Around each model call |
| `wrap_tool_call` | Around each tool call |

### Decorators (from `langchain.agents.middleware`)
- **Node-style:** `@before_agent`, `@before_model`, `@after_model`, `@after_agent`.
- **Wrap-style:** `@wrap_model_call`, `@wrap_tool_call`.
- **Convenience:** `@dynamic_prompt` (generate dynamic system prompts).
- Decorators accept config: `@before_model(can_jump_to=["end"])`, `@after_model(state_schema=...)`, etc.

### Class-based
- **`AgentMiddleware`** base class. Implement hook methods (`before_model`, `after_model`, `wrap_model_call`, `wrap_tool_call`, etc.) and async variants (`abefore_model`, `aafter_model`, ...).
- Three class attributes picked up at compile time:
  - **`state_schema`** — extend agent state with custom fields.
  - **`tools`** — register additional tools that ship with the middleware.
  - **`transformers`** — register scope-aware stream transformer factories (`Callable[[tuple[str,...]], StreamTransformer]`, `langchain>=1.3.2`).
- **`hook_config(can_jump_to=[...])`** — method decorator declaring jump targets.

### Types/objects
- **`AgentState`** — base state; subclass to add fields (`NotRequired[...]`).
- **`ModelRequest`** — has `.messages`, `.state`, `.runtime`, `.system_message` (always a `SystemMessage`, even if created from a string), `.tools`, `.override(...)`.
- **`ModelResponse`** — the model call result (returned by `handler`).
- **`ExtendedModelResponse(model_response=..., command=Command(update={...}))`** — wrap a `ModelResponse` + a `Command` to inject persistent state updates from `wrap_model_call`.
- **`ToolCallRequest`** (from `langchain.tools.tool_node`) — passed to `wrap_tool_call`; has `.tool_call['name']`, `.tool_call['args']`.
- **`Command`** (from `langgraph.types`) — state update; from tools or `ExtendedModelResponse`.
- **`Runtime`** (from `langgraph.runtime`), **`SystemMessage`**, **`AIMessage`**, **`ToolMessage`**.

### State updates (mechanism differs by style)
- **Node-style** (`before_agent`/`before_model`/`after_model`/`after_agent`): return a dict; applied to state via the graph's reducers.
- **Wrap-style** (`wrap_model_call`/`wrap_tool_call`): for model calls return `ExtendedModelResponse` with a `Command`; for tool calls return a `Command` directly.

### Jump targets (`jump_to`)
- `"end"` — jump to end (or first `after_agent` hook).
- `"tools"` — jump to the tools node.
- `"model"` — jump to the model node (or first `before_model` hook).

## 7.3 Annotated code (verbatim)

### (A) Node-style hooks (decorator) — with jump
```python
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any


@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if len(state["messages"]) >= 50:
        return {
            "messages": [AIMessage("Conversation limit reached.")],
            "jump_to": "end"
        }
    return None

@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Model returned: {state['messages'][-1].content}")
    return None
```
**WHY:** `before_model` checks a condition and can `jump_to: "end"` to short-circuit before calling the model. `after_model` observes/logs the latest message. Returning `None` = no change.

### (B) Wrap-style retry (decorator) — calls handler N times
```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")
```
**WHY:** The defining power of wrap-style: you decide how many times `handler(request)` runs. Here, up to 3 attempts — retry on exception, re-raise on the last. (Short-circuit = call handler 0 times and return a canned response.)

### (C) THE CUSTOM MIDDLEWARE CLASS example (multi-hook, sync+async)
```python
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime
from typing import Any, Callable

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None

    async def abefore_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        # Async version of before_model
        return None

    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        # Async version of after_model
        print(f"Model returned: {state['messages'][-1].content}")
        return None


agent = create_agent(
    model="gpt-5.4",
    middleware=[LoggingMiddleware()],
    tools=[...],
)
```
**WHY:** Class style is for multiple hooks + sync/async pairs (`before_model`/`abefore_model`) + configuration. Use classes when you need both sync and async, multiple hooks, complex config, or reuse with init-time params.

### (D) Custom state schema across hooks (track a counter)
```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.agents.middleware import AgentState, AgentMiddleware
from typing_extensions import NotRequired
from typing import Any


class CustomState(AgentState):
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]


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
    model="gpt-5.4",
    middleware=[CallCounterMiddleware()],
    tools=[],
)

result = agent.invoke({
    "messages": [HumanMessage("Hello")],
    "model_call_count": 0,
    "user_id": "user-123",
})
```
**WHY:** `state_schema = CustomState` (also `AgentMiddleware[CustomState]` generic) registers custom fields at compile time. `after_model` increments the counter (state write via returned dict + reducer); `before_model` reads it and jumps to end past a limit. This is how middleware shares data across hooks and implements cross-cutting concerns (rate limiting, usage tracking).

### (E) Wrap-style state update via ExtendedModelResponse + composition
```python
from typing import Callable
from langchain.agents.middleware import (
    wrap_model_call, ModelRequest, ModelResponse, AgentState, ExtendedModelResponse
)
from langgraph.types import Command
from typing_extensions import NotRequired

class UsageTrackingState(AgentState):
    last_model_call_tokens: NotRequired[int]


@wrap_model_call(state_schema=UsageTrackingState)
def track_usage(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ExtendedModelResponse:
    response = handler(request)
    return ExtendedModelResponse(
        model_response=response,
        command=Command(update={"last_model_call_tokens": 150}),
    )
```
**WHY:** Wrap-style hooks can't just return a dict to update state — they return `ExtendedModelResponse(model_response=..., command=Command(update={...}))`. The `Command` flows through reducers (messages additive, not replaced). Use when state depends on logic during the model call (token usage, summarization triggers).

**Composition rules when multiple middleware return `ExtendedModelResponse`:**
- Commands applied through reducers (messages additive).
- **Outer wins on conflicts** for non-reducer fields (applied inner-first then outer; outermost value wins).
- **Retry-safe:** if outer retries (calls handler again), commands from earlier calls are discarded.

### (F) `wrap_tool_call` monitoring (decorator)
```python
from collections.abc import Callable
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    print(f"Executing tool: {request.tool_call['name']}")
    print(f"Arguments: {request.tool_call['args']}")
    try:
        result = handler(request)
        print("Tool completed successfully")
        return result
    except Exception as e:
        print(f"Tool failed: {e}")
        raise
```
**WHY:** `wrap_tool_call` wraps EACH tool invocation. `request` is a `ToolCallRequest` exposing `.tool_call['name']`/`['args']`; `handler` runs the tool and returns `ToolMessage | Command`. Same wrap pattern as model calls (retry/short-circuit possible).

### (G) Dynamic prompt via `wrap_model_call` (content_blocks)
```python
from collections.abc import Callable
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import SystemMessage


@wrap_model_call
def add_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    new_content = list(request.system_message.content_blocks) + [
        {"type": "text", "text": "Additional context."}
    ]
    new_system_message = SystemMessage(content=new_content)
    return handler(request.override(system_message=new_system_message))
```
**WHY:** `request.system_message` is ALWAYS a `SystemMessage` (even if the agent was created with a string `system_prompt`). Use `.content_blocks` to read/modify as a list of blocks, append, and `override(system_message=...)`. (For Anthropic prompt caching, add `"cache_control": {"type": "ephemeral"}` to a block.)

## 7.4 Advanced concepts — EXECUTION ORDER & COMPOSITION

Given `middleware=[middleware1, middleware2, middleware3]`:

**Execution flow (verbatim):**
1. `middleware1.before_agent()`
2. `middleware2.before_agent()`
3. `middleware3.before_agent()`
— Agent loop starts —
4. `middleware1.before_model()`
5. `middleware2.before_model()`
6. `middleware3.before_model()`
— Wrap hooks nest like function calls —
7. `middleware1.wrap_model_call()` → `middleware2.wrap_model_call()` → `middleware3.wrap_model_call()` → model
— After hooks run in reverse —
8. `middleware3.after_model()`
9. `middleware2.after_model()`
10. `middleware1.after_model()`
— Agent loop ends —
11. `middleware3.after_agent()`
12. `middleware2.after_agent()`
13. `middleware1.after_agent()`

**Key rules (verbatim):**
- `before_*` hooks: first → last.
- `after_*` hooks: last → first (reverse).
- `wrap_*` hooks: nested (FIRST middleware wraps all others — it is the OUTERMOST layer).

**Implication:** place critical middleware FIRST (it runs first for before-hooks, last for after-hooks, and is outermost for wrap-hooks → it sees the final result and wins conflicts). "Consider execution order — place critical middleware first in the list."

**Best practices (verbatim list):** (1) keep middleware focused; (2) handle errors gracefully (don't crash the agent); (3) use node-style for sequential logic, wrap-style for control flow; (4) document custom state properties; (5) unit-test middleware independently; (6) consider execution order, critical first; (7) use built-in middleware when possible.

## 7.5 Cross-framework interaction points
- Custom middleware ↔ agent loop: each hook fires at a precise loop point — `before_agent`/`after_agent` (once per invoke), `before_model`/`after_model` (per model call), `wrap_model_call`/`wrap_tool_call` (around each call).
- Custom middleware ↔ LangGraph reducers: node-style dict returns and `Command` updates flow through the graph's reducers (`messages` additive).
- Custom middleware ↔ state: `state_schema` extends `AgentState`; shared across hooks.
- Custom middleware ↔ tools: `tools` class attribute registers tools (e.g. `write_todos`).
- Custom middleware ↔ event streaming: `transformers` class attribute registers stream transformers (`langchain>=1.3.2`); built-in `ToolCallTransformer` stays in front, caller-supplied land last.
- Custom middleware ↔ guardrails/memory: guardrails (§4) and memory trimming/summarizing (§1) are concrete custom-middleware applications.

## 7.6 Gotchas / version notes
- Wrap-style hooks CANNOT update state by returning a dict — must use `ExtendedModelResponse`/`Command`.
- On retry, earlier wrap-style commands are discarded (retry-safe but be aware).
- Outermost (first-listed) middleware wins on conflicting non-reducer state fields.
- `request.system_message` is always a `SystemMessage` object — use `.content_blocks`.
- Stream transformers require `langchain>=1.3.2`.
- `@after_model @hook_config(can_jump_to=["end"])` stacking (decorator) is valid; or pass `can_jump_to` directly to `@after_model(...)`.
- Source note: the custom-middleware doc has some stray `:::` / "python" admonition artifacts around the class-attributes section (cosmetic; content intact).

---

## Reusable diagrams

### Diagram 1 — `before_model` placement in the agent loop (mermaid, verbatim from 09-short-term-memory.md)
```mermaid
%%{
    init: {
        "fontFamily": "monospace",
        "flowchart": {
        "curve": "basis"
        }
    }
}%%
graph TD
    S(["\_\_start\_\_"])
    PRE(before_model)
    MODEL(model)
    TOOLS(tools)
    END(["\_\_end\_\_"])
    S --> PRE
    PRE --> MODEL
    MODEL -.-> TOOLS
    MODEL -.-> END
    TOOLS --> PRE
    classDef blueHighlight fill:#E5F4FF,stroke:#006DDD,color:#030710;
    classDef neutral fill:#F2FAFF,stroke:#40668D,stroke-width:2px,color:#2F4B68;
    class S blueHighlight;
    class END blueHighlight;
    class PRE,MODEL,TOOLS neutral;
```

### Diagram 2 — `after_model` placement in the agent loop (mermaid, verbatim)
```mermaid
%%{
    init: {
        "fontFamily": "monospace",
        "flowchart": {
        "curve": "basis"
        }
    }
}%%
graph TD
    S(["\_\_start\_\_"])
    MODEL(model)
    POST(after_model)
    TOOLS(tools)
    END(["\_\_end\_\_"])
    S --> MODEL
    MODEL --> POST
    POST -.-> END
    POST -.-> TOOLS
    TOOLS --> MODEL
    classDef blueHighlight fill:#E5F4FF,stroke:#006DDD,color:#030710;
    classDef greenHighlight fill:#F6FFDB,stroke:#6E8900,color:#2E3900;
    classDef neutral fill:#F2FAFF,stroke:#40668D,stroke-width:2px,color:#2F4B68;
    class S blueHighlight;
    class END blueHighlight;
    class POST greenHighlight;
    class MODEL,TOOLS neutral;
```

### Image references (not mermaid, but worth recreating in synthesis)
- `core_agent_loop.png` — the 2-step loop (model call ↔ tool execution).
- `middleware_final.png` — middleware hooks layered around the agent loop (used in context-engineering, guardrails, middleware overview).
- `summary.png` — summarization replacing old messages.

### Diagram 3 — PROPOSED middleware hook execution order (synthesize this)
```mermaid
flowchart TD
    START([invoke]) --> BA1[mw1.before_agent]
    BA1 --> BA2[mw2.before_agent]
    BA2 --> BA3[mw3.before_agent]
    BA3 --> LOOP{{agent loop}}
    LOOP --> BM1[mw1.before_model]
    BM1 --> BM2[mw2.before_model]
    BM2 --> BM3[mw3.before_model]
    BM3 --> W1[mw1.wrap_model_call OUTER]
    W1 --> W2[mw2.wrap_model_call]
    W2 --> W3[mw3.wrap_model_call INNER]
    W3 --> MODEL[(model call)]
    MODEL --> AM3[mw3.after_model]
    AM3 --> AM2[mw2.after_model]
    AM2 --> AM1[mw1.after_model]
    AM1 --> TOOLS{tools needed?}
    TOOLS -- yes: wrap_tool_call nests same way --> LOOP
    TOOLS -- no --> AA3[mw3.after_agent]
    AA3 --> AA2[mw2.after_agent]
    AA2 --> AA1[mw1.after_agent]
    AA1 --> DONE([response])
```
Caption to use: "before_* runs first→last; after_* runs last→first; wrap_* nests with the FIRST-listed middleware as the OUTERMOST layer (so it sees the final result and wins conflicts). Place critical middleware first."

### Diagram 4 — PROPOSED context-engineering map (synthesize this)
```mermaid
flowchart LR
    subgraph SOURCES[Data sources]
        RC[Runtime Context\nstatic config]
        ST[State\nshort-term memory]
        STORE[Store\nlong-term memory]
    end
    subgraph LEVERS[Context types]
        MC[Model Context\nprompt/messages/tools/model/format\nTRANSIENT]
        TC[Tool Context\ntool reads+writes\nPERSISTENT]
        LC[Life-cycle Context\nsummarize/guardrails/log\nPERSISTENT]
    end
    RC --> MC & TC & LC
    ST --> MC & TC & LC
    STORE --> MC & TC & LC
    MC -. via wrap_model_call/dynamic_prompt .-> MODEL[(LLM call)]
    LC -. via before/after hooks .-> MODEL
    TC -. via ToolRuntime + Command/store.put .-> TOOLS[(tools)]
```

---

## APPENDIX — Import cheat-sheet (for fidelity)
- `from langchain.agents import create_agent, AgentState`
- `from langchain.agents.middleware import (before_agent, before_model, after_model, after_agent, wrap_model_call, wrap_tool_call, dynamic_prompt, hook_config, AgentMiddleware, AgentState, ModelRequest, ModelResponse, ExtendedModelResponse)`
- Built-ins: `from langchain.agents.middleware import (SummarizationMiddleware, HumanInTheLoopMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware, ModelFallbackMiddleware, PIIMiddleware, TodoListMiddleware, LLMToolSelectorMiddleware, ToolRetryMiddleware, ModelRetryMiddleware, LLMToolEmulator, ContextEditingMiddleware, ClearToolUsesEdit, ShellToolMiddleware, HostExecutionPolicy, DockerExecutionPolicy, CodexSandboxExecutionPolicy, RedactionRule, FilesystemFileSearchMiddleware)`
- Deep Agents: `from deepagents.middleware.filesystem import FilesystemMiddleware`; `from deepagents.middleware.subagents import SubAgentMiddleware`; `from deepagents import CompiledSubAgent`; `from deepagents.backends import CompositeBackend, StateBackend, StoreBackend`
- Memory/persistence (FROM LangGraph): `from langgraph.checkpoint.memory import InMemorySaver`; `from langgraph.checkpoint.postgres import PostgresSaver`; `from langgraph.store.memory import InMemoryStore`; `from langgraph.store.postgres import PostgresStore`; `from langgraph.store.base import IndexConfig`
- Messages: `from langchain.messages import RemoveMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage`; `from langgraph.graph.message import REMOVE_ALL_MESSAGES`
- Tools: `from langchain.tools import tool, ToolRuntime`; `from langchain.tools.tool_node import ToolCallRequest`
- Types: `from langgraph.types import Command`; `from langgraph.runtime import Runtime`; `from langchain_core.runnables import RunnableConfig, Runnable`; `from langchain.chat_models import init_chat_model`
- Model id formats seen: `"anthropic:claude-sonnet-4-6"`, `"openai:gpt-5.4"`, `"gpt-5.4-mini"`, `"claude-haiku-4-5-20251001"`, `"claude-3-5-sonnet-20241022"` (fallback example).
