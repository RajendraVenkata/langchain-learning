---

## Part 3 — LangGraph: The Orchestration Runtime Underneath

In Part 2 we learned that `create_agent` *is* a LangGraph graph. Part 3 zooms into LangGraph itself — the low‑level framework that provides everything the harness gets "for free." You can use `create_agent` for a long time without touching LangGraph directly, but understanding it is what lets you reason about durability, streaming, and human‑in‑the‑loop with confidence — and it's what you drop down to when your control flow outgrows "loop until done" (Part 7).

### 3.1 Purpose

The LangChain history page tells the origin story precisely: the original LangChain had high‑level abstractions but **"was missing a low‑level orchestration layer that allowed developers to control the exact flow of their agent."** LangGraph filled that gap, and while building it the team added what reliable agents actually need: **streaming, durable execution, short‑term memory, human‑in‑the‑loop, and more.** By late 2024 it became the preferred way to build any AI app that is more than a single LLM call.

So LangGraph's purpose is **controllable, durable, stateful orchestration**. Where `create_agent` says "here's a great default loop," LangGraph says "here's the machine the loop runs on, and you can reshape it however you want."

### 3.2 Building blocks — the graph model

- **`StateGraph(StateType)`** (`langgraph.graph`) — the graph builder. `StateType` is typically a `TypedDict` (or `AgentState` subclass).
- **Nodes** (`.add_node(name, fn)`) — a node is a function `(state) -> partial state update`. It can be a plain function, a model call, or an entire `create_agent` agent.
- **Edges** (`.add_edge(a, b)`) and **conditional edges** (`.add_conditional_edges(src, routing_fn, [targets])`) — define control flow, including loops and branches.
- **`START`, `END`** — the entry and exit sentinels.
- **Reducers** — annotate a state field with how concurrent/successive writes combine, e.g. `messages: Annotated[list, add_messages]` (append, don't overwrite) or `results: Annotated[list, operator.add]` (concatenate parallel results). Reducers are *why* parallel branches and tool calls don't clobber each other.
- **`.compile(checkpointer=..., store=...)`** — produce a runnable `CompiledStateGraph` with `.invoke`/`.stream`/`.astream`.
- **Control primitives** (`langgraph.types`): **`Command`** (update state and/or `goto` another node, possibly in the parent graph via `Command.PARENT`) and **`Send`** (fan out to a node with a specific sub‑state — the basis of parallel map/router patterns).

### 3.3 Durable execution & persistence — the checkpointer

This is LangGraph's defining feature and the reason agents survive contact with the real world.

#### Purpose

Long‑running agents fail in messy ways: a tool times out, a process restarts, a rate limit hits mid‑loop. **Durable execution** means the graph snapshots its state after each step (super‑step) to a **checkpointer**, so it can *resume from exactly where it left off* rather than restarting. The same snapshots enable conversation memory, human‑in‑the‑loop pauses, and time‑travel.

#### Building blocks & code

- **Checkpointers** (short‑term memory / durability): `InMemorySaver` (`langgraph.checkpoint.memory`, dev), `PostgresSaver` / `AsyncPostgresSaver` (`langgraph.checkpoint.postgres`, prod). Selected per‑thread via `config={"configurable": {"thread_id": ...}}`.
- **Stores** (long‑term memory): `InMemoryStore`, `PostgresStore` (`langgraph.store.*`), with `IndexConfig` for semantic search.

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.agents import create_agent

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()                       # create tables once
    agent = create_agent("gpt-5.5", tools=[...], checkpointer=checkpointer)
```

The same checkpointer object is what powers *all three* of: multi‑turn memory (Part 2.5), human‑in‑the‑loop resume (3.5), and time‑travel (3.6). One primitive, three superpowers.

### 3.4 Streaming — seeing the loop as it runs

#### Purpose

LLM latency is real; showing output progressively is a huge UX win. LangGraph streams at multiple granularities because different UIs need different things — raw tokens for a typing effect, step updates for "now running tool X," custom events for domain progress.

#### Building blocks & code

The lower‑level API is `agent.stream(..., stream_mode=...)` (which *is* `CompiledStateGraph.stream`):

- **`stream_mode="updates"`** — state delta after each node (coarse: "the model node produced X").
- **`stream_mode="messages"`** — `(token, metadata)` tuples, token‑by‑token from any LLM node.
- **`stream_mode="custom"`** — arbitrary data emitted from inside a node via `get_stream_writer()`.
- **`stream_mode="values"`** — the full state snapshot after each step.
- Pass a list to combine modes: `stream_mode=["updates", "messages"]`.

```python
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for {city}")      # surfaces in stream_mode="custom"
    return f"It's always sunny in {city}!"

for chunk in agent.stream({"messages": [{"role": "user", "content": "Weather in SF?"}]},
                          stream_mode="custom"):
    print(chunk["data"])
```

On top of this, LangChain (v1.3+) adds **Event Streaming** — `agent.stream_events(input, version="v3")` — a *higher‑level, typed* API that returns a run object with **independent projections** so you don't branch on chunk types:

```python
stream = agent.stream_events({"messages": [{"role": "user", "content": "Weather in SF?"}]},
                             version="v3")
for message in stream.messages:        # one per LLM call
    for delta in message.text:         # live token deltas
        print(delta, end="", flush=True)
final_state = stream.output            # drained final state
```

Other projections: `stream.tool_calls` (tool execution lifecycle — inputs, output deltas, errors), `stream.values` (state snapshots), `stream.subagents` (named sub‑agents), `stream.interrupt`‑style events. The mental model: **same graph execution underneath; cleaner consumer ergonomics on top.** For new apps the docs recommend Event Streaming; the `stream_mode` API is the mechanism it's built on.

### 3.5 Human‑in‑the‑loop (HITL) — pausing for a human

#### Purpose

When the model proposes a risky action — write a file, run a `DELETE`, send an email — you often want a human to approve, edit, or reject it *before* the side effect happens. HITL is "pause the graph, persist its state, wait for a human decision, then resume from the exact point." It is the textbook payoff of durable execution.

#### Building blocks & code

- **`interrupt(payload)`** (`langgraph.types`) — halts the graph and surfaces a payload for review.
- **`Command(resume=...)`** — re‑enter the paused graph with the human's decision.
- **A checkpointer is mandatory** — without it there is no persisted state to resume.
- The built‑in **`HumanInTheLoopMiddleware`** (Part 4) wires this up declaratively for tool calls.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

agent = create_agent(
    model="gpt-5.4",
    tools=[write_file, execute_sql, read_data],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={
        "write_file": True,                                  # all decisions allowed
        "execute_sql": {"allowed_decisions": ["approve", "reject"]},  # no editing
        "read_data": False,                                  # auto-approve safe op
    })],
    checkpointer=InMemorySaver(),                            # REQUIRED for interrupts
)

cfg = {"configurable": {"thread_id": "t1"}}
result = agent.invoke({"messages": [{"role": "user", "content": "Delete old records"}]},
                      cfg, version="v2")
print(result.interrupts)   # GraphOutput.interrupts -> the action(s) awaiting a decision

# A human approves; resume on the SAME thread_id:
agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg, version="v2")
```

The four decision types are **approve** (run as‑is), **edit** (run with modified args), **reject** (don't run; the message becomes feedback to the model), and **respond** (the human's message *becomes* the tool result — for "ask the user" tools). When streaming, interrupts surface in `stream_mode="updates"` under the `"__interrupt__"` key — the bridge between streaming and HITL.

### 3.6 Time‑travel — navigating and forking the past

Because every super‑step persists a checkpoint, you can list a thread's history, rewind to any checkpoint, and **fork** a new branch by re‑executing from there. This is the basis of the "edit / retry / branch / audit" experiences in the frontend (Part 10). The same machinery — checkpoints — that durably resumes a crashed run also lets a user explore alternate histories. Concretely: `client.threads.getHistory(thread_id)` returns the list of `ThreadState`s, and resuming with a `forkFrom` checkpoint id rolls back and re‑runs, preserving the original timeline as a sibling branch.

### 3.7 The overall picture — LangGraph

```mermaid
flowchart TB
    subgraph GRAPH["Compiled StateGraph"]
      direction LR
      S([START]) --> N1[node A]
      N1 -- "conditional edge" --> N2[node B]
      N1 -- Send(parallel) --> N3[node C]
      N2 --> E([END])
      N3 --> E
    end
    STATE["Typed State<br/>fields + reducers<br/>(add_messages, operator.add)"] -. read/update .- GRAPH
    CP[(Checkpointer<br/>InMemory / Postgres)] -- "snapshot per super-step" --> GRAPH
    CP --> DUR["Durable execution<br/>resume after crash"]
    CP --> MEM["Short-term memory<br/>thread_id"]
    CP --> HITL["Human-in-the-loop<br/>interrupt → Command(resume)"]
    CP --> TT["Time-travel<br/>getHistory → forkFrom"]
    GRAPH -- "stream_mode: updates/messages/custom/values" --> STR["Streaming<br/>(+ Event Streaming v3 projections)"]
    STORE[(Store<br/>long-term memory)] -. cross-thread .- GRAPH

    classDef g fill:#F6FFDB,stroke:#6E8900,color:#2E3900
    classDef n fill:#F2FAFF,stroke:#40668D,color:#2F4B68
    class S,N1,N2,N3,E g
    class STATE,CP,STORE,STR,DUR,MEM,HITL,TT n
```

**One checkpointer, four superpowers** (durability, memory, HITL, time‑travel) — that single insight explains most of LangGraph's value. Everything `create_agent` does sits on top of this machine, which is why the agent inherits all of it.
