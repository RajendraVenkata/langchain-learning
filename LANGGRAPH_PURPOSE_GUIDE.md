# The Purpose‑Driven Guide to LangGraph

> A single, top‑down, *why‑first* walkthrough of **LangGraph** — the low‑level orchestration runtime that powers reliable, long‑running, stateful agents — and how it interlocks with LangChain, Deep Agents, LangSmith, the LangGraph Platform, and the Frontend SDK.
>
> Companion to `LANGCHAIN_PURPOSE_GUIDE.md`. That guide looked *down* from the agent (`create_agent`) and treated LangGraph as "the runtime underneath." **This guide flips the camera around**: LangGraph is the protagonist, and `create_agent`/Deep Agents are things built *on top of* it. For every layer we ask: **What is its purpose? What would you suffer without it? Which APIs, patterns, and primitives achieve it?** — then we read the code. Wherever two frameworks (or two APIs) meet, the seam is explained **twice — once from each side**.

---

## How to read this guide

LangGraph is deliberately **low‑level**. It does not abstract prompts or agent architecture; it gives you precise control over *execution*. So this guide is a descent into a *machine*: we start with why you'd want that control, build up the execution model, then the two ways to author it, then everything the runtime gives you for free (persistence, durability, streaming, human‑in‑the‑loop, time‑travel), and finally how to compose, operate, and ship it.

Each major section follows the same skeleton:

1. **Purpose** — the job this thing exists to do, and the pain without it.
2. **Building blocks** — the concrete classes, functions, parameters, patterns.
3. **Annotated code** — real code, explained block by block.
4. **Advanced concepts** — deeper ideas, in call‑out boxes.
5. **Two perspectives** — wherever this layer meets a *different* framework or API, the seam from both sides.
6. **The overall picture** — a Mermaid diagram.

> **A note on the code samples.** The docs use forward‑dated, illustrative model names (`gpt-5.5`, `claude-sonnet-4-6`, `gpt-5.4-mini`). Treat them as placeholders for "the current best model." The *shape* of the code is what's stable.

---

## Part 0 — The Big Why

### 0.1 The problem LangGraph exists to solve

The LangGraph overview states its purpose in one line:

> *"Gain control with LangGraph to design agents that reliably handle complex tasks."*

LangGraph is described as *"a low‑level orchestration framework and runtime for building, managing, and deploying long‑running, stateful agents,"* trusted in production by Klarna, Uber, J.P. Morgan, LinkedIn, Replit, and dozens more (Part 11). The keyword is **control**. High‑level abstractions (a one‑line agent) are wonderful until your app needs to do something the abstraction didn't anticipate: branch on a condition, run steps in parallel and merge them, pause for a human for three days, survive a process crash mid‑run, or replay history to debug. That is the gap LangGraph fills.

Its own positioning is explicit and humble: *"LangGraph is very low‑level, and focused entirely on agent **orchestration**."* If you just want a tool‑calling loop, the docs tell you to use LangChain's `create_agent`. You reach for LangGraph when you need the **underlying capabilities that make agents reliable**: **durable execution, streaming, human‑in‑the‑loop, and persistence** — and the freedom to shape the exact control flow.

### 0.2 The inversion: LangGraph is the foundation, not a detail

The companion LangChain guide's central thesis was *Agent = Model + Harness*, and it noted that the harness (`create_agent`) "compiles to a LangGraph graph." This guide is the view from the bedrock. The official stack diagram puts it plainly:

- **LangGraph** is *the orchestration runtime*: durable execution, streaming, human‑in‑the‑loop, persistence.
- **LangChain** is *the agent framework*: model/tool abstractions and prebuilt agent loops — **built on LangGraph**.
- **Deep Agents** is *an agent harness* (planning, subagents, filesystem, context management) — **built on LangGraph**.
- **LangSmith** is *the platform* for tracing, evaluation, and deployment across all of the above.

So `create_agent` is, from down here, *a prebuilt graph* — one very common graph shape that LangGraph hands you. Everything you learn about graphs, channels, checkpoints, and interrupts applies to it directly, because it *is* one. (We make this concrete and dual‑sided in Part 11.)

Crucially: **you can use LangGraph without LangChain at all.** It's inspired by Google's **Pregel** (large‑scale graph computation) and **Apache Beam**, with a public interface borrowing from **NetworkX**. The model/tool integrations are convenience, not requirement.

### 0.3 The shape of a LangGraph program — hello world

Three moves define every LangGraph app: **break the work into nodes, connect them with edges, share data through state.**

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

graph = StateGraph(MessagesState)   # 1. a graph over a shared State schema
graph.add_node(mock_llm)            # 2. a node = a function (state) -> state update
graph.add_edge(START, "mock_llm")   # 3. wire entry → node → exit
graph.add_edge("mock_llm", END)
graph = graph.compile()             # 4. compile to a runnable (a Pregel instance)

graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
```

That's the whole essence: **State + Nodes + Edges → compile → invoke.** Everything else in this guide is what you can layer onto that skeleton — reducers, checkpointers, streaming, interrupts, retries, subgraphs.

### 0.4 What you get "for free" once you're on the runtime

The reason to pay the low‑level tax is the set of **core benefits** the runtime provides to *any* stateful workflow:

| Capability | What it gives you | Part |
|---|---|---|
| **Persistence** | Agents that survive failures and resume from where they left off; conversation memory across turns. | 5 |
| **Durable execution** | Long‑running workflows that recover from crashes without re‑doing completed work. | 6 |
| **Human‑in‑the‑loop** | Inspect and modify agent state at *any* point; pause indefinitely for approval. | 8 |
| **Streaming** | Token‑by‑token output, per‑step progress, and custom events as the graph runs. | 7 |
| **Time travel** | Replay or fork from any past checkpoint to debug or explore alternatives. | 8 |
| **Production deployment** | Purpose‑built hosting for stateful, long‑running, background workloads. | 10 |
| **Observability** | Every step traced in LangSmith with no extra code. | 10 |

All of these rest on **one** machine — the Pregel runtime and its checkpointing — which is why Part 2 (the execution model) is the conceptual key that unlocks everything after it.

### 0.5 The master map

Here is the whole picture. Every later part is a zoom‑in on one box or one arrow.

```mermaid
flowchart TB
    subgraph AUTHOR["✍️ AUTHOR — two ways, one runtime"]
        GA["Graph API<br/>StateGraph · nodes · edges · reducers<br/><i>declarative, visual</i>"]
        FA["Functional API<br/>@entrypoint · @task<br/><i>imperative, minimal change</i>"]
    end
    subgraph RUNTIME["⚙️ THE RUNTIME"]
        PR["Pregel<br/><i>actors + channels · Bulk Synchronous Parallel super-steps</i>"]
    end
    subgraph PERSIST["💾 PERSISTENCE (the foundation)"]
        CK["Checkpointer<br/><i>thread-scoped state snapshots</i>"]
        ST["Store<br/><i>cross-thread long-term memory</i>"]
    end
    subgraph GIVES["🎁 WHAT THE RUNTIME GIVES YOU"]
        DUR["Durable execution<br/>+ fault tolerance"]
        STR["Streaming<br/>+ event streaming v3"]
        HITL["Human-in-the-loop<br/>+ time travel"]
        SUB["Subgraphs<br/>composition"]
    end
    subgraph DELIVER["🚀 OPERATE & DELIVER"]
        STU["Studio · Deploy<br/>(LangGraph Platform)"]
        OBS["LangSmith<br/>observability + evals"]
        FE["Frontend SDK<br/>useStream"]
    end
    subgraph ONTOP["🏗️ BUILT ON TOP"]
        LC["LangChain create_agent<br/><i>a prebuilt graph</i>"]
        DA["Deep Agents"]
    end

    GA == "compile()" ==> PR
    FA == "@entrypoint" ==> PR
    PR --- CK
    PR --- ST
    CK --> DUR & HITL
    PR --> STR & SUB
    LC -. "IS a" .-> PR
    DA -. "built on" .-> PR
    PR -- "hosted by" --> STU
    PR -- "auto-traced" --> OBS
    STU -- "streams to" --> FE

    classDef author fill:#E5F4FF,stroke:#006DDD,color:#03070F
    classDef runtime fill:#F6FFDB,stroke:#6E8900,color:#2E3900
    classDef persist fill:#FFF6DB,stroke:#B8860B,color:#3A2E00
    classDef gives fill:#F2FAFF,stroke:#40668D,color:#2F4B68
    classDef deliver fill:#EBD0F0,stroke:#885270,color:#441E33
    classDef ontop fill:#FDF3FF,stroke:#7E65AE,color:#3C2E5A
    class GA,FA author
    class PR runtime
    class CK,ST persist
    class DUR,STR,HITL,SUB gives
    class STU,OBS,FE deliver
    class LC,DA ontop
```

**Read the map as a sentence:** You *author* a graph two ways (Graph or Functional API), both of which compile to the **Pregel runtime**. The runtime, anchored by **persistence** (checkpointer + store), gives you durable execution, streaming, human‑in‑the‑loop, time‑travel, and composition. You *operate* it via Studio/Deploy and observe it in LangSmith, and a frontend consumes it. LangChain's `create_agent` and Deep Agents are *built on this same runtime* — they are particular graphs.

Let's start where every LangGraph program starts: learning to *think* in nodes, edges, and shared state.


---

## Part 1 — Thinking in LangGraph

Before any API, LangGraph asks you to adopt a **mental model**. Get this right and the rest is mechanical; get it wrong and you'll fight the framework.

### 1.1 Purpose

The model is three moves, in order:

1. **Break the work into discrete steps** — each becomes a **node** (a function that does one thing).
2. **Describe the transitions** — how control moves from node to node (fixed edges or runtime decisions).
3. **Connect through shared state** — a single state object each node can read from and write to.

Why decompose at all? Because the decomposition is what *buys* you LangGraph's powers: nodes are the boundaries at which the runtime **streams progress**, **checkpoints state** (so it can pause/resume/recover), and lets you **inspect** what happened. A monolithic function gets none of that.

### 1.2 The five‑step method

The docs teach a repeatable recipe (illustrated with a customer‑support email agent):

1. **Map the workflow as discrete steps.** Sketch the nodes (`Read Email`, `Classify Intent`, `Doc Search`, `Draft Reply`, `Human Review`, `Send Reply`) and the possible paths. The actual *decision* of which path to take lives **inside** the nodes.
2. **Identify what each step needs.** Classify each node by type — **LLM** (understand/generate/decide), **Data** (retrieve), **Action** (side effects), **User‑input** (human intervention) — and note its static context (prompt) vs dynamic context (from state).
3. **Design the state.** State is the shared notebook. The rule: *include data that must persist across steps; don't store what you can derive.*
4. **Build the nodes.** Each is a function `(state) -> state update`; when it must also choose where to go next, it returns a `Command`.
5. **Wire it together.** Add the essential edges; nodes handle their own routing.

> **Advanced — the load‑bearing principle: store raw data, format prompts on demand.** Your state should hold *raw* data (the email text, the classification dict, the retrieved chunks), **not** formatted prompt strings. Format prompts *inside nodes* when needed. This lets different nodes use the same data differently, lets you change prompt templates without touching the state schema, and makes debugging clear — you see exactly what data each node received.

#### Annotated code — state and a routing node

State is a typed schema of *raw* fields:

```python
from typing import TypedDict, Literal

class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str

class EmailAgentState(TypedDict):
    email_content: str          # raw inputs (can't reconstruct later)
    sender_email: str
    classification: EmailClassification | None   # results needed downstream
    search_results: list[str] | None             # expensive to refetch
    draft_response: str | None
```

A node does work and **routes itself** by returning a `Command` (state update *and* next destination in one object):

```python
from langgraph.types import Command

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "bug_tracking", "draft_response"]]:
    structured_llm = llm.with_structured_output(EmailClassification)
    classification = structured_llm.invoke(  # format the prompt on demand, from raw state
        f"Classify this email:\n{state['email_content']}\nFrom: {state['sender_email']}"
    )
    if classification['intent'] == 'billing' or classification['urgency'] == 'critical':
        goto = "human_review"
    elif classification['intent'] in ['question', 'feature']:
        goto = "search_documentation"
    elif classification['intent'] == 'bug':
        goto = "bug_tracking"
    else:
        goto = "draft_response"
    return Command(update={"classification": classification}, goto=goto)
```

The return annotation `Command[Literal[...]]` lists the node's possible destinations — LangGraph uses it to render the graph and validate routing. Because routing lives *in the node*, the wiring stays minimal:

```python
from langgraph.checkpoint.memory import MemorySaver

workflow = StateGraph(EmailAgentState)
workflow.add_node("classify_intent", classify_intent)
# ... add other nodes ...
workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)
app = workflow.compile(checkpointer=MemorySaver())   # checkpointer enables HITL + memory
```

### 1.3 Errors are part of the flow — the five‑way taxonomy

LangGraph's most distinctive teaching is that **errors aren't exceptional — they're a designed‑for part of the control flow.** Different failures need different handlers:

| Error type | Who fixes it | Strategy | LangGraph mechanism |
|---|---|---|---|
| **Transient** (network, rate limit, 5xx) | System (automatic) | retry with backoff | `RetryPolicy` (Part 6) |
| **LLM‑recoverable** (bad tool args, parse error) | the LLM | store error in state, loop back | `Command(goto=...)` back to the model |
| **User‑fixable** (missing info, ambiguity) | a human | pause for input | `interrupt()` (Part 8) |
| **Recoverable after retries** (payment failed N times) | developer (declarative) | compensation branch | `error_handler` → Saga (Part 6) |
| **Unexpected** (programmer bug) | developer | let it bubble up | don't catch it |

This table reappears throughout the guide because it's the spine of LangGraph reliability: each bucket maps to a specific primitive. Keep it in mind.

> **Advanced — node granularity is a real design decision.** Checkpoints land at node boundaries, so **smaller nodes = more frequent checkpoints = less work to redo on failure**. Separate nodes also isolate external services (give the API node a retry policy without affecting LLM nodes), expose intermediate state for inspection, and allow per‑node retry/timeout config. The cost is a little more wiring. The docs note performance is *not* a reason to merge nodes — checkpoints are written in the background by default (Part 5's durability modes), so frequent checkpoints are cheap.

### 1.4 The overall picture — the LangGraph way of thinking

```mermaid
flowchart TD
    P["1. Map workflow into discrete STEPS"] --> N["2. Each step = a NODE<br/>(LLM · data · action · user-input)"]
    N --> S["3. Design STATE<br/>(raw data only; derive the rest)"]
    S --> B["4. Build nodes:<br/>(state) → update, or Command(update, goto)"]
    B --> W["5. Wire essential EDGES<br/>(nodes self-route)"]
    W --> C["compile(checkpointer=...)"]
    C --> RUN(["invoke / stream"])
    ERR["Errors are part of the flow:<br/>transient→retry · LLM→loop ·<br/>user→interrupt · post-retry→error_handler ·<br/>unexpected→bubble up"] -. shapes .-> B
    classDef step fill:#E5F4FF,stroke:#006DDD,color:#03070F
    classDef note fill:#F8E8E6,stroke:#B27D75,color:#634643
    class P,N,S,B,W,C step
    class ERR note
```

With the mindset in place, we descend to the machine that actually runs these graphs — and it's the most important section in the guide.


---

## Part 2 — The Execution Model: Pregel & Bulk Synchronous Parallel

This is the conceptual key to all of LangGraph. **Pregel is the runtime** — the engine that actually executes your graph. Understand it and durability, parallelism, streaming, and time‑travel stop being magic and become obvious consequences of one design.

### 2.1 Purpose

Why does LangGraph have a runtime at all, distinct from the authoring APIs? Because it needs **one execution engine underneath both** the Graph API and the Functional API, so that the hard parts — checkpointing, parallelism, deterministic replay — are implemented *once* and shared. The central fact:

> **Compiling a `StateGraph` *or* creating an `@entrypoint` produces a `Pregel` instance** that can be invoked with input.

`Pregel` is named after [Google's Pregel algorithm](https://research.google/pubs/pub37252/) for large‑scale parallel graph computation. You rarely write Pregel directly, but everything you *do* write reduces to it.

### 2.2 Building blocks — actors + channels

Pregel combines two primitives into one application:

- **Actors** (`PregelNode`s) — they **read from channels and write to channels**. An actor is your node. (Actors implement LangChain's `Runnable` interface.)
- **Channels** — typed communication conduits between actors. Each channel has a value type, an update type, and an **update function** that folds a sequence of updates into the stored value. A channel can pass data from one actor to another, **or from an actor back to itself in a future step** — which is how cycles (loops) work.

The **channel types** are where merge‑and‑persist semantics live:

| Channel | Behavior | Reducer runs |
|---|---|---|
| **`LastValue`** (default) | stores the last value written, overwriting | n/a |
| **`Topic`** | PubSub; can `accumulate=True` or deduplicate across steps | n/a |
| **`BinaryOperatorAggregate`** | running aggregate via a binary op (e.g. `operator.add`) | **at write time** (combined value is serialized) |
| **`DeltaChannel`** (beta, ≥1.2) | stores only the per‑step *delta*, not the full value | **on reconstruction** (raw writes serialized) |
| **`EphemeralValue`** | transient value for the current pass; plumbing for edges | n/a |

When you write `Annotated[list, add_messages]` in a `StateGraph` (Part 3), you're choosing a channel and its update function. Channel choice directly controls how concurrent writes merge and how state is persisted.

### 2.3 The super‑step — Bulk Synchronous Parallel (BSP)

This is the heart. Pregel executes in **super‑steps**, each with three phases (verbatim):

- **Plan** — determine which actors to run this step. Step 0 selects actors subscribed to the **input** channels; later steps select actors subscribed to channels that were **updated in the previous step**.
- **Execution** — run **all selected actors in parallel** until all complete, one fails, or a timeout hits. *Channel updates are invisible to actors until the next step.*
- **Update** — apply all the writes to the channels (via their update functions / reducers).

Repeat **until no actors are selected** (nothing was updated last step) or a max step count is reached.

Three profound consequences fall out of this design — and they *are* the rest of the guide:

> **Advanced — why BSP gives you everything.**
> - **Durability (Part 5–6):** because the **Update** phase produces a clean, consistent snapshot of all channel values *between* steps, the super‑step boundary is the natural place to write a **checkpoint**. State == the set of channel values after a super‑step. Resume = re‑enter at a boundary.
> - **Parallelism (Part 3):** the **Execution** phase runs all selected actors *concurrently*. Independent nodes in the same step parallelize for free.
> - **Deterministic replay / time‑travel (Part 8):** because writes are applied as a batch in **Update** and are *invisible until the next step*, there is no read‑after‑write ordering between actors within a step. State transitions are reproducible from persisted writes.
>
> "Channel‑updates‑invisible‑until‑next‑step" is the single rule that makes the super‑step boundary the unit of consistency.

### 2.4 Annotated code — Pregel in the raw, and the proof that the APIs compile to it

You can build a Pregel app directly (you usually won't, but it demystifies everything):

```python
from langgraph.channels import EphemeralValue
from langgraph.pregel import Pregel, NodeBuilder

node1 = (
    NodeBuilder().subscribe_only("a")   # read channel "a" (bare value)
    .do(lambda x: x + x)                 # the actor's work
    .write_to("b")                       # write result to channel "b"
)
app = Pregel(
    nodes={"node1": node1},
    channels={"a": EphemeralValue(str), "b": EphemeralValue(str)},
    input_channels=["a"], output_channels=["b"],
)
app.invoke({"a": "foo"})   # {'b': 'foofoo'}
```

Plan selects `node1` (subscribed to input `a`); Execution doubles `"foo"`; Update writes `b="foofoo"`; nothing else subscribes to a changed channel → halt → return the output channel. A **cycle** is just a node writing to a channel it subscribes to, halting when it stops writing (`skip_none=True` on a `None`).

The load‑bearing proof — a `StateGraph` *is* a Pregel instance you can introspect:

```python
builder = StateGraph(Essay)   # Essay = TypedDict with topic/content/score
builder.add_node(write_essay); builder.add_node(score_essay)
builder.add_edge(START, "write_essay"); builder.add_edge("write_essay", "score_essay")
graph = builder.compile()     # returns a Pregel instance

print(graph.nodes)
# {'__start__': PregelNode, 'write_essay': PregelNode, 'score_essay': PregelNode}
print(graph.channels)
# {'topic': LastValue, 'content': LastValue, 'score': LastValue,
#  '__start__': EphemeralValue, 'branch:...': EphemeralValue, 'start:write_essay': EphemeralValue, ...}
```

Each **state key** (`topic`, `content`, `score`) became a **`LastValue` channel**; nodes became `PregelNode`s; edges/branches added `EphemeralValue` plumbing channels. The `@entrypoint` form is the same, with minimal channels (`__start__`, `__end__`, and `__previous__` for prior‑run state) — concrete proof both APIs are sugar over Pregel.

### 2.5 Two perspectives: the Pregel runtime ↔ the high‑level APIs

#### 👁️ From the API's perspective ("I'm a `StateGraph` / `@entrypoint`")

You think in *nodes, edges, shared state* (Graph API) or *functions, tasks, control flow* (Functional API). You call `.compile()` or decorate with `@entrypoint` and get something with `.invoke`/`.stream`. You never think about actors, channels, or super‑steps. You declare a reducer (`Annotated[list, add_messages]`) without knowing it becomes a channel update function; you write `add_edge(a, b)` without knowing it spawns `EphemeralValue` plumbing channels. The runtime is an implementation detail you benefit from.

#### 👁️ From Pregel's perspective ("I'm the runtime executing")

You see only **actors and channels** and you run **BSP super‑steps**. A "graph" is a dict of `PregelNode`s plus a dict of channels; an "edge" is plumbing channels that make one node's output trigger the next node's selection in the next Plan phase; a "reducer" is a channel's update function. You don't know about "agents" or "tools" — you Plan which actors to run, Execute them in parallel, Update channels, and repeat. `invoke` = run to halt and return output channels; `stream` = emit per super‑step; `checkpointer` = snapshot channel values at the Update boundary. From here, the Graph API and Functional API are merely two front‑ends that both produce the same `Pregel` object — which is exactly why they share every feature (persistence, streaming, HITL, memory) and can be combined in one app (Part 4).

### 2.6 The overall picture — the BSP super‑step loop

```mermaid
flowchart TD
    IN([input seeds input channels]) --> PLAN
    subgraph STEP["one SUPER-STEP"]
        direction TB
        PLAN["① PLAN<br/>select actors subscribed to<br/>channels updated last step"]
        PLAN --> EXEC["② EXECUTION<br/>run ALL selected PregelNodes IN PARALLEL<br/>‼ writes invisible to actors until next step"]
        EXEC --> UPD["③ UPDATE<br/>apply writes via channel update fns / reducers<br/>— consistent snapshot boundary —"]
    end
    UPD -->|"checkpoint here (Part 5)"| CP[(checkpointer)]
    UPD --> Q{any channels<br/>updated?}
    Q -->|yes| PLAN
    Q -->|"no actors selected / max steps"| OUT([return output channels])
    CHAN["Channels: LastValue · Topic ·<br/>BinaryOperatorAggregate · DeltaChannel · EphemeralValue"] -. store state .- UPD
    classDef phase fill:#F6FFDB,stroke:#6E8900,color:#2E3900
    classDef neutral fill:#F2FAFF,stroke:#40668D,color:#2F4B68
    class PLAN,EXEC,UPD phase
    class IN,OUT,Q,CP,CHAN neutral
```

**Parallelism lives in Execution; durability lives at the Update boundary; planning is driven purely by "which channels changed."** Hold this picture — every later part is a feature hanging off it.


---

## Part 3 — The Graph API (`StateGraph`)

The Graph API is the declarative, visual way to author a Pregel app — the one most people use most of the time. You define **State**, **Nodes**, and **Edges**, and `.compile()` turns them into the runnable from Part 2.

### 3.1 Purpose

The Graph API exists for workflows where you want **explicit control over structure**: complex branching, parallel paths that merge, shared state across many components, and a graph you can *visualize* for debugging and team collaboration. "Nodes do the work; edges tell what to do next." Both are just functions — they can wrap an LLM or plain code.

### 3.2 Building blocks

**The graph + state:**
- **`StateGraph(StateSchema, context_schema=..., input_schema=..., output_schema=...)`** — the builder. State schema is usually a `TypedDict` (default), a `dataclass` (when you want defaults), or a Pydantic `BaseModel` (runtime input validation; slower; note `create_agent` does *not* accept Pydantic state).
- **Reducers** — `Annotated[type, reducer]` attaches a per‑key update function. **No reducer ⇒ overwrite.** Common reducers: `operator.add` (append/concat), `add_messages` (from `langgraph.graph.message` — append‑or‑update messages by **ID**, and deserialize dicts to `Message` objects). `Overwrite(...)` (from `langgraph.types`) bypasses a reducer for one write.
- **`MessagesState`** — prebuilt state with a single `messages` key (`add_messages` reducer). Subclass to add fields.
- **Multiple schemas** — `input_schema`/`output_schema` constrain what `invoke` accepts/returns; nodes can write any channel in the union; a private schema carries internal‑only data. (Gotcha: private channels still appear in `stream_mode="values"` unless you pass `output_keys=`.)

**Nodes & edges:**
- **`.add_node(name, fn, retry_policy=, cache_policy=, error_handler=, timeout=, defer=)`** — a node fn takes `(state)`, optionally `config: RunnableConfig` and `runtime: Runtime`. Default name = function name.
- **`.add_edge(a, b)`** — always A→B. A node with multiple outgoing edges → all targets run **in parallel** next super‑step.
- **`.add_conditional_edges(src, routing_fn, path_map=None)`** — `routing_fn(state)` returns the next node name(s); the optional dict maps return values → node names. Return a *list* to fan out.
- **`START` / `END`** — entry/exit sentinels (`add_edge(START, "n")`, `add_edge("n", END)`).
- **`.compile(checkpointer=, store=, cache=, interrupt_before=, interrupt_after=, name=)`** — **you must compile before use.**

**Dynamic control primitives** (`langgraph.types`):
- **`Command(update=, goto=, graph=, resume=)`** — return from a node to update state **and** route in one move; `graph=Command.PARENT` navigates to the parent graph (multi‑agent handoffs); `resume` is the input form for HITL (Part 8).
- **`Send(node_name, state)`** — returned from a conditional edge to **dynamically fan out**: one invocation of `node_name` per item, each with a bespoke state (map‑reduce).

**Reliability & limits** (detailed in Part 6): `RetryPolicy`, `CachePolicy`/`InMemoryCache`, `recursion_limit` (max super‑steps; default **1000** since v1.0.6 → `GraphRecursionError`), `RemainingSteps` for graceful degradation. **Visualization:** `graph.get_graph().draw_mermaid()` / `.draw_mermaid_png()`.

### 3.3 Annotated code

**State + reducer — the single most consequential decision:**

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain.messages import AnyMessage, AIMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]   # append + update-by-id
    extra_field: int                                       # no reducer → overwrite

def node(state: State):
    return {"messages": [AIMessage("Hello!")], "extra_field": 10}   # reducer appends the message
```

`add_messages` is what makes chat state work (append new, overwrite by ID — needed for HITL edits and dedup); `operator.add` would only ever append; no reducer would overwrite the whole list each turn.

**`Command`‑based routing — update state AND choose the next node:**

```python
import random
from typing import Literal
from langgraph.types import Command

def node_a(state: State) -> Command[Literal["node_b", "node_c"]]:
    value = random.choice(["b", "c"])
    return Command(
        update={"foo": value},                 # state update
        goto="node_b" if value == "b" else "node_c",   # routing — replaces an edge
    )
# NOTE: no edges between A, B, C — control flow lives in the node.
```

Conditional edges *route only* (no state change); `Command` does both. The `Command[Literal[...]]` annotation is mandatory so LangGraph knows the targets.

**`Send` — map‑reduce with a runtime‑decided number of branches:**

```python
import operator
from langgraph.types import Send

class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]   # reducer MERGES parallel results (fan-in)

def continue_to_jokes(state: OverallState):     # a conditional edge
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

def generate_joke(state):                       # reads state["subject"] — injected by Send
    return {"jokes": [f"a joke about {state['subject']}"]}

builder.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
```

Each `Send` carries a *different* state into the same node (the **map**); the `operator.add` reducer on `jokes` performs the **fan‑in/merge**; a downstream node is the **reduce**. The branch count is unknown at author time — this is the canonical map‑reduce.

**Parallel fan‑out + fan‑in:**

```python
class State(TypedDict):
    aggregate: Annotated[list, operator.add]   # reducer needed for safe concurrent writes

builder.add_edge("a", "b")   # a fans out to b and c
builder.add_edge("a", "c")   # (same super-step, concurrent)
builder.add_edge("b", "d")   # d is fan-in: runs only after BOTH b and c finish
builder.add_edge("c", "d")
```

Without the reducer, concurrent writes to `aggregate` would conflict. The reducer is *the* mechanism for safe parallel fan‑in. (Update **ordering** across a parallel step is not guaranteed — use a separate ordered field if order matters.)

**A loop (the ReAct shape) with explicit termination:**

```python
def route(state: State) -> Literal["b", END]:
    return "b" if len(state["aggregate"]) < 7 else END

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)   # a → (loop to b) or END
builder.add_edge("b", "a")                  # b loops back to a
```

Loops **require** an explicit termination edge to `END`; `recursion_limit` is the runaway safety net. (`a`=model, `b`=tools is the agent loop.)

> **Advanced — `Command` vs conditional edges, and the "don't mix" rule.** Use a **conditional edge** when you only need to pick the next node; use **`Command`** when you must update state *and* route, or for subgraph→parent handoffs (`Command.PARENT`), or to route from a tool. Critically: `Command`/conditional edges add *dynamic* routing **on top of** static `add_edge`s — if a node has both a static edge and returns a `Command(goto=...)`, **both fire**. Pick one mechanism per node.

> **Advanced — nodes re‑run from the start on resume; make them idempotent.** Checkpoints are at super‑step boundaries, not mid‑node. After an interrupt/retry, the affected node re‑executes *from the beginning of its function* — code before the pause runs again. Design side effects to be idempotent (upserts, idempotency keys). This rule recurs in Parts 6 and 8.

### 3.4 The overall picture — anatomy of a `StateGraph`

```mermaid
flowchart TB
    subgraph AUTHOR["Author time — StateGraph builder"]
        SCH["State schema (TypedDict / dataclass / Pydantic)<br/>Annotated[type, reducer] per key"]
        N["add_node(name, fn,<br/>retry_policy / cache_policy / error_handler / timeout / defer)"]
        E1["add_edge(START, n) · add_edge(a, b)"]
        E2["add_conditional_edges(src, route_fn, path_map)<br/>(route_fn may return Send[] for fan-out)"]
        SCH --> N --> E1 --> E2
    end
    AUTHOR -->|".compile(checkpointer=, store=, cache=,<br/>interrupt_before/after=, name=)"| COMPILED
    subgraph COMPILED["Compiled graph = a Pregel runnable (Part 2)"]
        CH["channels == state keys · reducers == update fns"]
        SS["super-step engine · parallel = same step"]
        CH --- SS
    end
    COMPILED -->|"invoke / stream<br/>config={recursion_limit, max_concurrency} · context={...}"| RUN["Run<br/>node returns dict | Command(update, goto, graph)"]
    RUN --> CH
    classDef a fill:#E5F4FF,stroke:#006DDD,color:#03070F
    classDef c fill:#F6FFDB,stroke:#6E8900,color:#2E3900
    class SCH,N,E1,E2 a
    class CH,SS,RUN c
```


---

## Part 4 — The Functional API & the Workflow↔Agent Spectrum

The Graph API asks you to think in graphs. The **Functional API** asks for almost nothing — write ordinary Python, sprinkle two decorators, and get LangGraph's powers anyway.

### 4.1 Purpose

The Functional API adds **persistence, memory, human‑in‑the‑loop, and streaming** to existing code **with minimal changes**. You keep standard control flow — `if`, `for`, function calls — instead of declaring nodes/edges/state. There's no rigid DAG and (usually) no explicit state schema or reducers: **state is scoped to the function.** It's the on‑ramp for procedural code and rapid prototyping.

### 4.2 Building blocks

- **`@entrypoint(checkpointer=, store=, cache=, timeout=)`** — marks the workflow's start; **produces a `Pregel` instance** (same runtime as the Graph API). Takes a **single** positional input (use a dict for multiple). Inputs/outputs must be **JSON‑serializable** (for checkpointing). Injectable params (by name + annotation): **`previous`** (prior‑run state — short‑term memory), **`store`** (`BaseStore` — long‑term memory), **`writer`** (`StreamWriter`), **`config`** (`RunnableConfig`).
- **`@task(retry_policy=, cache_policy=, timeout=)`** — a discrete unit of work. **Returns a future**; resolve with `.result()` (sync) or `await`. **Task results are checkpointed** — on resume they're restored, not recomputed. Tasks can only be called from inside an entrypoint, another task, or a graph node.
- **`entrypoint.final(value=, save=)`** — decouple the value *returned to the caller* from the value *saved to the checkpoint* (the next run's `previous`).
- **Resume / HITL:** `interrupt()` inside the body; resume with `Command(resume=...)`. Resume‑after‑error: `workflow.invoke(None, config)` (same `thread_id`).
- **Parallelism:** call multiple tasks, gather their futures (`[f.result() for f in futures]`).

### 4.3 Annotated code

The foundational pattern — a task for the work, an entrypoint that orchestrates and pauses for a human:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import interrupt, Command

@task
def write_essay(topic: str) -> str:
    """A long-running unit of work — its result is checkpointed."""
    return f"An essay about topic: {topic}"

@entrypoint(checkpointer=InMemorySaver())
def workflow(topic: str) -> dict:
    essay = write_essay(topic).result()          # future → value; result saved to checkpoint
    is_approved = interrupt({"essay": essay, "action": "Approve?"})  # pause for a human
    return {"essay": essay, "is_approved": is_approved}

config = {"configurable": {"thread_id": "1"}}
workflow.invoke("cats", config)                  # runs write_essay, then pauses at interrupt
workflow.invoke(Command(resume=True), config)    # resumes; write_essay NOT recomputed
```

Parallelism is just gathering futures:

```python
@task
def generate_paragraph(topic: str) -> str:
    return model.invoke(f"Write a paragraph about {topic}.").content

@entrypoint(checkpointer=InMemorySaver())
def workflow(topics: list[str]) -> str:
    futures = [generate_paragraph(t) for t in topics]   # kicked off concurrently
    return "\n\n".join(f.result() for f in futures)     # gather (I/O-bound → real speedup)
```

Cross‑run memory uses `previous` + `entrypoint.final`:

```python
@entrypoint(checkpointer=InMemorySaver())
def accumulate(n: int, *, previous: int | None) -> entrypoint.final[int, int]:
    previous = previous or 0
    return entrypoint.final(value=previous, save=previous + n)  # return prev, save new total

# invoke(1)→0, invoke(2)→1, invoke(3)→3   (previous carries across runs on the same thread)
```

> **Advanced — why side effects MUST live in `@task` (the determinism rule).** On resume, execution does **not** continue from the line where it stopped. It returns to a checkpoint boundary and **replays the entrypoint body forward**, restoring completed `@task`/subgraph results from the checkpoint instead of recomputing them, until it reaches the pause again. So a bare side effect in the entrypoint body (a file write, an email send, `time.time()`, `random()`) **runs again on every replay** — and a branch that depends on wall‑clock time may take a different path on resume, mismatching the index‑based interrupt↔resume pairing. The fix is mechanical: **wrap every side effect and every non‑deterministic value in a `@task`** (its output is checkpointed and replayed), and make those tasks **idempotent** (a task that started but didn't finish may re‑run).

### 4.4 Two perspectives: Graph API ↔ Functional API

Both produce a `Pregel` app and offer the **same** features (persistence, streaming, HITL, memory). They differ only in paradigm — so understanding the seam lets you pick well and even combine them.

#### 👁️ From the Graph API's perspective ("I'm declarative")

You externalize structure: a `StateGraph` with explicit `State` (and reducers), nodes, and edges. Control flow is *data* — `add_conditional_edges`, `Command`, `Send` — which means it's **visualizable** (`draw_mermaid_png()`), inspectable, and parallelizable by construction (fan‑out edges run in one super‑step). A new checkpoint is written **after every super‑step**. You choose this for complex branching, parallel‑and‑merge, shared cross‑node state, and team collaboration. From here, an `@entrypoint` looks like *a single node* you can call (`entrypoint.invoke(...)`) from inside one of your nodes.

#### 👁️ From the Functional API's perspective ("I'm imperative")

You keep ordinary Python: `if/else`, `for`, function calls, `.result()`. There's no `State` schema and no reducers — state is **function‑scoped**, carried across runs via `previous`/`entrypoint.final`. Checkpointing differs subtly: instead of a new checkpoint per super‑step, **`@task` results are saved into the entrypoint's existing checkpoint**. You don't get a graph picture (the graph is generated at runtime). You choose this to add LangGraph features to existing procedural code with minimal refactor, for linear workflows, and for rapid prototyping. From here, a compiled `StateGraph` is just a `Runnable` you call (`graph.invoke(...)`) when one part of your app genuinely needs a graph.

They **interoperate** because both are Pregel: a `StateGraph` node can call `entrypoint.invoke(...)`, and an `@entrypoint` can call `graph.invoke(...)`. Migration is mechanical too — imperative `if/else` ↦ `add_conditional_edges` + router functions; local variables ↦ explicit `TypedDict` state keys (and back).

### 4.5 The workflow↔agent spectrum

LangGraph frames a continuum: **workflows** have predetermined code paths (the developer directs control flow); **agents** are dynamic (the **LLM** directs control flow). LangGraph serves the whole spectrum, and every pattern can be built with *either* API:

| Pattern | What it does |
|---|---|
| **Prompt chaining** | Each LLM call processes the previous output, often with a "gate" check between steps. |
| **Parallelization** | Run independent subtasks at once (or the same task N times to compare). |
| **Routing** | Classify input, then dispatch to a specialized flow (structured‑output router). |
| **Orchestrator‑worker** | Orchestrator splits a task into runtime‑decided subtasks, workers run them (Graph version uses `Send`), then synthesize. |
| **Evaluator‑optimizer** | One LLM generates, another evaluates; loop until acceptable. |
| **Agent** | LLM runs a tool‑calling loop until done — unpredictable problems/solutions. |

The prebuilt building blocks: **`create_agent`** (LangChain's agent, a prebuilt graph — Part 11) and **`ToolNode`** (`langgraph.prebuilt`, a node that executes tool calls with parallel execution, error handling, and state injection).

```python
# Orchestrator-worker with Send (Graph API): runtime-decided number of workers
def assign_workers(state):
    return [Send("llm_call", {"section": s}) for s in state["sections"]]
builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
builder.add_edge("llm_call", "synthesizer")   # workers write to an operator.add channel; synthesizer reduces
```

### 4.6 The overall picture — two front‑ends, one runtime; one spectrum

```mermaid
flowchart TB
    subgraph G["Graph API (declarative)"]
        G1["StateGraph · add_node · add_edge<br/>add_conditional_edges · Command · Send<br/>explicit State + reducers · visual"]
    end
    subgraph F["Functional API (imperative)"]
        F1["@entrypoint · @task · .result()<br/>standard if/for · function-scoped state<br/>previous / entrypoint.final · minimal code"]
    end
    G1 -->|"compile()"| PR
    F1 -->|"@entrypoint"| PR
    PR["SAME RUNTIME: Pregel<br/>same features: persistence · streaming · HITL · memory<br/>interop: a node can call entrypoint.invoke(); an entrypoint can call graph.invoke()"]
    PR --> SPECTRUM["Both express the WORKFLOW → AGENT spectrum:<br/>prompt chaining → parallelization → routing →<br/>orchestrator-worker → evaluator-optimizer → AGENT (tool loop)"]
    classDef g fill:#E5F4FF,stroke:#006DDD,color:#03070F
    classDef f fill:#FDF3FF,stroke:#7E65AE,color:#3C2E5A
    classDef p fill:#F6FFDB,stroke:#6E8900,color:#2E3900
    class G1 g
    class F1 f
    class PR,SPECTRUM p
```


---

## Part 5 — Persistence: Checkpointers & Stores

This is the foundation everything durable rests on. The super‑step boundary (Part 2) is *where* state can be snapshotted; **persistence is the machinery that snapshots it** — and it's the single feature that unlocks memory, durable execution, human‑in‑the‑loop, *and* time‑travel.

### 5.1 Purpose

Persistence lets a LangGraph app keep useful information **beyond a single run** — to continue a conversation, resume after an interruption, recover from a failure, or remember across sessions. There are **two complementary systems**:

- **Checkpointers** persist a thread's **graph state** as checkpoints → **short‑term, thread‑scoped** memory. Powers conversation continuity, HITL, time‑travel, fault tolerance.
- **Stores** persist application‑defined key/value data **outside** the graph state → **long‑term, cross‑thread** memory. Powers user preferences, learned facts, shared knowledge.

Most apps use both. The mental model: a **thread** organizes a session's interactions (like an email thread); a **store namespace** organizes durable facts (like a user's folder).

### 5.2 Checkpointers — the heart of it

#### Building blocks

- **Thread** — a unique id; the **primary key** for storing/retrieving checkpoints. Passed as `config={"configurable": {"thread_id": "1"}}`. **Mandatory** with a checkpointer.
- **Checkpoint** — a snapshot of graph state at a super‑step boundary, surfaced as a **`StateSnapshot`** (`.values`, `.next` = next nodes to run [`()` = done], `.config` [carries `checkpoint_id`], `.metadata` [`source`/`writes`/`step`], `.parent_config` [links to the previous checkpoint — the chain that makes time‑travel work], `.tasks`).
- **`checkpoint_ns`** — namespace: `""` = root graph; `"node:uuid"` = a subgraph (nested joined by `|`).
- **Libraries:** `InMemorySaver` (dev), `SqliteSaver`/`AsyncSqliteSaver`, `PostgresSaver`/`AsyncPostgresSaver` (production; `from_conn_string` + `.setup()` once), plus community Mongo/Redis/Cosmos. All conform to `BaseCheckpointSaver`.
- **Inspect/edit:** `graph.get_state(config)` → latest (or specific) `StateSnapshot`; `graph.get_state_history(config)` → all snapshots (newest first); `graph.update_state(config, values, as_node=...)` → writes a *new* checkpoint (values pass through reducers).
- **Durability modes** (`graph.stream(..., durability=...)`): `"exit"` (persist only at end — fastest, no mid‑run crash recovery), `"async"` (persist while next step runs — balanced default), `"sync"` (persist before next step — most durable).

#### Annotated code — memory is one parameter

```python
from langgraph.checkpoint.memory import InMemorySaver
graph = builder.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": [{"role": "user", "content": "hi! I'm Bob"}]}, cfg)
graph.invoke({"messages": [{"role": "user", "content": "what's my name?"}]}, cfg)  # → "Bob"
```

Reusing the same `thread_id` is the *only* thing that makes the second turn remember "Bob." Production swaps the saver, same API:

```python
from langgraph.checkpoint.postgres import PostgresSaver
DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()   # run ONCE — creates tables / migrations
    graph = builder.compile(checkpointer=checkpointer)
```

A simple sequential run produces **one checkpoint per super‑step boundary** — inspect the history to see the chain:

```python
for snap in graph.get_state_history(cfg):     # newest first
    print(snap.next, snap.config["configurable"]["checkpoint_id"])
# (), <id3>          ← complete
# ('node_b',), <id2>
# ('node_a',), <id1>
# ('__start__',), <id0>   ← parent_config=None marks the first
```

Each snapshot's `config` carries a distinct `checkpoint_id` — feed any one back into `get_state`/`invoke` to time‑travel (Part 8).

### 5.3 Stores — long‑term, cross‑thread memory

#### Building blocks & code

A **store** holds arbitrary JSON documents under a **namespace** (a tuple, e.g. `(user_id, "memories")`) + **key**, reachable from *any* thread. `BaseStore` is the interface; `InMemoryStore`/`PostgresStore` the implementations. Core methods: `.put(namespace, key, value)`, `.get(namespace, key)`, `.search(namespace_prefix, query=, filter=, limit=)`, `.delete`, `.list_namespaces`.

```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(index={                       # index => semantic search
    "embed": init_embeddings("openai:text-embedding-3-small"),
    "dims": 1536,
    "fields": ["food_preference", "$"],
})
store.put(("user_123", "memories"), "1", {"food_preference": "I love pizza"})
items = store.search(("user_123", "memories"), query="What does the user like to eat?", limit=3)
```

Wire it into a graph (`compile(store=store)`) and read/write inside nodes via the injected `Runtime`:

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_id: str

async def call_model(state: MessagesState, runtime: Runtime[Context]):
    ns = (runtime.context.user_id, "memories")
    memories = await runtime.store.asearch(ns, query=state["messages"][-1].content, limit=3)
    info = "\n".join(d.value["data"] for d in memories)   # personalize the model call with recalled facts
    await runtime.store.aput(ns, str(uuid.uuid4()), {"data": "User prefers dark mode"})
```

Because the namespace is keyed by `user_id` (not `thread_id`), the **same user on a new thread still reaches the same memories** — that's the whole point of long‑term memory.

### 5.4 Managing short‑term memory — trim, delete, summarize

A growing message list overflows the context window and degrades quality, so you actively manage it (all inside a node, before the model call):

```python
# Trim (per-call, doesn't mutate state):
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
messages = trim_messages(state["messages"], strategy="last",
                         token_counter=count_tokens_approximately, max_tokens=128,
                         start_on="human", end_on=("human", "tool"))

# Delete (permanent — needs add_messages reducer):
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}   # or RemoveMessage(id=m.id) per message

# Summarize (condense then prune): one LLM call to summarize older turns, keep a `summary` state key,
# then RemoveMessage all but the last few. (LangMem's SummarizationNode automates this.)
```

> **Advanced — `DeltaChannel` for long threads.** With a normal reducer, the full `messages` list is re‑serialized into *every* checkpoint, so checkpoint size grows linearly with thread length. `DeltaChannel` (Part 2, beta ≥1.2) stores only each step's *delta*; set `snapshot_frequency=K` to bound read latency. Reach for it when you see checkpoint size scaling with conversation length.

> **Advanced — the long‑term memory taxonomy.** The docs map agent memory to cognitive science: **semantic** (facts about a user — stored as a continuously‑updated *profile* or a *collection* of documents), **episodic** (past experiences — few‑shot examples), **procedural** (instructions — the agent rewriting its own system prompt). And *when* to write: **hot path** (during the run — fresh but adds latency) vs **background** (a separate async task — no latency but you choose the trigger). Don't conflate **semantic memory** (storing facts) with **semantic search** (embedding‑based retrieval).

### 5.5 The overall picture — two persistence systems

```mermaid
flowchart LR
    subgraph ST["SHORT-TERM (thread-scoped)"]
        State["Graph State<br/>(messages + fields)"]
        CK["Checkpointer<br/>InMemory / Sqlite / Postgres"]
        TID["thread_id"]
        State <-->|"snapshot per super-step (StateSnapshot)"| CK
        TID --> CK
    end
    subgraph LT["LONG-TERM (cross-thread)"]
        Store["Store (BaseStore)<br/>namespace + key + JSON value"]
        Idx["index: embed + dims + fields<br/>→ semantic search"]
        Idx -.-> Store
    end
    Node["Graph node (Runtime[Context] / store: BaseStore)"]
    Node -->|"read/write state"| State
    Node -->|"put/get/search by user_id"| Store
    G["builder.compile(checkpointer=..., store=...)"] --> CK & Store
    CK --> POW["powers: memory · durable execution ·<br/>human-in-the-loop · time-travel"]
    classDef s fill:#FFF6DB,stroke:#B8860B,color:#3A2E00
    classDef n fill:#F2FAFF,stroke:#40668D,color:#2F4B68
    class State,CK,Store,Idx s
    class TID,Node,G,POW n
```

**One checkpointer → four superpowers.** That's the line to remember; Parts 6 and 8 cash it out.


---

## Part 6 — Durable Execution & Fault Tolerance

Real agents run for minutes, hours, or days, calling flaky external services. Part 6 is how LangGraph survives that: resume from where it left off after a crash, retry transient faults, bound runaway work, and recover gracefully — turning the Part‑1 error taxonomy into concrete primitives.

### 6.1 Purpose

**Durable execution** = run long workflows that **survive failures and resume from the last completed step** without re‑doing finished work. The enabler is the checkpointer (Part 5): state is snapshotted at every super‑step boundary, so on resume LangGraph skips completed steps and re‑runs only what remains.

> **Fault‑tolerance guarantee:** if nodes fail at a super‑step, restart from the last successful step. **Pending writes:** when one node fails mid‑super‑step, the successful *siblings'* writes are already persisted (at the node/task level), so they aren't re‑run on resume.

### 6.2 Durable execution — the determinism contract

Resume does **not** continue from the line that crashed. It returns to a **checkpoint boundary and replays forward**, restoring completed `@task`/subgraph results from the checkpoint instead of recomputing them, until it reaches the unfinished work. Two rules follow:

- **Encapsulate non‑determinism and side effects in tasks/nodes** — replay re‑runs the body, so a bare `time.time()`/`random()`/file‑write executes again. A `@task`'s result is checkpointed and restored.
- **Make them idempotent** — a task that *started but didn't finish* may run again on resume; use idempotency keys or check‑then‑write.

```python
# WRONG — file written twice on resume (entrypoint body replays):
@entrypoint(checkpointer=checkpointer)
def wf(inputs):
    with open("out.txt", "w") as f: f.write("done")   # ← re-executes on replay
    return interrupt("question")

# RIGHT — side effect in a task; its result is restored, not re-run:
@task
def write_file():
    with open("out.txt", "w") as f: f.write("done")

@entrypoint(checkpointer=checkpointer)
def wf(inputs):
    write_file().result()
    return interrupt("question")
```

Resume after a fixed error with `invoke(None, config)` (same `thread_id`): completed tasks load from the checkpoint, only the remaining work runs.

Two other durability knobs (Part 3/5 recap): **`recursion_limit`** caps super‑steps (default **1000** since v1.0.6 → `GraphRecursionError`; `RemainingSteps` lets you degrade gracefully before hitting it), and **node caching** (`CachePolicy(ttl=, key_func=)` + `compile(cache=InMemoryCache())`) skips recomputing expensive nodes.

### 6.3 Fault tolerance — three composable mechanisms

When a node attempt raises **any** exception (including a timeout): the **retry policy** decides whether to retry; **only after retries are exhausted** does the **error handler** run.

**1. Retries — `RetryPolicy`** (`langgraph.types`), passed to `add_node(retry_policy=...)`:

```python
from langgraph.types import RetryPolicy, default_retry_on
builder.add_node("call_api", call_api, retry_policy=RetryPolicy(max_attempts=3))
```

Params: `max_attempts` (default 3, **includes the first**), `initial_interval` (0.5s), `backoff_factor` (2.0), `max_interval` (128s), `jitter` (True), `retry_on`. The default **retries any exception EXCEPT** programmer errors (`ValueError`, `TypeError`, `RuntimeError`, `OSError`, etc.), and for `requests`/`httpx` **only retries 5xx**. Inspect attempts via `runtime.execution_info.node_attempt` (1‑indexed) to switch to a fallback on retry.

**2. Timeouts — `timeout=` / `TimeoutPolicy`** (≥1.2, **async‑only**):

```python
from langgraph.types import TimeoutPolicy
builder.add_node("call_model", call_model,
                 timeout=TimeoutPolicy(run_timeout=120, idle_timeout=30),  # hard cap + stall watchdog
                 retry_policy=RetryPolicy(max_attempts=3))
```

`run_timeout` is an absolute ceiling (never refreshed); `idle_timeout` fires only when the node stops making progress (resets on stream chunks, child tasks, LLM callbacks — or only on explicit `runtime.heartbeat()` with `refresh_on="heartbeat"`). On fire → **`NodeTimeoutError`** (retryable by default), the timed‑out attempt's writes are **cleared**, and the retry starts with a fresh clock.

**3. Error handler — `error_handler=` → Saga/compensation** (≥1.2): runs after retries are exhausted, receives a typed `NodeError(node, error)`, and can return a `Command` to update state and route to a recovery branch:

```python
from langgraph.errors import NodeError
from langgraph.types import Command, RetryPolicy

def payment_error_handler(state, error: NodeError) -> Command:
    return Command(update={"status": f"compensated: {error.error}"}, goto="finalize")

builder.add_node("charge_payment", charge_payment,
                 retry_policy=RetryPolicy(max_attempts=3, retry_on=ConnectionError),
                 error_handler=payment_error_handler)   # retry, then compensate instead of aborting
```

Plus **`set_node_defaults(...)`** for graph‑wide defaults, and **graceful shutdown**: a `RunControl().request_drain(reason)` (e.g. from a SIGTERM handler) stops *after the current super‑step*, saves a resumable checkpoint, and raises `GraphDrained`. (`interrupt()` bypasses both retries and error handlers — it's the "user‑fixable" channel, Part 8.)

### 6.4 The five‑way taxonomy, now with primitives

Part 1's table, fully wired:

```mermaid
flowchart TD
    fail([Node raises / stalls]) --> q1{Transient?<br/>network 5xx · NodeTimeoutError}
    q1 -->|yes| retry[[RetryPolicy<br/>backoff + jitter]]
    q1 -->|no| q2{LLM-recoverable?<br/>bad tool args}
    q2 -->|yes| loop[[loop back / fallback<br/>via node_attempt or Command goto]]
    q2 -->|no| q3{User-fixable?<br/>needs approval}
    q3 -->|yes| intr[[interrupt&#40;&#41; — durable pause<br/>bypasses retry + handler]]
    q3 -->|no| q4{Recoverable after<br/>retries exhausted?}
    q4 -->|yes| eh[[error_handler →<br/>Command compensate &#40;Saga&#41;]]
    q4 -->|no| bub([Unexpected → bubble up])
    retry -->|still failing| q4
    loop -->|still failing| q4
    classDef d fill:#FDF3FF,stroke:#7E65AE,color:#3C2E5A
    classDef m fill:#E5F4FF,stroke:#006DDD,color:#03070F
    classDef o fill:#F8E8E6,stroke:#B27D75,color:#634643
    class q1,q2,q3,q4 d
    class retry,loop,intr,eh m
    class bub,fail o
```

> **Advanced — the Saga pattern.** Model a multi‑step transaction (reserve → charge → ship) where each step has a compensating action. The `error_handler` of a failing step routes to compensation instead of aborting, keeping the system consistent across distributed side effects. This is exactly why retries and compensation are *decoupled* (configure when‑to‑retry and when‑to‑compensate independently). And it's why **smaller nodes** (Part 1) pay off here too: failure rewinds to the node boundary, so finer nodes redo less.

### 6.5 The overall picture — durable resume after a crash

```mermaid
sequenceDiagram
    participant App
    participant Graph
    participant CK as Checkpointer
    Note over App,CK: durability="sync", thread_id=T1
    App->>Graph: invoke(inputs, {thread_id:T1})
    Graph->>Graph: super-step 1 — node A runs
    Graph->>CK: write checkpoint (after A)
    Graph->>Graph: super-step 2 — node B (partial)
    Graph--xApp: 💥 crash mid super-step 2
    Note over CK: last full checkpoint = after A
    App->>Graph: invoke(None, {thread_id:T1})  ← resume
    Graph->>CK: load checkpoint after A
    Note over Graph: A skipped (restored); only B (+remaining) re-run
    Graph->>Graph: re-run node B (idempotent!) → checkpoint
    Graph-->>App: final result
```

Reliability is not bolted on — it's the natural payoff of BSP super‑steps (Part 2) plus the checkpointer (Part 5). Retries, timeouts, error handlers, and graceful drain are the dials on top.


---

## Part 7 — Streaming & Event Streaming

LLM latency is real; showing output as it's produced is a huge UX win. LangGraph streams at multiple granularities, all derived from the super‑step (Part 2). There are two layers: the lower‑level **stream‑mode API** and the higher‑level **event‑streaming API** built on top of it.

### 7.1 Purpose

Stream so users see progress before the run finishes: token‑by‑token text, per‑step "now running node X," and custom domain events ("fetched 50/100 records"). Because Pregel emits events **per super‑step**, streaming is just exposing those emissions at the granularity you choose.

### 7.2 The stream‑mode API (`stream`/`astream`)

#### Building blocks

`graph.stream(input, stream_mode=..., version="v2")` (v2 requires ≥1.1). Modes:

| Mode | Payload | Use |
|---|---|---|
| `"values"` | full state after each step | render whole‑graph state |
| `"updates"` | `{node: delta}` after each step | "node X just produced Y" |
| `"messages"` | `(token, metadata)` from LLM calls | token‑by‑token typing |
| `"custom"` | arbitrary data via `get_stream_writer()` | domain progress |
| `"checkpoints"` / `"tasks"` / `"debug"` | execution events (need a checkpointer) | deep debugging |

Pass a **list** to combine modes. With `version="v2"` every chunk is a uniform **`StreamPart`** dict (`type`/`ns`/`data`) — no tuple‑unpacking gymnastics. `subgraphs=True` surfaces nested‑graph emissions (`ns` identifies the source). `invoke(..., version="v2")` returns a **`GraphOutput`** with `.value` and `.interrupts`.

#### Annotated code — the four modes and a custom writer

```python
for part in graph.stream({"topic": "ice cream"},
                         stream_mode=["values", "updates", "messages", "custom"], version="v2"):
    if part["type"] == "values":    print("state:", part["data"])             # full snapshot
    elif part["type"] == "updates": print("node delta:", part["data"])        # {node: delta}
    elif part["type"] == "messages":
        msg, metadata = part["data"]; print(msg.content, end="")              # token + metadata
    elif part["type"] == "custom":  print("progress:", part["data"])          # arbitrary
```

```python
from langgraph.config import get_stream_writer

def generate_joke(state):
    writer = get_stream_writer()
    writer({"status": "thinking..."})        # emitted in stream_mode="custom"
    return {"joke": "..."}
```

Two useful controls: filter LLM tokens by node (`metadata["langgraph_node"] == "some_node"`) or suppress an LLM's tokens with the `nostream` tag (`model.with_config({"tags": ["nostream"]})`); and stream **any** non‑LangChain LLM by writing its raw chunks via `get_stream_writer()` in `custom` mode.

### 7.3 Event streaming v3 — typed projections

#### Purpose & building blocks

For most *application* code, the docs recommend **event streaming** (`stream_events(version="v3")`, ≥1.2): instead of branching on `stream_mode` chunks, you get a run object with **typed projections you can consume concurrently** — reading one doesn't starve the others:

| Projection | Use |
|---|---|
| `stream.messages` | chat messages + token deltas (`.text`, `.reasoning`, `.tool_calls`, `.node`) |
| `stream.values` | state snapshots / await final value |
| `stream.output` | await the final output |
| `stream.subgraphs` | observe nested runs by name (`.graph_name`, `.path`) — no namespace parsing |
| `stream.interrupts` / `stream.interrupted` | HITL payloads + paused flag (Part 8) |
| `stream.tool_calls` | tool lifecycle (via `ToolCallTransformer`) |
| `stream.extensions` | custom transformer projections |

Underneath is a two‑layer stack: Pregel emits raw events → an **event router** runs them through **stream transformers** → typed projections. Each transformer declares `required_stream_modes` (the runtime emits the union), and publishes a `StreamChannel` (named = serializable, flows into the main stream as `custom:<name>`; unnamed = side‑channel under `stream.extensions`).

#### Annotated code — the canonical idiom

```python
stream = graph.stream_events({"messages": [{"role": "user", "content": "What is 42 * 17?"}]},
                             version="v3")
for message in stream.messages:          # one per LLM call
    for token in message.text:           # live token deltas
        print(token, end="", flush=True)
final_state = stream.output              # drained final state
```

Concurrent multi‑projection (async): `await graph.astream_events(...)` then `asyncio.gather(consume_messages(), consume_subgraphs())` — they're independent streams.

> **Advanced — custom stream channels are a server‑side transformer.** A `StreamTransformer.process(event)` runs for *every* protocol event, can mutate it in place (e.g. scrub PII from `messages`/`values`) and `push(...)` structured side‑data on a named channel, returning `True` to keep the (now‑redacted) event. This is the mechanism a frontend reads via `useExtension`/`useChannel` (Part 10) — the same machinery that powers LangChain's `PIIMiddleware` output redaction.

### 7.4 The overall picture — the two streaming layers

```mermaid
flowchart TB
    PR["Pregel engine<br/>emits raw events per super-step"] --> ROUTER["Event router"]
    PR -. "raw stream-mode API" .-> SM["graph.stream(stream_mode=…, version='v2')<br/>values · updates · messages · custom · checkpoints · tasks · debug<br/>→ uniform StreamPart{type, ns, data}"]
    ROUTER --> TX["Stream transformers<br/>(built-in + custom; required_stream_modes)"]
    TX --> PROJ["Typed projections (stream_events v3)<br/>stream.messages · .values · .output ·<br/>.subgraphs · .interrupts · .tool_calls · .extensions"]
    SM -. "use for direct mode access" .-> APP1["app code"]
    PROJ -. "recommended for apps" .-> APP2["app code (concurrent consumers)"]
    classDef p fill:#F6FFDB,stroke:#6E8900,color:#2E3900
    classDef n fill:#F2FAFF,stroke:#40668D,color:#2F4B68
    class PR,ROUTER,TX p
    class SM,PROJ,APP1,APP2 n
```

Same graph execution underneath; pick the raw `stream_mode` layer for direct control, or the typed `v3` projections for clean application code.


---

## Part 8 — Human‑in‑the‑Loop & Time Travel

Two more payoffs of the checkpointer (Part 5), and the most distinctively "agentic" capabilities LangGraph offers: **pause indefinitely for a human**, and **rewind/branch through history**.

### 8.1 Human‑in‑the‑loop — `interrupt()` + `Command(resume=...)`

#### Purpose

When the model proposes a risky action — run SQL, send an email, spend money — you want a human to approve/edit/reject it *before* the side effect. HITL is "pause the graph, persist its state, wait indefinitely, then resume from the exact pause." It's the textbook payoff of durable execution, and a **native LangGraph primitive** (LangChain's `HumanInTheLoopMiddleware` is a thin wrapper over it).

#### Building blocks

- **`interrupt(payload)`** (`langgraph.types`) — pauses execution, surfaces any JSON‑serializable `payload` to the caller, and **returns the resume value** when continued.
- **`Command(resume=...)`** — re‑invoke the graph with this; the value becomes `interrupt()`'s return value inside the node.
- **Requirements:** a **checkpointer** (durable in prod), a **`thread_id`**, and `interrupt()` placed where you want to pause.
- **Surfacing:** `stream.interrupted` + `stream.interrupts` (event streaming v3); `result.interrupts` on `GraphOutput` (v2 invoke); `__interrupt__` key (v1). `Interrupt` has `.value` and `.id`.
- **Resume rules:** same `thread_id`; the **node re‑runs from its start** on resume (code before `interrupt` runs again); multiple interrupts match resume values **strictly by order/id**.

#### Annotated code — run to interrupt, then resume

```python
from langgraph.types import interrupt, Command

def approval_node(state) -> Command[Literal["proceed", "cancel"]]:
    is_approved = interrupt({"question": "Proceed?", "details": state["action_details"]})
    return Command(goto="proceed") if is_approved else Command(goto="cancel")

# 1) Run until the interrupt:
cfg = {"configurable": {"thread_id": "thread-1"}}
stream = graph.stream_events({"input": "data"}, config=cfg, version="v3")
_ = stream.output
if stream.interrupted:
    print(stream.interrupts)   # (Interrupt(value={'question': 'Proceed?', ...}),)

# 2) Resume with the human's decision (same thread_id) — resume value becomes interrupt()'s return:
graph.stream_events(Command(resume=True), config=cfg, version="v3").output
```

The interactive‑agent loop combines streaming + HITL: stream tokens, detect `stream.interrupted`, collect input, resume with `Command(resume=...)`, repeat until not interrupted. Approval logic can also live *inside a tool* (`interrupt(...)` in a `@tool`), making it reusable across graphs and able to override the tool's args on approve/edit.

> **Advanced — the "code before interrupt re‑runs" gotcha.** On resume the runtime restarts the **entire node from the beginning** (it raises a special exception the runtime catches). Therefore: (1) **never wrap `interrupt` in a bare `try/except`** — it'll swallow the resume exception; (2) **don't reorder or conditionally‑skip `interrupt` calls** in a node (matching is index‑based); (3) **make side effects before an `interrupt` idempotent** (or place them after it); (4) pass only **JSON‑serializable** payloads. Static `interrupt_before`/`interrupt_after` breakpoints exist but are for *debugging*, not HITL — use `interrupt()`.

### 8.2 Time travel — replay & fork

#### Purpose & building blocks

Every super‑step persists a checkpoint (Part 5), so you can **inspect** history, **replay** (re‑run from a prior checkpoint along the same path), and **fork** (branch from a prior checkpoint with modified state to explore an alternative). Both work by resuming from a `checkpoint_id`.

- **`get_state_history(config)`** → `StateSnapshot[]` (newest first); pick one by `.next`/`.metadata["step"]`.
- **Replay:** `graph.invoke(None, prior_snapshot.config)` — nodes *before* the checkpoint are restored (not re‑run); nodes *after* re‑execute (LLM calls, API calls, and **interrupts always re‑fire**).
- **Fork:** `graph.update_state(prior.config, values={...}, as_node=...)` returns a **new** branch `fork_config`; `graph.invoke(None, fork_config)` runs the new path. The original history stays intact.

#### Annotated code

```python
history = list(graph.get_state_history(config))            # newest first
before_joke = next(s for s in history if s.next == ("write_joke",))

# REPLAY — re-run from that checkpoint (same path):
graph.invoke(None, before_joke.config)

# FORK — change the state and branch (alternative path):
fork_config = graph.update_state(before_joke.config, values={"topic": "chickens"})
graph.invoke(None, fork_config)   # write_joke re-runs with the new topic; original timeline preserved
```

> **Advanced — subgraph time‑travel granularity.** With the default *inherited* checkpointer, the parent treats a whole subgraph as a single super‑step — you can only time‑travel at the parent level. Compile the subgraph with `checkpointer=True` to give it its own checkpoint history, enabling time‑travel *between* its internal steps (e.g. between two interrupts). Access nested state with `get_state(config, subgraphs=True)`.

### 8.3 The overall picture — interrupt/resume and replay/fork

```mermaid
flowchart TD
    subgraph HITL["Human-in-the-loop"]
        R1([invoke + thread_id]) --> I["node hits interrupt(payload)"]
        I --> P["checkpointer saves state · graph waits"]
        P --> S["stream.interrupts / GraphOutput.interrupts"]
        S --> H{human decides}
        H --> RES["invoke(Command(resume=value), SAME thread_id)"]
        RES --> CONT["node re-runs from start · interrupt() returns value · continue"]
    end
    subgraph TT["Time travel (same checkpoints)"]
        GH["get_state_history → pick checkpoint_id"] --> CH{replay or fork?}
        CH -->|"replay: invoke(None, snap.config)"| RP["re-run same path<br/>(nodes after re-execute)"]
        CH -->|"fork: update_state(snap.config, values=…)"| FK["new branch<br/>(original history intact)"]
    end
    CKP[(checkpointer)] -. enables both .- P
    CKP -. enables both .- GH
    classDef h fill:#E5F4FF,stroke:#006DDD,color:#03070F
    classDef t fill:#F6FFDB,stroke:#6E8900,color:#2E3900
    classDef n fill:#F2FAFF,stroke:#40668D,color:#2F4B68
    class R1,I,P,S,H,RES,CONT h
    class GH,CH,RP,FK t
    class CKP n
```

Notice both halves hang off the same `checkpointer` — pausing for a human and rewinding through history are the *same* mechanism (durable checkpoints) used two ways.


---

## Part 9 — Subgraphs, Composition, App Structure & Testing

Real systems are built from parts. A **subgraph** is *a graph used as a node in another graph* — the unit of composition, reuse, and multi‑agent structure.

### 9.1 Subgraphs — purpose & the two communication patterns

A subgraph lets you build multi‑agent systems, reuse a set of nodes, and distribute development: as long as the **interface (input/output schemas)** holds, the parent can be built without knowing the subgraph's internals. There are exactly **two ways** to connect them — the key decision:

| Pattern | When | How |
|---|---|---|
| **Add a subgraph as a node** | parent & subgraph **share state keys** | pass the compiled subgraph straight to `add_node` — no wrapper |
| **Call a subgraph inside a node** | **different** schemas / need to transform | write a wrapper node that maps parent→sub input and sub→parent output |

```python
# (A) Shared keys — add as a node:
builder.add_node("node_1", subgraph)          # subgraph reads/writes the parent's channels

# (B) Different schemas — call inside a wrapper node:
def call_subgraph(state: State):
    out = subgraph.invoke({"bar": state["foo"]})   # map parent → subgraph
    return {"foo": out["bar"]}                       # map subgraph → parent
builder.add_node("node_1", call_subgraph)
```

**Subgraph persistence** is controlled by the subgraph's `.compile(checkpointer=...)`:

| Mode | `checkpointer=` | Behavior |
|---|---|---|
| **Per‑invocation** (default) | `None` | fresh each call, but inherits the parent's checkpointer for interrupts & durability; parallel‑safe |
| **Per‑thread** | `True` | accumulates state across calls (multi‑turn subagent memory); **no parallel calls** |
| **Stateless** | `False` | no checkpointing — runs like a plain function |

The **parent must be compiled with a checkpointer** for any of this to work. Subgraph runs are namespaced `node_name:<uuid>` (`checkpoint_ns`); inspect with `get_state(config, subgraphs=True)` and stream with the `stream.subgraphs` projection (Part 7) — but only when the subgraph is **statically discoverable** (added as a node or called in a node), *not* when hidden inside a tool function. **Interrupts always propagate to the top‑level graph regardless of nesting.**

### 9.2 Two perspectives: subgraph ↔ parent graph

#### 👁️ From the subgraph's perspective ("I'm a self‑contained graph")

You're a normal compiled `StateGraph` with your own schema, nodes, and edges. You don't know you're embedded — you just `invoke` on your own state. You decide your persistence story (`checkpointer=None|True|False`). If your schema differs from your caller's, you trust them to translate at the boundary; if you share keys, you read/write the parent's channels directly. You can interrupt, and the pause will surface all the way up without your involvement.

#### 👁️ From the parent's perspective ("I'm orchestrating")

You see the subgraph as **one node**. If you share keys, you `add_node(compiled_subgraph)` and it participates in your state like any node; if not, you wrap it in a node that transforms state in and out. You own the checkpointer (the subgraph inherits it per‑invocation), and you assign the subgraph a **namespace** (`node_name:<uuid>`) so multiple invocations don't collide — which is why, for *per‑thread* subagents, you wrap each in its own `StateGraph` with a unique node name for a stable namespace. You observe its nested execution via `get_state(..., subgraphs=True)` and `stream.subgraphs`. From here, a subgraph is just a node that happens to be a whole graph — composition all the way down.

> **Advanced — `Command(graph=Command.PARENT)`.** A node *inside* a subgraph can return `Command(update=..., goto="parent_node", graph=Command.PARENT)` to navigate the **parent** graph — the basis of multi‑agent handoffs. If the update touches a key shared by both schemas, the parent must define a **reducer** for it.

### 9.3 App structure & testing

#### Building blocks

A deployable LangGraph app = **graphs** + a **`langgraph.json`** config + a **dependencies** file (`requirements.txt`/`pyproject.toml`) + optional **`.env`**. The config:

```json
{
  "dependencies": ["."],
  "graphs": { "agent": "./src/agent.py:agent" },
  "env": ".env"
}
```

`graphs` maps a **name → `path:variable`** (a compiled graph or a factory). That **one name** is the shared identifier across Studio, deploy, the SDK, and the frontend (Part 10). `create_agent` returns a compiled graph, exactly what `graphs` expects.

**Testing** uses `pytest`: a `create_graph()` factory, compiled with a **fresh `MemorySaver`** per test, then `invoke` + assert on returned state. Test a node in isolation via `compiled_graph.nodes["node1"].invoke({...})` (bypasses the checkpointer). Test a *slice* with `update_state(..., as_node=...)` to simulate prior state + `invoke(None, ..., interrupt_after="node3")` to stop early.

```python
def test_basic_execution():
    graph = create_graph().compile(checkpointer=MemorySaver())
    result = graph.invoke({"my_key": "initial"}, {"configurable": {"thread_id": "1"}})
    assert result["my_key"] == "hello from node2"
```

### 9.4 The overall picture — composition

```mermaid
flowchart TB
    subgraph Parent["Parent graph (owns the checkpointer)"]
        P1["node_1"]
        WN["node_2 = wrapper<br/>call_subgraph(): map in/out<br/>(different schemas)"]
        SN["node_3 = compiled subgraph<br/>(shared keys, no wrapper)"]
        P1 --> WN --> SN
    end
    subgraph SubA["Subgraph A (own schema)"]
        SA1["node"] --> SA2["node"]
    end
    subgraph SubB["Subgraph B (shares 'foo')"]
        SB1["node"] --> SB2["node"]
    end
    WN -- "invoke({bar: foo})" --> SubA
    SubA -- "return {foo: bar}" --> WN
    SN -. "reads/writes parent channels<br/>ns = node_3:&lt;uuid&gt;" .- SubB
    NOTE["persistence: None=per-invocation · True=per-thread · False=stateless<br/>interrupts always propagate to top-level"]
    classDef p fill:#E5F4FF,stroke:#006DDD,color:#03070F
    classDef s fill:#F6FFDB,stroke:#6E8900,color:#2E3900
    class P1,WN,SN p
    class SA1,SA2,SB1,SB2 s
```


---

## Part 10 — Operate & Deliver: Studio, Deploy, Observability & Frontend

A graph that runs on your laptop isn't a product. Part 10 is the path from local debugging to production hosting to a reactive UI — and it's dense with cross‑framework seams.

### 10.1 The shared contract: `langgraph.json`

One name ties the whole delivery pipeline together. The `graphs` key (`"agent": "./src/agent.py:agent"`) defines a graph id reused by **five consumers**: Studio's graph, the deploy `assistant_id`, the SDK's `client.runs.stream("agent", ...)`, the REST `assistant_id`, and the frontend's `useStream({ assistantId: "agent" })`. **One name, five consumers** — because all of them operate on the same compiled Pregel graph.

### 10.2 Studio — local visual debugging

`langgraph dev` (from `pip install --upgrade "langgraph-cli[inmem]"`, Python ≥3.11) serves your graph as a local **Agent Server** at `http://127.0.0.1:2024`, and **LangGraph Studio** is a visual UI over it: see each step's prompts/tool calls/results, test inputs, inspect intermediate state, **hot‑reload** prompts/tools, and re‑run threads from any step. `LANGSMITH_TRACING=false` keeps all data local. (Safari blocks localhost → `langgraph dev --tunnel`.)

### 10.3 Deploy — the LangGraph Platform

Traditional hosts assume stateless, short requests; LangGraph apps are **stateful and long‑running**, needing persistent state and background execution. The **LangGraph Platform** (LangSmith Deployment) is purpose‑built for that: push a GitHub repo, it builds (~15 min) and gives an API URL, with a **checkpointer auto‑provisioned** (memory/HITL "just work" in prod). Hosting options: cloud, self‑hosted, hybrid. Consume via SDK or REST — same `stream_mode` vocabulary as local:

```python
from langgraph_sdk import get_sync_client
client = get_sync_client(url="your-deployment-url", api_key="your-langsmith-api-key")
for chunk in client.runs.stream(None, "agent",          # graph name from langgraph.json
                                input={"messages": [{"role": "human", "content": "Hi"}]},
                                stream_mode="updates"):
    print(chunk.event, chunk.data)
```

You consume a deployed graph through the SDK, **not** by importing the graph object — the platform speaks the protocol over HTTP.

### 10.4 Observability — and the LangGraph ↔ LangSmith perspectives

Enable tracing with two env vars (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`); every run becomes a **trace** (a tree of step "runs") with no extra code. Scope it with `ls.tracing_context(...)`, route to a project with `LANGSMITH_PROJECT`, attach `config={"tags": [...], "metadata": {...}}`, and mask data with anonymizers.

#### 👁️ From LangGraph's perspective ("I'm a running graph")

You do nothing special. You run your super‑steps; because you're auto‑instrumented, each node/model/tool execution **emits a span**. The `RunnableConfig` you already thread for `tags`/`metadata` flows into those spans. You don't know what a "trace" is — you just execute and emit.

#### 👁️ From LangSmith's perspective ("I'm the observability platform")

You receive spans and assemble a **trace tree** — root run (the invocation) with children per node, model call, and tool call, each carrying inputs/outputs/latency/tokens. You store **datasets** and run **experiments** to score behavior (evals; same role as in the LangChain guide). You don't care whether the thing emitting spans is a raw `StateGraph`, a `create_agent`, or a Deep Agent — they all emit the same shape, so you observe and evaluate them uniformly. You're also where Studio reads its traces and where LangSmith Engine monitors production runs and proposes fixes.

### 10.5 Frontend — and the server ↔ client perspectives

The **frontend SDK** (`useStream` in React/Vue/Svelte, `injectStream` in Angular) turns a deployed graph into a UI that **mirrors the graph's structure** — nodes, state keys, checkpoints, interrupts, subgraphs, and streamed messages are all visible runtime concepts, so the UI explains what the system is doing instead of hiding it behind one message.

```tsx
import { useStream } from "@langchain/react";
function Pipeline() {
  const stream = useStream<typeof graph>({ apiUrl: "http://localhost:2024", assistantId: "pipeline" });
  const graphNodes = [...stream.subgraphs.values()];   // nodes discovered as they run (SubgraphDiscoverySnapshot)
  // each node: node.nodeName, node.status ("pending"|"running"|"complete"|"error")
  const synthesis = stream.values?.synthesis;          // whole-graph state by key
}
```

The server can also stream **custom channels** (Part 7): a server‑side `StreamTransformer` publishes structured data on a named channel; the client reads it with `useExtension(stream, "name")` (latest typed payload) or `useChannel(stream, ["custom:name"])` (raw buffer).

#### 👁️ From the server's perspective ("I'm the LangGraph Agent Server")

You expose a `/threads` + `/runs` streaming HTTP API on `:2024` (local) or the deployment URL. A client `submit` starts a run on the named graph (`assistantId`); as super‑steps execute you **stream events** (per‑node updates, message tokens, tool lifecycle, state values) and **persist a checkpoint each step**. On `interrupt()` you pause durably and emit the interrupt. You don't know it's React vs Angular — you speak the protocol; resume (`command.resume`), history (`getHistory`), and forking (`forkFrom`) are just more requests against the persisted thread. Your statefulness is exactly why a client can disconnect and reattach.

#### 👁️ From the client's perspective ("I'm `useStream`")

You hold a `threadId` and `submit`. You receive the server's streamed events and **assemble them into reactive state**: tokens into `stream.messages`, per‑node status into `stream.subgraphs` (discovered, not hardcoded), state into `stream.values`, pauses into `stream.interrupt`. You don't *run* the graph — you *render and steer* it: approve a pause with `submit(null, {command:{resume}})`, time‑travel by reading history and `submit({}, {forkFrom})`, leave a long run with `disconnect()` and reattach later with the same `threadId`. Every "control plane" power you have is a typed request against the server's durable thread — the checkpointer on the backend is what makes your UI feel stateful. (The Agent Chat UI is a ready‑made Next.js client of exactly this protocol.)

### 10.6 The overall picture — delivery topology

```mermaid
flowchart TD
    subgraph SRC["Source (GitHub repo)"]
        AG["agent.py: a compiled graph (or create_agent)"]
        CFG["langgraph.json (graphs: agent → ./src/agent.py:agent)"]
    end
    subgraph LOCAL["Local dev"]
        CLI["langgraph dev (Agent Server :2024)"] --> STU["LangGraph Studio"]
    end
    subgraph CLOUD["LangGraph Platform"]
        DEP["managed runtime: cloud / self-hosted / hybrid<br/>auto-provisioned checkpointer"]
    end
    subgraph CONS["Consumers (Graph ID + URL)"]
        SDK["langgraph-sdk: client.runs.stream('agent', …)"]
        FE["useStream({apiUrl, assistantId:'agent'})"]
        ACUI["Agent Chat UI"]
    end
    CFG --> CLI
    SRC -->|"git push, ~15 min"| DEP
    DEP --> SDK & FE & ACUI
    CLI -. ":2024" .-> FE
    CLI -. ":2024" .-> ACUI
    STU -. same traces .-> OBS["LangSmith (LANGSMITH_TRACING)"]
    DEP -. prod traces .-> OBS
    classDef s fill:#F2FAFF,stroke:#40668D,color:#2F4B68
    classDef d fill:#EBD0F0,stroke:#885270,color:#441E33
    class AG,CFG,CLI,STU s
    class DEP,SDK,FE,ACUI,OBS d
```

The same compiled graph, named once in `langgraph.json`, flows from your editor to Studio to production to a reactive UI — without changing the graph code.


---

## Part 11 — Building Agents on LangGraph

We've built the runtime, both APIs, persistence, durability, streaming, HITL, and composition. Now we use them to build real agents *at the graph level* — and confront the central relationship head‑on: **`create_agent` is a prebuilt graph; LangGraph is what you drop to when you need more control than a prompt can buy.**

### 11.1 Two perspectives: LangGraph ↔ LangChain (`create_agent`)

This is the inverse of the companion guide's headline seam, so it deserves both sides.

#### 👁️ From LangChain's perspective ("I'm `create_agent`")

You offer a *batteries‑included* tool‑calling loop: pass a model, tools, and a prompt; get an agent with `.invoke`/`.stream`. You hide the graph — the model‑node/tool‑node cycle, the conditional edge, the state schema — behind a clean factory. You configure behavior through **middleware and the system prompt**. You're the right tool 90% of the time, and you *are* a LangGraph graph, so you inherit durability, persistence, streaming, and HITL for free.

#### 👁️ From LangGraph's perspective ("I'm the runtime; `create_agent` is one graph I produce")

`create_agent` compiles down to *me* — a `StateGraph` over `MessagesState` with a model node, a `ToolNode`, and a conditional edge. When that prebuilt loop isn't enough — you need explicit nodes for distinct steps, conditional edges that *enforce* an order, self‑correction loops the prompt can only *request*, or a topology that isn't "loop until done" — you stop using the prebuilt and **author the graph directly**. The docs say exactly this: *"LangChain offers built‑in agent implementations, implemented using LangGraph primitives. If deeper customization is required, agents can be implemented directly in LangGraph."* The difference is **where control lives**: in `create_agent`, in a system prompt you *hope* the LLM obeys; in raw LangGraph, in the **graph topology** that the runtime *guarantees*.

The next two examples are the same RAG and SQL agents from the LangChain guide — rebuilt at the graph level — so you can see precisely what dropping down buys you.

### 11.2 Graph‑level Agentic RAG — retrieve → grade → rewrite/generate

The prebuilt agentic‑RAG (LangChain guide, Part 6) is "give the agent a retrieval tool and let it decide." The **graph version adds explicit quality control**: grade the retrieved docs, rewrite the query if they're irrelevant, and only then answer — a self‑correction loop the prompt can't guarantee.

```python
from typing import Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

# Node 1: decide to retrieve or answer directly
def generate_query_or_respond(state: MessagesState):
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}

# Node 2 (used AS a conditional edge): grade relevance with structured output
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if relevant, else 'no'")

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content
    context  = state["messages"][-1].content          # the ToolMessage from retrieval
    score = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": GRADE_PROMPT.format(question=question, context=context)}]
    ).binary_score
    return "generate_answer" if score == "yes" else "rewrite_question"

# Nodes 3 & 4: rewrite the query, or generate the final answer (omitted for brevity)

workflow = StateGraph(MessagesState)
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)
workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges("generate_query_or_respond", route_on_tool_calls,  # retrieve or END
                               {"tools": "retrieve", END: END})
workflow.add_conditional_edges("retrieve", grade_documents)   # grade → generate or rewrite
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")   # self-correction loop
graph = workflow.compile()
```

The teaching points: `grade_documents` is a **structured‑output grader used *as* a conditional edge** — its `Literal[...]` return *is* the routing, so LangGraph needs no path map. The cycle `retrieve → rewrite_question → generate_query_or_respond` is **explicit self‑correction** baked into the topology. `route_on_tool_calls` is a hand‑rolled equivalent of the prebuilt `tools_condition`. Control is enforced by edges, not hoped for in a prompt.

### 11.3 Graph‑level SQL Agent — dedicated nodes enforce the workflow

The prebuilt SQL agent (LangChain guide, Part 6) relied on the *system prompt* to say "always list tables first" and "always check the query before running." The **graph version enforces it with dedicated nodes and edges** — *"a simple ReAct‑agent setup, with dedicated nodes for specific tool‑calls."*

```python
# A PREDETERMINED first step — no LLM discretion: always list tables.
def list_tables(state: MessagesState):
    tool_call = {"name": "sql_db_list_tables", "args": {}, "id": "abc123", "type": "tool_call"}
    msg = list_tables_tool.invoke(tool_call)
    return {"messages": [AIMessage(content="", tool_calls=[tool_call]), msg, AIMessage(f"Tables: {msg.content}")]}

# FORCE a schema-fetch tool call with tool_choice="any":
def call_get_schema(state: MessagesState):
    return {"messages": [model.bind_tools([get_schema_tool], tool_choice="any").invoke(state["messages"])]}

# Generate SQL — do NOT force a tool call, so the model can answer once it has the result:
def generate_query(state: MessagesState):
    sys = {"role": "system", "content": generate_query_system_prompt}   # "no DML", top_k, etc.
    return {"messages": [model.bind_tools([run_query_tool]).invoke([sys] + state["messages"])]}

def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    return "check_query" if state["messages"][-1].tool_calls else END

builder = StateGraph(MessagesState)
builder.add_node(list_tables); builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema"); builder.add_node(generate_query)
builder.add_node(check_query); builder.add_node(run_query_node, "run_query")
builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges("generate_query", should_continue)   # tool call → check_query, else END
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")                    # self-correction loop
agent = builder.compile()
```

The graph **guarantees** the order `list_tables → get_schema → generate_query`, **forces** tool calls where needed (`tool_choice="any"`), and loops `run_query → generate_query` for self‑correction (a bad query returns an error string the model sees and fixes). HITL drops in by wrapping `run_query` with `interrupt(...)` (Part 8) + a checkpointer. This is control the prebuilt agent can only *suggest*.

### 11.4 Who builds on LangGraph — case studies

LangGraph runs in production across finance, healthcare, legal, logistics, telecom, government, and dev tooling — Klarna, Uber, J.P. Morgan, BlackRock, Morningstar, LinkedIn, Replit, GitLab, Qodo, Cisco, Vodafone, Rakuten, and dozens more. The dominant patterns map directly onto what this guide taught: **copilots for domain tasks**, **code generation / DevOps agents** (Uber, Replit, GitLab, Qodo, Cisco — tool‑using ReAct graphs), **multi‑agent customer support** (Minimal — subgraphs + LangSmith), **text‑to‑SQL for analytics** (LinkedIn — exactly Part 11.3), and **research/summarization**. LangSmith recurs as the companion for tracing and evaluation. The lesson: the low‑level control you've been learning is what these teams reached for when reliability mattered at scale.

### 11.5 The overall picture — prebuilt vs. graph‑level

```mermaid
flowchart TB
    subgraph PREBUILT["LangChain create_agent (prebuilt graph)"]
        CA["model + tools + prompt + middleware<br/>control lives in the SYSTEM PROMPT<br/>'loop until done'"]
    end
    CA == "compiles to" ==> LG
    subgraph LG["Raw LangGraph (author the graph)"]
        direction TB
        N["explicit nodes (retrieve · grade · rewrite · generate /<br/>list_tables · get_schema · generate · check · run)"]
        E["conditional edges ENFORCE order +<br/>self-correction loops"]
        C["control lives in the GRAPH TOPOLOGY<br/>(guaranteed by the runtime)"]
        N --> E --> C
    end
    DROP["Drop down when you need: explicit steps · enforced order ·<br/>self-correction · non-'loop-until-done' topology · deterministic + agentic mix"]
    DROP -.-> LG
    classDef pb fill:#FDF3FF,stroke:#7E65AE,color:#3C2E5A
    classDef lg fill:#F6FFDB,stroke:#6E8900,color:#2E3900
    class CA pb
    class N,E,C lg
```

Same agents, two altitudes: the prebuilt for speed, the graph for guaranteed control. Knowing both — and that the first *is* the second underneath — is the whole point of learning LangGraph.


---

## Part 12 — The Whole Picture

We descended from "why control?" all the way to the streaming protocol between a server and a browser. Now zoom back out, see how it all connects, understand *why* it evolved this way, and leave with decision guides.

### 12.1 The full lifecycle — author → run → persist → operate

```mermaid
flowchart LR
    subgraph AUTH["✍️ AUTHOR"]
        direction TB
        GA["Graph API<br/>StateGraph (Part 3)"]
        FA["Functional API<br/>@entrypoint (Part 4)"]
    end
    subgraph RUN["⚙️ RUN"]
        PR["Pregel runtime<br/>actors + channels · BSP super-steps<br/>(Part 2)"]
    end
    subgraph FOUND["💾 FOUNDATION"]
        CK["Checkpointer + Store<br/>(Part 5)"]
    end
    subgraph POWERS["🎁 RUNTIME POWERS"]
        DUR["Durable execution +<br/>fault tolerance (Part 6)"]
        STR["Streaming + events v3<br/>(Part 7)"]
        HT["HITL + time-travel<br/>(Part 8)"]
        SUB["Subgraphs (Part 9)"]
    end
    subgraph SHIP["🚀 OPERATE & DELIVER"]
        STU["Studio · Deploy (Part 10)"]
        OBS["LangSmith (Part 10)"]
        FE["Frontend useStream (Part 10)"]
    end
    subgraph ON["🏗️ BUILT ON TOP"]
        AGT["create_agent · Deep Agents ·<br/>graph-level RAG/SQL (Part 11)"]
    end
    GA == compile ==> PR
    FA == @entrypoint ==> PR
    PR --- CK
    CK --> DUR & HT
    PR --> STR & SUB
    PR -- hosted by --> STU
    PR -- auto-traced --> OBS
    STU -- streams to --> FE
    AGT -. "ARE graphs" .-> PR
    classDef a fill:#E5F4FF,stroke:#006DDD,color:#03070F
    classDef r fill:#F6FFDB,stroke:#6E8900,color:#2E3900
    classDef f fill:#FFF6DB,stroke:#B8860B,color:#3A2E00
    classDef p fill:#F2FAFF,stroke:#40668D,color:#2F4B68
    classDef s fill:#EBD0F0,stroke:#885270,color:#441E33
    classDef o fill:#FDF3FF,stroke:#7E65AE,color:#3C2E5A
    class GA,FA a
    class PR r
    class CK f
    class DUR,STR,HT,SUB p
    class STU,OBS,FE s
    class AGT o
```

### 12.2 Purpose recap — every layer in one table

| Layer | Purpose (the *why*) | Achieved by | Part |
|---|---|---|---|
| **Positioning** | Low‑level control for reliable, stateful, long‑running agents | a runtime, not an abstraction | 0 |
| **Mental model** | Decompose into nodes → transitions → shared state | the 5‑step method; errors as flow | 1 |
| **Pregel** | One execution engine for both APIs | actors + channels + BSP super‑steps | 2 |
| **Graph API** | Declarative, visual, explicit control | `StateGraph`, reducers, `Command`, `Send` | 3 |
| **Functional API** | LangGraph powers with minimal code change | `@entrypoint`, `@task`, `previous` | 4 |
| **Persistence** | Keep state beyond one run | checkpointer (thread) + store (cross‑thread) | 5 |
| **Durable execution** | Survive crashes; recover from faults | replay + `RetryPolicy`/`error_handler`/timeouts | 6 |
| **Streaming** | Show progress as it happens | `stream_mode` + `stream_events(v3)` projections | 7 |
| **HITL + time‑travel** | Human oversight; replay/fork history | `interrupt`/`Command`; `get_state_history`/`update_state` | 8 |
| **Subgraphs** | Compose, reuse, multi‑agent | graph‑as‑a‑node; `langgraph.json`; testing | 9 |
| **Operate & deliver** | Debug, host, observe, surface | Studio · Platform · LangSmith · `useStream` | 10 |
| **Building agents** | Control beyond a prebuilt loop | graph‑level RAG/SQL; `create_agent` is a graph | 11 |

### 12.3 Why it's built this way — the evolution

The architecture is the residue of hard‑won lessons. The version arc explains the design:

```mermaid
timeline
    title LangGraph (+ ecosystem) evolution
    Oct 2025 : v1.0.0 — langchain + langgraph unify at 1.0 (v0.x archived)
    Nov 2025 : langchain 1.1 — model profiles, middleware
    Dec 2025 : langchain 1.2 — tool extras, strict structured output
    Mar 2026 : langgraph 1.1 — TYPE the I/O (streaming/invoke v2, GraphOutput, Pydantic coercion, time-travel fixes)
    Apr 2026 : deepagents 0.5 — async subagents, multimodal
    May 2026 : langgraph 1.2 — HARDEN execution (node timeouts + error handlers, DeltaChannel, graceful shutdown, event streaming v3)
    May 2026 : langchain 1.3 + deepagents 0.6 — event streaming v3 adopted; deepagents adopts DeltaChannel
```

The LangGraph‑only arc reads as a clear three‑act story:
1. **v1.0 — stabilize.** Unify with LangChain at 1.0; one versioning baseline.
2. **v1.1 — type the boundaries.** Developers were tired of untyped dict chunks and dict outputs, so streaming/invoke got opt‑in `version="v2"` with `StreamPart` and `GraphOutput`, plus Pydantic coercion and time‑travel correctness fixes.
3. **v1.2 — harden execution for scale.** Long‑running production graphs needed: `DeltaChannel` (checkpoints stop ballooning on long threads), per‑node `timeout=`, node `error_handler=` (Saga/compensation), graceful shutdown, and content‑block‑centric event streaming v3.

Tellingly, `DeltaChannel` was introduced in `langgraph` 1.2 and *immediately adopted* by `deepagents` 0.6 — proof that runtime improvements flow **up** to the higher‑level frameworks built on LangGraph. That direction of flow is the whole thesis of this guide: **LangGraph is the foundation; everything else stands on it.**

### 12.4 Decision guide — what to reach for, when

| If you need… | Reach for | Part |
|---|---|---|
| A simple tool‑calling agent | LangChain `create_agent` (a prebuilt graph) | 11 |
| Explicit steps / enforced order / self‑correction | author a `StateGraph` directly | 3, 11 |
| LangGraph powers in existing procedural code | the Functional API (`@entrypoint`/`@task`) | 4 |
| Conversation memory | a checkpointer + stable `thread_id` | 5 |
| Cross‑session memory | a `store` + namespaces (+ embeddings for semantic) | 5 |
| Parallel work that merges | fan‑out edges / `Send` + a reducer channel | 2, 3 |
| Dynamic, runtime‑decided fan‑out (map‑reduce) | `Send` + `operator.add` | 3, 4 |
| Survive crashes / retry / compensate | durability mode + `RetryPolicy`/`timeout`/`error_handler` | 6 |
| Human approval before risky actions | `interrupt()` + `Command(resume=...)` + checkpointer | 8 |
| Debug by rewinding / explore alternatives | time‑travel (`get_state_history` + replay/fork) | 8 |
| Progress UX (tokens, steps, custom events) | `stream_mode` or `stream_events(v3)` projections | 7 |
| Compose / reuse / multi‑agent | subgraphs (as a node or in a node) | 9 |
| Local visual debugging | `langgraph dev` + Studio | 10 |
| Production hosting (stateful, long‑running) | LangGraph Platform (deploy from repo) | 10 |
| A UI that mirrors the graph | `useStream` + custom channels | 10 |
| To know *why* it did that / catch regressions | LangSmith tracing + evals | 10 |

### 12.5 Reference — packages

- **`langgraph`** — the runtime: `StateGraph`, `Pregel`, channels, `langgraph.func` (`@entrypoint`/`@task`), `langgraph.types` (`Command`, `Send`, `interrupt`, `RetryPolicy`, `TimeoutPolicy`, `GraphOutput`), `langgraph.prebuilt` (`ToolNode`, `tools_condition`), `langgraph.constants` (`START`/`END`), `langgraph.errors` (`GraphRecursionError`, `NodeError`, `NodeTimeoutError`, `GraphDrained`).
- **`langgraph-checkpoint`** (+ `-sqlite`, `-postgres`) — checkpointers & stores (`InMemorySaver`, `PostgresSaver`, `InMemoryStore`, `PostgresStore`, `BaseStore`, `IndexConfig`).
- **`langgraph-cli[inmem]`** — `langgraph dev` (Studio/local Agent Server). **`langgraph-sdk`** — deployment client. **`@langchain/{react,vue,svelte,angular}`** — frontend `useStream`/`injectStream`.
- **`langchain`** — models/tools/messages and `create_agent` (the prebuilt graph). **`langsmith`** — tracing + evals. **`deepagents`** — the batteries‑included harness, built on LangGraph.

```bash
pip install -U langgraph                       # the runtime
pip install -U "langgraph-cli[inmem]" langgraph-sdk   # dev server + deploy client
pip install -U langchain langsmith             # framework + observability
pip install -U langgraph-checkpoint-postgres   # production persistence
```

---

### Closing thought

The companion guide's thesis was *Agent = Model + Harness*, and it called LangGraph "the runtime underneath." From down here the picture inverts into a single sentence: **the harness is borrowed control; LangGraph is where you build control you can trust.** A model gives you intelligence; the Pregel runtime — super‑steps you can checkpoint, channels you can merge, graphs you can pause, replay, and compose — gives you the *reliability* that turns that intelligence into a production agent. Everything else in the ecosystem, `create_agent` and Deep Agents included, is a convenience built on that one foundation.

*— End of guide.*


