# Build a data analysis agent from scratch

> Build a data analysis agent step by step using create_agent and deepagents middleware.

This guide builds a data analysis agent from first principles using `create_agent` and deepagents middleware. Rather than starting with `create_deep_agent`, we assemble the harness one piece at a time: so you can see exactly what each component adds and swap in only what your use case needs.

The agent we'll build:

1. Accepts a CSV file for analysis
2. Writes and executes Python code in an isolated sandbox
3. Delegates visualization work to a specialized subagent
4. Loads data analysis patterns from a skills file

## Setup

```bash theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
pip install deepagents langsmith
```

Enable LangSmith tracing to inspect every step:

```bash theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
```

***

## Step 1: The minimal agent

A model, a loop. Nothing else yet.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent

agent = create_agent("anthropic:claude-sonnet-4-6", tools=[])
```

This runs, but the agent has no filesystem and no way to execute code. The next steps add those.

***

## Step 2: Add a sandbox backend

`LangSmithSandbox` gives the agent an isolated environment with a filesystem and an `execute` tool for running shell commands. The agent can install packages, write scripts, and run them: without touching the host.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents import create_agent
from langsmith.sandbox import SandboxClient
from deepagents.backends.langsmith import LangSmithSandbox
from deepagents.middleware import FilesystemMiddleware

client = SandboxClient()
sandbox = client.create_sandbox(template_name="deepagents-deploy")
backend = LangSmithSandbox(sandbox=sandbox)

agent = create_agent(
    "anthropic:claude-sonnet-4-6",
    tools=[],
    middleware=[FilesystemMiddleware(backend=backend)],
)
```

[`FilesystemMiddleware`](https://reference.langchain.com/python/deepagents/middleware/filesystem/FilesystemMiddleware) adds `read_file`, `write_file`, `edit_file`, `glob`, and `grep`. Because `LangSmithSandbox` implements the sandbox protocol, it also adds `execute`: the agent can now run shell commands.

Upload a CSV and invoke:

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import csv, io

rows = [
    ["Date", "Product", "Units", "Revenue"],
    ["2025-08-01", "Widget A", 10, 250],
    ["2025-08-02", "Widget B", 5, 125],
    ["2025-08-03", "Widget A", 7, 175],
    ["2025-08-04", "Widget C", 3, 90],
]
buf = io.StringIO()
csv.writer(buf).writerows(rows)
backend.upload("sales.csv", buf.getvalue().encode())

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze sales.csv. Summarize trends."}]
})
```

***

## Step 3: Add context management

For longer analysis sessions the context window fills. `SummarizationMiddleware` compresses history automatically so the agent keeps working without hitting token limits.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.middleware import FilesystemMiddleware, SummarizationMiddleware

model = "anthropic:claude-sonnet-4-6"

agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(model=model, backend=backend),
    ],
)
```

***

## Step 4: Add skills

Skills give the agent on-demand domain knowledge via progressive disclosure: loaded only when the current task calls for it. Create a skill file in your skills directory:

```
skills/
  pandas-patterns/
    SKILL.md
```

```markdown theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
---
name: pandas-patterns
description: Common pandas and matplotlib patterns for data analysis and visualization
---

## Data loading
Use `pd.read_csv()` for CSV files. Always check `df.info()` and `df.describe()` first.

## Visualization
Use `matplotlib` for bar charts, `seaborn` for statistical plots.
Save figures with `plt.savefig("output.png", dpi=150, bbox_inches="tight")`.

## Reporting
Write a markdown summary to `report.md` alongside any generated charts.
```

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents.middleware import FilesystemMiddleware, SkillsMiddleware, SummarizationMiddleware

agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(model=model, backend=backend),
        SkillsMiddleware(backend=backend, sources=["./skills/"]),
    ],
)
```

***

## Step 5: Add a visualization subagent

Some tasks benefit from isolation. A visualization subagent runs in its own context window, keeping chart generation separate from the main analysis: and enabling parallel execution.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from langchain.agents.middleware import TodoListMiddleware
from deepagents import SubAgent
from deepagents.middleware import (
    FilesystemMiddleware,
    SkillsMiddleware,
    SubAgentMiddleware,
    SummarizationMiddleware,
)

visualizer: SubAgent = {
    "name": "visualizer",
    "description": "Generates charts and visualizations from data files in the sandbox.",
    "system_prompt": "You are a data visualization specialist. Write Python scripts using matplotlib and seaborn. Save all figures as PNG files.",
    "tools": [],
}

agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(model=model, backend=backend),
        SkillsMiddleware(backend=backend, sources=["./skills/"]),
        TodoListMiddleware(),
        SubAgentMiddleware(backend=backend, subagents=[visualizer]),
    ],
)
```

The main agent handles analysis and planning; it delegates chart generation to the `visualizer` subagent via the `task` tool.

***

## What you built

| Middleware                                                                                                                                                                                                                             | What it adds                         |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| [`FilesystemMiddleware`](https://reference.langchain.com/python/deepagents/middleware/filesystem/FilesystemMiddleware) + `LangSmithSandbox`                                                                                            | Isolated filesystem + `execute` tool |
| [`SummarizationMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/summarization/SummarizationMiddleware)                                                                                                  | Automatic context compression        |
| [`SkillsMiddleware`](https://reference.langchain.com/python/deepagents/middleware/skills/SkillsMiddleware)                                                                                                                             | Domain knowledge loaded on demand    |
| [`TodoListMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/todo/TodoListMiddleware) + [`SubAgentMiddleware`](https://reference.langchain.com/python/deepagents/middleware/subagents/SubAgentMiddleware) | Parallel visualization subagent      |

This is the same foundation as `create_deep_agent`: assembled manually so you control exactly what's included. The possibilities don't end here: see [Prebuilt middleware](/oss/python/langchain/middleware/built-in) for the full list of composable capabilities, and the [`create_agent`](https://reference.langchain.com/python/langchain/agents/factory/create_agent) reference for all configuration options.

For a pre-assembled version, see [`create_deep_agent`](https://reference.langchain.com/python/deepagents/graph/create_deep_agent) and [Customize Deep Agents](/oss/python/deepagents/customization). For the full data analysis example using `create_deep_agent`, see [Data analysis](/oss/python/deepagents/data-analysis).

***

<div className="source-links">
  <Callout icon="terminal-2">
    [Connect these docs](/use-these-docs) to Claude, VSCode, and more via MCP for real-time answers.
  </Callout>

  <Callout icon="edit">
    [Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/deep-agent-from-scratch.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).
  </Callout>
</div>
