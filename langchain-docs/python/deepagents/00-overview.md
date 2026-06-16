# Deep Agents overview

> Build agents that can plan, use subagents, and leverage file systems for complex tasks

The easiest way to start building agents and applications powered by LLMs—with built-in capabilities for task planning, file systems for context management, subagent-spawning, and long-term memory.
You can use deep agents for any task, including complex, multi-step tasks.

Deep Agents is an ["agent harness"](/oss/python/concepts/products#agent-harnesses-like-the-deep-agents-sdk). It is the same core tool calling loop as other agent frameworks, but with built-in capabilities that make agents reliable for real tasks:

<CardGroup cols={2}>
  <Card title="Take actions in an environment" icon="bolt">
    Take actions via tools, read and write files, execute code
  </Card>

  <Card title="Connect to your data" icon="database">
    Load memories, skills, and domain knowledge at the right moment
  </Card>

  <Card title="Manage growing context" icon="scissors">
    Summarize history and offload large results across long runs
  </Card>

  <Card title="Parallelize tasks" icon="sitemap">
    Delegate to general or specialized subagents running in isolated context windows
  </Card>

  <Card title="Stay in the loop" icon="user">
    Pause for human approval at critical decision points
  </Card>

  <Card title="Improve over time" icon="rocket">
    Update memory, skills, and prompts based on real usage
  </Card>
</CardGroup>

See [Harness capabilities](/oss/python/deepagents/harness) for a full breakdown of each component.

[`deepagents`](https://pypi.org/project/deepagents/) is a standalone library built on top of [LangChain](/oss/python/langchain/)'s core building blocks for agents. It uses the [LangGraph](/oss/python/langgraph/) runtime for durable execution, streaming, human-in-the-loop, and other features.

[LangChain](/oss/python/langchain/) is the framework that provides the core building blocks for your agents.
To learn more about the differences between LangChain, LangGraph, and Deep Agents, see [Frameworks, runtimes, and harnesses](/oss/python/concepts/products). For a side-by-side comparison with Anthropic's harness, see [Deep Agents vs. Claude Agent SDK](/oss/python/deepagents/comparison).

## <Icon icon="wand" /> Create a deep agent

<Tabs>
  <Tab title="Google">
    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    # pip install -qU deepagents langchain-google-genai
    from deepagents import create_deep_agent

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    agent = create_deep_agent(
        model="google_genai:gemini-3.5-flash",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent
    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    ```
  </Tab>

  <Tab title="OpenAI">
    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    # pip install -qU deepagents langchain-openai
    from deepagents import create_deep_agent

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    agent = create_deep_agent(
        model="openai:gpt-5.5",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent
    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    ```
  </Tab>

  <Tab title="Anthropic">
    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    # pip install -qU deepagents langchain-anthropic
    from deepagents import create_deep_agent

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent
    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    ```
  </Tab>

  <Tab title="OpenRouter">
    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    # pip install -qU deepagents langchain-openrouter
    from deepagents import create_deep_agent

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    agent = create_deep_agent(
        model="openrouter:anthropic/claude-sonnet-4-6",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent
    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    ```
  </Tab>

  <Tab title="Fireworks">
    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    # pip install -qU deepagents langchain-fireworks
    from deepagents import create_deep_agent

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    agent = create_deep_agent(
        model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent
    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    ```
  </Tab>

  <Tab title="Baseten">
    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    # pip install -qU deepagents langchain-baseten
    from deepagents import create_deep_agent

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    agent = create_deep_agent(
        model="baseten:zai-org/GLM-5",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent
    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    ```
  </Tab>

  <Tab title="Ollama">
    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    # pip install -qU deepagents langchain-ollama
    from deepagents import create_deep_agent

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    agent = create_deep_agent(
        model="ollama:devstral-2",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent
    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    ```
  </Tab>
</Tabs>

See the [Quickstart](/oss/python/deepagents/quickstart/) and [Customization guide](/oss/python/deepagents/customization/) to get started building your own agents and applications with Deep Agents.

<Tip>
  Trace requests, debug agent behavior, and evaluate outputs with [LangSmith](https://smith.langchain.com?utm_source=docs\&utm_medium=cta\&utm_campaign=langsmith-signup\&utm_content=oss-deepagents-overview). Follow the [observability quickstart](/langsmith/observability-quickstart) to get set up. When ready for production, see [Going to production](/oss/python/deepagents/going-to-production) for LangSmith deployment options.
</Tip>

## Core capabilities

Use the **Deep Agents SDK** to build agents that handle complex, multi-step tasks across **any [model provider](/oss/python/deepagents/models)**. The SDK ships with the following built-in capabilities:

<Card title="Planning and task decomposition" icon="timeline">
  A built-in [`write_todos`](/oss/python/langchain/middleware/built-in#to-do-list) tool lets agents break down complex tasks into discrete steps, track progress, and adapt plans as new information emerges.
</Card>

<Card title="Context management" icon="scissors">
  Built-in [context compression](/oss/python/deepagents/context-engineering#context-compression) offloads large tool inputs and results to the [virtual filesystem](/oss/python/deepagents/harness#virtual-filesystem-access) and [summarizes](/oss/python/deepagents/context-engineering#summarization) older messages to keep agents effective across extended sessions.
</Card>

<Card title="Tools and MCP" icon="tool">
  Pass custom functions, LangChain tools, or tools from any [MCP server](/oss/python/deepagents/tools#mcp-tools) to `create_deep_agent`. Deep Agents fully support the [Model Context Protocol (MCP)](/oss/python/langchain/mcp), letting you connect to databases, APIs, file systems, and more through a standard interface.
</Card>

<Card title="Pluggable filesystem backends" icon="plug">
  Swap the virtual filesystem via [pluggable backends](/oss/python/deepagents/backends): in-memory state, local disk, LangGraph store, composite routing, or a custom backend with [permission rules](/oss/python/deepagents/permissions) for read and write access.
</Card>

<Card title="Shell execution" icon="terminal">
  Shell-capable backends add an `execute` tool for tests, builds, git operations, and system tasks. Use [`LocalShellBackend`](/oss/python/deepagents/backends#localshellbackend-local-shell) on the host for local development, or a [sandbox backend](/oss/python/deepagents/sandboxes) when you need isolation from your host system.
</Card>

<Card title="Interpreters" icon="code">
  Add an [interpreter](/oss/python/deepagents/interpreters) to run JavaScript in an in-memory runtime. Interpreters let agents compose tools programmatically, orchestrate subagents, and transform structured data without a full shell environment.
</Card>

<Card title="Subagent spawning" icon="users-group">
  A built-in `task` tool spawns general-purpose or specialized [subagents](/oss/python/deepagents/subagents) for context isolation on subtasks. For long-running or parallel work, [async subagents](/oss/python/deepagents/async-subagents) run in the background with progress checks, follow-ups, and cancellation.
</Card>

<Card title="Streaming" icon="broadcast">
  [Event streaming](/oss/python/deepagents/event-streaming) exposes agent runs as typed projections for messages, tool calls, values, and output. Deep Agents add `stream.subagents` so each delegated task gets its own handle with independent message, tool-call, and nested subagent streams.
</Card>

<Card title="Long-term memory" icon="database">
  Persist memory across threads and conversations using LangGraph's [Memory Store](/oss/python/langgraph/stores).
</Card>

<Card title="Filesystem permissions" icon="lock">
  Declare [permission rules](/oss/python/deepagents/permissions) that control which files and directories agents can read or write. Subagents can inherit or override the parent's rules.
</Card>

<Card title="Human-in-the-loop" icon="user-check">
  Configure [human approval](/oss/python/deepagents/human-in-the-loop) for sensitive tool operations using LangGraph's interrupt capabilities.
</Card>

<Card title="Skills" icon="puzzle">
  Extend agents with reusable [skills](/oss/python/deepagents/skills) that provide specialized workflows, domain knowledge, and custom instructions.
</Card>

<Card title="Smart defaults" icon="wand">
  Ships with opinionated system prompts that teach the model to plan before acting, verify work, and manage context. Customize or replace the defaults as needed.
</Card>

For building custom agents without these builtin capabilities, consider using LangChain's [`create_agent`](/oss/python/langchain/agents) or building a custom [LangGraph](/oss/python/langgraph/overview) workflow.

## Get started

<CardGroup cols={2}>
  <Card title="Quickstart" icon="rocket" href="/oss/python/deepagents/quickstart">
    Build your first deep agent
  </Card>

  <Card title="Customization" icon="adjustments" href="/oss/python/deepagents/customization">
    Learn about customization options
  </Card>

  <Card title="Tools and MCP" icon="tool" href="/oss/python/deepagents/tools">
    Connect custom functions, APIs, and MCP servers
  </Card>

  <Card title="Models" icon="cpu" href="/oss/python/deepagents/models">
    Configure models and providers
  </Card>

  <Card title="Backends" icon="plug" href="/oss/python/deepagents/backends">
    Choose and configure pluggable filesystem backends
  </Card>

  <Card title="Sandboxes" icon="cube" href="/oss/python/deepagents/sandboxes">
    Execute code in isolated environments
  </Card>

  <Card title="Interpreters" icon="code" href="/oss/python/deepagents/interpreters">
    Compose tools and transform data in QuickJS
  </Card>

  <Card title="Permissions" icon="lock" href="/oss/python/deepagents/permissions">
    Control filesystem access with permission rules
  </Card>

  <Card title="Human-in-the-loop" icon="user-check" href="/oss/python/deepagents/human-in-the-loop">
    Configure approval for sensitive operations
  </Card>

  <Card title="Code" icon="terminal" href="/oss/python/deepagents/code/overview">
    Use Deep Agents Code
  </Card>

  <Card title="ACP" icon="plug-connected" href="/oss/python/deepagents/acp">
    Use deep agents in code editors via ACP
  </Card>

  <Card title="Reference" icon="external-link" href="https://reference.langchain.com/python/deepagents/">
    See the `deepagents` API reference
  </Card>
</CardGroup>

***

<div className="source-links">
  <Callout icon="terminal-2">
    [Connect these docs](/use-these-docs) to Claude, VSCode, and more via MCP for real-time answers.
  </Callout>

  <Callout icon="edit">
    [Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/deepagents/overview.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).
  </Callout>
</div>
