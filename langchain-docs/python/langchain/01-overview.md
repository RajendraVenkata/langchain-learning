# LangChain overview

> LangChain provides create_agent: a minimal, highly configurable agent harness. Compose exactly the agent your use case needs from model, tools, prompt, and middleware.

**Agent = Model + Harness.** LangChain provides `create_agent`: a minimal, highly configurable harness. The harness is everything around the model loop: the prompt, the tools, and any middleware that shapes behavior. Start with the primitives and compose exactly what your use case needs. Supports [OpenAI, Anthropic, Google, and more](/oss/python/integrations/providers/overview).

<Tip>
  **LangChain vs. LangGraph vs. Deep Agents**

  Start with [Deep Agents](/oss/python/deepagents/overview/) for a "batteries-included" agent with features like automatic context compression, a virtual filesystem, and subagent-spawning. Deep Agents are built on LangChain [agents](/oss/python/langchain/agents/) which you can also use directly.

  Use [LangChain](/oss/python/langchain/agents) (`create_agent`) for a highly customizable harness, easily tailored to your use case and data.

  Use [LangGraph](/oss/python/langgraph/overview), our low-level orchestration framework, for advanced needs combining deterministic and agentic workflows.

  Use [LangSmith](/langsmith/home) to trace, debug, and evaluate agents built with any of these frameworks. Follow the [tracing quickstart](/langsmith/trace-with-langchain) to get set up. We recommend you also set up [LangSmith Engine](/langsmith/engine) which monitors your traces, detects issues, and proposes fixes.
</Tip>

## <Icon icon="wand" /> Create an agent

This example demonstrates how to create a simple LangChain agent with a custom tool:

<CodeGroup>
  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  # pip install -qU langchain "langchain[openai]"
  from langchain.agents import create_agent

  def get_weather(city: str) -> str:
      """Get weather for a given city."""
      return f"It's always sunny in {city}!"

  agent = create_agent(
      model="openai:gpt-5.4",
      tools=[get_weather],
      system_prompt="You are a helpful assistant",
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
  )
  print(result["messages"][-1].content_blocks)
  ```

  ```python Google Gemini theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  # pip install -qU langchain "langchain[google-genai]"
  from langchain.agents import create_agent

  def get_weather(city: str) -> str:
      """Get weather for a given city."""
      return f"It's always sunny in {city}!"

  agent = create_agent(
      model="google_genai:gemini-2.5-flash-lite",
      tools=[get_weather],
      system_prompt="You are a helpful assistant",
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
  )
  print(result["messages"][-1].content_blocks)
  ```

  ```python Claude (Anthropic) theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  # pip install -qU langchain "langchain[anthropic]"
  from langchain.agents import create_agent

  def get_weather(city: str) -> str:
      """Get weather for a given city."""
      return f"It's always sunny in {city}!"

  agent = create_agent(
      model="claude-sonnet-4-6",
      tools=[get_weather],
      system_prompt="You are a helpful assistant",
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
  )
  print(result["messages"][-1].content_blocks)
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  # pip install -qU langchain langchain-openrouter
  from langchain.agents import create_agent

  def get_weather(city: str) -> str:
      """Get weather for a given city."""
      return f"It's always sunny in {city}!"

  agent = create_agent(
      model="openrouter:anthropic/claude-sonnet-4-6",
      tools=[get_weather],
      system_prompt="You are a helpful assistant",
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
  )
  print(result["messages"][-1].content_blocks)
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  # pip install -qU langchain langchain-fireworks
  from langchain.agents import create_agent

  def get_weather(city: str) -> str:
      """Get weather for a given city."""
      return f"It's always sunny in {city}!"

  agent = create_agent(
      model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
      tools=[get_weather],
      system_prompt="You are a helpful assistant",
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
  )
  print(result["messages"][-1].content_blocks)
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  # pip install -qU langchain langchain-baseten
  from langchain.agents import create_agent

  def get_weather(city: str) -> str:
      """Get weather for a given city."""
      return f"It's always sunny in {city}!"

  agent = create_agent(
      model="baseten:zai-org/GLM-5",
      tools=[get_weather],
      system_prompt="You are a helpful assistant",
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
  )
  print(result["messages"][-1].content_blocks)
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  # pip install -qU langchain langchain-ollama
  from langchain.agents import create_agent

  def get_weather(city: str) -> str:
      """Get weather for a given city."""
      return f"It's always sunny in {city}!"

  agent = create_agent(
      model="ollama:devstral-2",
      tools=[get_weather],
      system_prompt="You are a helpful assistant",
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
  )
  print(result["messages"][-1].content_blocks)
  ```

  ```python Azure theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  # pip install -qU langchain "langchain[openai]"
  import os
  from langchain.agents import create_agent

  def get_weather(city: str) -> str:
      """Get weather for a given city."""
      return f"It's always sunny in {city}!"

  agent = create_agent(
      model="azure_openai:gpt-5.4",
      tools=[get_weather],
      system_prompt="You are a helpful assistant",
      azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
  )
  print(result["messages"][-1].content_blocks)
  ```

  ```python AWS Bedrock theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  # pip install -qU langchain langchain-aws
  from langchain.agents import create_agent

  def get_weather(city: str) -> str:
      """Get weather for a given city."""
      return f"It's always sunny in {city}!"

  agent = create_agent(
      model="anthropic.claude-3-5-sonnet-20240620-v1:0",
      model_provider="bedrock_converse",
      tools=[get_weather],
      system_prompt="You are a helpful assistant",
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
  )
  print(result["messages"][-1].content_blocks)
  ```

  ```python HuggingFace theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  # pip install -qU langchain "langchain[huggingface]"
  from langchain.agents import create_agent

  def get_weather(city: str) -> str:
      """Get weather for a given city."""
      return f"It's always sunny in {city}!"

  agent = create_agent(
      model="microsoft/Phi-3-mini-4k-instruct",
      model_provider="huggingface",
      tools=[get_weather],
      system_prompt="You are a helpful assistant",
      temperature=0.7,
      max_tokens=1024,
  )

  result = agent.invoke(
      {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
  )
  print(result["messages"][-1].content_blocks)
  ```
</CodeGroup>

See the [Installation instructions](/oss/python/langchain/install) and [Quickstart guide](/oss/python/langchain/quickstart) to get started building your own agents and applications with LangChain.

<Tip>
  Use [LangSmith](/langsmith/home) to trace requests, debug agent behavior, and evaluate outputs. Set `LANGSMITH_TRACING=true` and your API key to get started.
</Tip>

## <Icon icon="star" size={20} /> Core benefits

<Columns cols={2}>
  <Card title="Standard model interface" icon="refresh" href="/oss/python/langchain/models" arrow cta="Learn more">
    Different providers have unique APIs for interacting with models, including the format of responses. LangChain standardizes how you interact with models so that you can seamlessly swap providers and avoid lock-in.
  </Card>

  <Card title="Highly configurable harness" icon="wand" href="/oss/python/langchain/agents" arrow cta="Learn more">
    `create_agent` is a minimal harness: model, tools, prompt, loop. Extend it with middleware: each piece handles one concern and composes freely. Build exactly the agent your use case needs, nothing more.
  </Card>

  <Card title="Built on top of LangGraph" icon="https://mintcdn.com/langchain-5e9cc07a/nQm-sjd_MByLhgeW/images/brand/langgraph-icon.png?fit=max&auto=format&n=nQm-sjd_MByLhgeW&q=85&s=b997e1a7487d507a36556eedbfd99f81" href="/oss/python/langgraph/overview" arrow cta="Learn more" width="195" height="195" data-path="images/brand/langgraph-icon.png">
    LangChain's agents are built on top of LangGraph. This allows us to take advantage of LangGraph's durable execution, human-in-the-loop support, persistence, and more.
  </Card>

  <Card title="Debug with LangSmith" icon="https://mintcdn.com/langchain-5e9cc07a/nQm-sjd_MByLhgeW/images/brand/observability-icon-dark.png?fit=max&auto=format&n=nQm-sjd_MByLhgeW&q=85&s=ccbc183bca2a5e4ca78d30149e3836cc" href="/langsmith/observability" arrow cta="Learn more" width="200" height="200" data-path="images/brand/observability-icon-dark.png">
    Gain deep visibility into complex agent behavior with visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics.
  </Card>
</Columns>

***

<div className="source-links">
  <Callout icon="terminal-2">
    [Connect these docs](/use-these-docs) to Claude, VSCode, and more via MCP for real-time answers.
  </Callout>

  <Callout icon="edit">
    [Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/overview.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).
  </Callout>
</div>
