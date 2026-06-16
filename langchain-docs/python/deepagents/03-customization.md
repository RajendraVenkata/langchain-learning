# Customize Deep Agents

> Learn how to customize Deep Agents with system prompts, tools, subagents, and more

Build the harness around your goal. `create_deep_agent` gives you a production-ready foundation: connect it to your data, shape its behavior, and add the capabilities your use case needs.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    system_prompt="You are a helpful assistant.",
    tools=[search, fetch_url],
    memory=["./AGENTS.md"],
    skills=["./skills/"],
)
```

| Parameter                                                                         | What it does                                                                |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| [`model=`](#model)                                                                | Which model to use                                                          |
| [`system_prompt=`](#system-prompt)                                                | Custom instructions for the agent                                           |
| [`tools=`](#tools)                                                                | Domain tools the agent can call                                             |
| [`memory=`](#memory)                                                              | AGENTS.md files loaded at startup                                           |
| [`skills=`](#skills)                                                              | Skills directory for on-demand knowledge                                    |
| [`backend=`](#backends)                                                           | Filesystem backend (StateBackend by default)                                |
| [`permissions=`](/oss/python/deepagents/permissions)                              | Path-level access control for the filesystem                                |
| [`subagents=`](#subagents)                                                        | Custom subagents for delegated tasks                                        |
| [`middleware=`](#middleware)                                                      | Extra middleware appended to the [default stack](#default-stack-main-agent) |
| [`interrupt_on=`](#human-in-the-loop)                                             | Pause before tool calls for human approval                                  |
| [`response_format=`](#structured-output)                                          | Structured output schema                                                    |
| [`state_schema=`](/oss/python/deepagents/context-engineering#custom-state-schema) | Custom graph state schema                                                   |
| [profiles](#profiles)                                                             | Per-model defaults as a reusable bundle                                     |

<Accordion title="Full function signature">
  ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  create_deep_agent(
      model: str | BaseChatModel | None = None,
      tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
      *,
      system_prompt: str | SystemMessage | None = None,
      middleware: Sequence[AgentMiddleware] = (),
      subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
      skills: list[str] | None = None,
      memory: list[str] | None = None,
      permissions: list[FilesystemPermission] | None = None,
      backend: BackendProtocol | BackendFactory | None = None,
      interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
      response_format: ResponseFormat[ResponseT] | type[ResponseT] | dict[str, Any] | None = None,
      state_schema: type[DeepAgentState] | None = None,
      context_schema: type[ContextT] | None = None,
      checkpointer: Checkpointer | None = None,
      store: BaseStore | None = None,
      debug: bool = False,
      name: str | None = None,
      cache: BaseCache | None = None
  ) -> CompiledStateGraph[AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]]
  ```
</Accordion>

For the full parameter list, see the [`create_deep_agent`](https://reference.langchain.com/python/deepagents/graph/create_deep_agent) API reference. To compose a fully custom harness from scratch, see [Configure the harness](/oss/python/langchain/agents#configure-the-harness) or follow the step-by-step [Build a deep agent from scratch](/oss/python/langchain/deep-agent-from-scratch) guide.

<Tip>
  As you add tools, subagents, and backends, use [LangSmith](https://smith.langchain.com?utm_source=docs\&utm_medium=cta\&utm_campaign=langsmith-signup\&utm_content=oss-deepagents-customization) to trace how each piece behaves together. Follow the [observability quickstart](/langsmith/observability-quickstart) to get set up, and see [Going to production](/oss/python/deepagents/going-to-production) for deployment on LangSmith.

  We recommend you also set up [LangSmith Engine](/langsmith/engine), which monitors your traces, detects issues, and proposes fixes.
</Tip>

## Model

Pass a `model` string in `provider:model` format, or an initialized model instance. See [supported models](/oss/python/deepagents/models#supported-models) for all providers and [suggested models](/oss/python/deepagents/models#suggested-models) for tested recommendations.

<Tip>
  Use the `provider:model` format (for example `openai:gpt-5.5`) to quickly switch between models.
</Tip>

<Tabs>
  <Tab title="OpenAI">
    👉 Read the [OpenAI chat model integration docs](/oss/python/integrations/chat/openai/)

    ```shell theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    pip install -U "langchain[openai]"
    ```

    <CodeGroup>
      ```python default parameters theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from deepagents import create_deep_agent

      os.environ["OPENAI_API_KEY"] = "sk-..."

      agent = create_deep_agent(model="openai:gpt-5.5")
      # this calls init_chat_model for the specified model with default parameters
      # to use specific model parameters, use init_chat_model directly
      ```

      ```python init_chat_model theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from langchain.chat_models import init_chat_model
      from deepagents import create_deep_agent

      os.environ["OPENAI_API_KEY"] = "sk-..."

      model = init_chat_model(model="openai:gpt-5.5")
      agent = create_deep_agent(model=model)
      ```

      ```python Model Class theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from langchain_openai import ChatOpenAI
      from deepagents import create_deep_agent

      os.environ["OPENAI_API_KEY"] = "sk-..."

      model = ChatOpenAI(model="gpt-5.5")
      agent = create_deep_agent(model=model)
      ```
    </CodeGroup>
  </Tab>

  <Tab title="Anthropic">
    👉 Read the [Anthropic chat model integration docs](/oss/python/integrations/chat/anthropic/)

    ```shell theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    pip install -U "langchain[anthropic]"
    ```

    <CodeGroup>
      ```python default parameters theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from deepagents import create_deep_agent

      os.environ["ANTHROPIC_API_KEY"] = "sk-..."

      agent = create_deep_agent(model="anthropic:claude-sonnet-4-6")
      # this calls init_chat_model for the specified model with default parameters
      # to use specific model parameters, use init_chat_model directly
      ```

      ```python init_chat_model theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from langchain.chat_models import init_chat_model
      from deepagents import create_deep_agent

      os.environ["ANTHROPIC_API_KEY"] = "sk-..."

      model = init_chat_model(model="claude-sonnet-4-6")
      agent = create_deep_agent(model=model)
      ```

      ```python Model Class theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from langchain_anthropic import ChatAnthropic
      from deepagents import create_deep_agent

      os.environ["ANTHROPIC_API_KEY"] = "sk-..."

      model = ChatAnthropic(model="claude-sonnet-4-6")
      agent = create_deep_agent(model=model)
      ```
    </CodeGroup>
  </Tab>

  <Tab title="Azure">
    👉 Read the [Azure chat model integration docs](/oss/python/integrations/chat/azure_chat_openai/)

    ```shell theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    pip install -U "langchain[openai]"
    ```

    <CodeGroup>
      ```python default parameters theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from deepagents import create_deep_agent

      os.environ["AZURE_OPENAI_API_KEY"] = "..."
      os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
      os.environ["OPENAI_API_VERSION"] = "2025-03-01-preview"

      agent = create_deep_agent(model="azure_openai:gpt-5.5")
      # this calls init_chat_model for the specified model with default parameters
      # to use specific model parameters, use init_chat_model directly
      ```

      ```python init_chat_model theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from langchain.chat_models import init_chat_model
      from deepagents import create_deep_agent

      os.environ["AZURE_OPENAI_API_KEY"] = "..."
      os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
      os.environ["OPENAI_API_VERSION"] = "2025-03-01-preview"

      model = init_chat_model(
          model="azure_openai:gpt-5.5",
          azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
      )
      agent = create_deep_agent(model=model)
      ```

      ```python Model Class theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from langchain_openai import AzureChatOpenAI
      from deepagents import create_deep_agent

      os.environ["AZURE_OPENAI_API_KEY"] = "..."
      os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
      os.environ["OPENAI_API_VERSION"] = "2025-03-01-preview"

      model = AzureChatOpenAI(
          model="gpt-5.5",
          azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
      )
      agent = create_deep_agent(model=model)
      ```
    </CodeGroup>
  </Tab>

  <Tab title="Google Gemini">
    👉 Read the [Google GenAI chat model integration docs](/oss/python/integrations/chat/google_generative_ai/)

    ```shell theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    pip install -U "langchain[google-genai]"
    ```

    <CodeGroup>
      ```python default parameters theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from deepagents import create_deep_agent

      os.environ["GOOGLE_API_KEY"] = "..."

      agent = create_deep_agent(model="google_genai:gemini-3.5-flash")
      # this calls init_chat_model for the specified model with default parameters
      # to use specific model parameters, use init_chat_model directly
      ```

      ```python init_chat_model theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from langchain.chat_models import init_chat_model
      from deepagents import create_deep_agent

      os.environ["GOOGLE_API_KEY"] = "..."

      model = init_chat_model(model="google_genai:gemini-3.5-flash")
      agent = create_deep_agent(model=model)
      ```

      ```python Model Class theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from langchain_google_genai import ChatGoogleGenerativeAI
      from deepagents import create_deep_agent

      os.environ["GOOGLE_API_KEY"] = "..."

      model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")
      agent = create_deep_agent(model=model)
      ```
    </CodeGroup>
  </Tab>

  <Tab title="AWS Bedrock">
    👉 Read the [AWS Bedrock chat model integration docs](/oss/python/integrations/chat/bedrock/)

    ```shell theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    pip install -U "langchain[aws]"
    ```

    <CodeGroup>
      ```python default parameters theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent

      # Follow the steps here to configure your credentials:
      # https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html

      agent = create_deep_agent(
          model="anthropic.claude-sonnet-4-6",
          model_provider="bedrock_converse",
      )
      # this calls init_chat_model for the specified model with default parameters
      # to use specific model parameters, use init_chat_model directly
      ```

      ```python init_chat_model theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from langchain.chat_models import init_chat_model
      from deepagents import create_deep_agent

      # Follow the steps here to configure your credentials:
      # https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html

      model = init_chat_model(
          model="anthropic.claude-sonnet-4-6",
          model_provider="bedrock_converse",
      )
      agent = create_deep_agent(model=model)
      ```

      ```python Model Class theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from langchain_aws import ChatBedrock
      from deepagents import create_deep_agent

      # Follow the steps here to configure your credentials:
      # https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html

      model = ChatBedrock(model="anthropic.claude-sonnet-4-6")
      agent = create_deep_agent(model=model)
      ```
    </CodeGroup>
  </Tab>

  <Tab title="HuggingFace">
    👉 Read the [HuggingFace chat model integration docs](/oss/python/integrations/chat/huggingface/)

    ```shell theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    pip install -U "langchain[huggingface]"
    ```

    <CodeGroup>
      ```python default parameters theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from deepagents import create_deep_agent

      os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."

      agent = create_deep_agent(
          model="microsoft/Phi-3-mini-4k-instruct",
          model_provider="huggingface",
          temperature=0.7,
          max_tokens=1024,
      )
      # this calls init_chat_model for the specified model with default parameters
      # to use specific model parameters, use init_chat_model directly
      ```

      ```python init_chat_model theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from langchain.chat_models import init_chat_model
      from deepagents import create_deep_agent

      os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."

      model = init_chat_model(
          model="microsoft/Phi-3-mini-4k-instruct",
          model_provider="huggingface",
          temperature=0.7,
          max_tokens=1024,
      )
      agent = create_deep_agent(model=model)
      ```

      ```python Model Class theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      import os
      from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
      from deepagents import create_deep_agent

      os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."

      llm = HuggingFaceEndpoint(
          repo_id="microsoft/Phi-3-mini-4k-instruct",
          temperature=0.7,
          max_length=1024,
      )
      model = ChatHuggingFace(llm=llm)
      agent = create_deep_agent(model=model)
      ```
    </CodeGroup>
  </Tab>

  <Tab title="Other">
    Pass any [supported model string](/oss/python/deepagents/models#supported-models), or an initialized model instance:

    <CodeGroup>
      ```python model string theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent

      agent = create_deep_agent(model="provider:model-name")
      ```

      ```python init_chat_model theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from langchain.chat_models import init_chat_model

      model = init_chat_model("provider:model-name")
      agent = create_deep_agent(model=model)
      ```

      ```python model class theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from langchain_<provider> import Chat<Provider>
      from deepagents import create_deep_agent

      model = Chat<Provider>(model="model-name")
      agent = create_deep_agent(model=model)
      ```
    </CodeGroup>
  </Tab>
</Tabs>

<Tip>
  Chat models automatically retry transient API failures (with exponential backoff). For defaults, limits, and code samples for tuning `max_retries` / `timeout` live on the LangChain [Models](/oss/python/langchain/models#connection-resilience) page.
</Tip>

## Tools

In addition to [built-in tools](/oss/python/deepagents/overview#core-capabilities) for planning, file management, and subagent spawning, you can provide custom tools:

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  import os
  from typing import Literal
  from tavily import TavilyClient
  from deepagents import create_deep_agent

  tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


  def internet_search(
      query: str,
      max_results: int = 5,
      topic: Literal["general", "news", "finance"] = "general",
      include_raw_content: bool = False,
  ):
      """Run a web search"""
      return tavily_client.search(
          query,
          max_results=max_results,
          include_raw_content=include_raw_content,
          topic=topic,
      )


  agent = create_deep_agent(
      model="google_genai:gemini-3.5-flash",
      tools=[internet_search],
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  import os
  from typing import Literal
  from tavily import TavilyClient
  from deepagents import create_deep_agent

  tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


  def internet_search(
      query: str,
      max_results: int = 5,
      topic: Literal["general", "news", "finance"] = "general",
      include_raw_content: bool = False,
  ):
      """Run a web search"""
      return tavily_client.search(
          query,
          max_results=max_results,
          include_raw_content=include_raw_content,
          topic=topic,
      )


  agent = create_deep_agent(
      model="openai:gpt-5.5",
      tools=[internet_search],
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  import os
  from typing import Literal
  from tavily import TavilyClient
  from deepagents import create_deep_agent

  tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


  def internet_search(
      query: str,
      max_results: int = 5,
      topic: Literal["general", "news", "finance"] = "general",
      include_raw_content: bool = False,
  ):
      """Run a web search"""
      return tavily_client.search(
          query,
          max_results=max_results,
          include_raw_content=include_raw_content,
          topic=topic,
      )


  agent = create_deep_agent(
      model="anthropic:claude-sonnet-4-6",
      tools=[internet_search],
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  import os
  from typing import Literal
  from tavily import TavilyClient
  from deepagents import create_deep_agent

  tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


  def internet_search(
      query: str,
      max_results: int = 5,
      topic: Literal["general", "news", "finance"] = "general",
      include_raw_content: bool = False,
  ):
      """Run a web search"""
      return tavily_client.search(
          query,
          max_results=max_results,
          include_raw_content=include_raw_content,
          topic=topic,
      )


  agent = create_deep_agent(
      model="openrouter:anthropic/claude-sonnet-4-6",
      tools=[internet_search],
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  import os
  from typing import Literal
  from tavily import TavilyClient
  from deepagents import create_deep_agent

  tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


  def internet_search(
      query: str,
      max_results: int = 5,
      topic: Literal["general", "news", "finance"] = "general",
      include_raw_content: bool = False,
  ):
      """Run a web search"""
      return tavily_client.search(
          query,
          max_results=max_results,
          include_raw_content=include_raw_content,
          topic=topic,
      )


  agent = create_deep_agent(
      model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
      tools=[internet_search],
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  import os
  from typing import Literal
  from tavily import TavilyClient
  from deepagents import create_deep_agent

  tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


  def internet_search(
      query: str,
      max_results: int = 5,
      topic: Literal["general", "news", "finance"] = "general",
      include_raw_content: bool = False,
  ):
      """Run a web search"""
      return tavily_client.search(
          query,
          max_results=max_results,
          include_raw_content=include_raw_content,
          topic=topic,
      )


  agent = create_deep_agent(
      model="baseten:zai-org/GLM-5",
      tools=[internet_search],
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  import os
  from typing import Literal
  from tavily import TavilyClient
  from deepagents import create_deep_agent

  tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


  def internet_search(
      query: str,
      max_results: int = 5,
      topic: Literal["general", "news", "finance"] = "general",
      include_raw_content: bool = False,
  ):
      """Run a web search"""
      return tavily_client.search(
          query,
          max_results=max_results,
          include_raw_content=include_raw_content,
          topic=topic,
      )


  agent = create_deep_agent(
      model="ollama:devstral-2",
      tools=[internet_search],
  )
  ```
</CodeGroup>

### MCP tools

<Tip>
  Deep Agents fully support [Model Context Protocol (MCP)](/oss/python/langchain/mcp) tools. You can load tools from any MCP server—databases, APIs, file systems, and more—and pass them directly to `create_deep_agent`.
</Tip>

Install `langchain-mcp-adapters` to connect to MCP servers:

```bash theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
pip install langchain-mcp-adapters
```

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent

async def main():
    async with MultiServerMCPClient(
        {
            "my_server": {
                "transport": "http",
                "url": "http://localhost:8000/mcp",
            }
        }
    ) as client:
        tools = await client.get_tools()

        agent = create_deep_agent(
            model="openai:gpt-5.5",
            tools=tools,
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Use the MCP server to help me."}]},
            config={"configurable": {"thread_id": "1"}},
        )

asyncio.run(main())
```

For detailed configuration options including stdio servers, OAuth authentication, tool filtering, and stateful sessions, see the full [MCP guide](/oss/python/langchain/mcp).

## System prompt

Deep Agents come with a built-in system prompt. A deep agent's value comes from the orchestration layer the SDK provides on top of the model—planning, virtual-filesystem tools, and subagents—and the model needs to know those exist and when to reach for them. The built-in prompt teaches the agent how to use that scaffolding so you don't have to re-derive it for every project; tweak it through a [profile](/oss/python/deepagents/profiles#harness-profiles) or your own `system_prompt=` rather than copying it verbatim.

When middleware add special tools, like the filesystem tools, it appends them to the system prompt.

Each deep agent should also include a custom system prompt specific to its specific use case:

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent

  research_instructions = """\
  You are an expert researcher. Your job is to conduct \
  thorough research, and then write a polished report. \
  """

  agent = create_deep_agent(
      model="google_genai:gemini-3.5-flash",
      system_prompt=research_instructions,
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent

  research_instructions = """\
  You are an expert researcher. Your job is to conduct \
  thorough research, and then write a polished report. \
  """

  agent = create_deep_agent(
      model="openai:gpt-5.5",
      system_prompt=research_instructions,
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent

  research_instructions = """\
  You are an expert researcher. Your job is to conduct \
  thorough research, and then write a polished report. \
  """

  agent = create_deep_agent(
      model="anthropic:claude-sonnet-4-6",
      system_prompt=research_instructions,
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent

  research_instructions = """\
  You are an expert researcher. Your job is to conduct \
  thorough research, and then write a polished report. \
  """

  agent = create_deep_agent(
      model="openrouter:anthropic/claude-sonnet-4-6",
      system_prompt=research_instructions,
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent

  research_instructions = """\
  You are an expert researcher. Your job is to conduct \
  thorough research, and then write a polished report. \
  """

  agent = create_deep_agent(
      model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
      system_prompt=research_instructions,
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent

  research_instructions = """\
  You are an expert researcher. Your job is to conduct \
  thorough research, and then write a polished report. \
  """

  agent = create_deep_agent(
      model="baseten:zai-org/GLM-5",
      system_prompt=research_instructions,
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent

  research_instructions = """\
  You are an expert researcher. Your job is to conduct \
  thorough research, and then write a polished report. \
  """

  agent = create_deep_agent(
      model="ollama:devstral-2",
      system_prompt=research_instructions,
  )
  ```
</CodeGroup>

### Prompt assembly

Deep Agents builds the system prompt from up to four named parts so that caller-supplied instructions, the SDK's built-in agent guidance, and any model-specific [profile](/oss/python/deepagents/profiles) overrides can coexist with predictable precedence. Without this layering, a profile suffix tuned for Claude (for example) could overwrite or be overwritten by your `system_prompt=` argument depending on call order; the named slots make the ordering explicit and stable.

In practice, most callers only encounter two slots: `USER` (your `system_prompt=`) and `BASE` (the SDK default). Selecting a model with a built-in profile—Anthropic or OpenAI today—adds a `SUFFIX`. The full four-part assembly is mainly relevant when you author a custom `HarnessProfile` or debug why a profile's text appears where it does.

The four named parts (each may be absent):

| Name     | Source                                                                                    | Notes                                                     |
| -------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `USER`   | `system_prompt=` argument to `create_deep_agent`                                          | `str` or `SystemMessage`; omitted when unset.             |
| `BASE`   | The SDK default (`BASE_AGENT_PROMPT`)                                                     | Always present unless replaced by a profile's `CUSTOM`.   |
| `CUSTOM` | [`HarnessProfile.base_system_prompt`](/oss/python/deepagents/profiles#harness-profiles)   | Replaces `BASE` outright when a matching profile sets it. |
| `SUFFIX` | [`HarnessProfile.system_prompt_suffix`](/oss/python/deepagents/profiles#harness-profiles) | Appended last when a matching profile sets it.            |

The order is always **`USER` -> (`BASE` or `CUSTOM`) -> `SUFFIX`**, joined by blank lines (`\n\n`). Two invariants follow:

1. **`USER` is always at the front.** The caller's text precedes any SDK or profile content, so persona/instructions take precedence regardless of which model is selected.
2. **`SUFFIX` is always at the end.** Profile suffixes sit closest to the conversation history, where model-tuning guidance lands most reliably.

Assembled shapes (✓ = field is set, - = field is unset):

| `system_prompt=` | profile `base_system_prompt` (`CUSTOM`) | profile `system_prompt_suffix` (`SUFFIX`) | Final assembled system prompt |
| ---------------- | :-------------------------------------: | :---------------------------------------: | ----------------------------- |
| `None`           |                    -                    |                     -                     | `BASE`                        |
| `None`           |                    -                    |                     ✓                     | `BASE` + `SUFFIX`             |
| `None`           |                    ✓                    |                     -                     | `CUSTOM`                      |
| `None`           |                    ✓                    |                     ✓                     | `CUSTOM` + `SUFFIX`           |
| `str`            |                    -                    |                     -                     | `USER` + `BASE`               |
| `str`            |                    -                    |                     ✓                     | `USER` + `BASE` + `SUFFIX`    |
| `str`            |                    ✓                    |                     -                     | `USER` + `CUSTOM`             |
| `str`            |                    ✓                    |                     ✓                     | `USER` + `CUSTOM` + `SUFFIX`  |

Worked example—built-in profiles (Anthropic, OpenAI) ship only a `system_prompt_suffix`, so a typical call lands in the `str` + `-` + `✓` row:

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    system_prompt="You are a customer-support agent for ACME Corp.",
)
# Final = USER + BASE + SUFFIX
#       = "You are a customer-support agent for ACME Corp."
#         + "\n\n"
#         + BASE_AGENT_PROMPT
#         + "\n\n"
#         + <Claude-specific guidance>
```

<Note>
  Passing a `SystemMessage` (rather than a string) triggers a different concatenation path: the right-hand assembly (`BASE`-or-`CUSTOM` plus any `SUFFIX`) is appended as an additional text content block onto the message's existing `content_blocks`. The same logical ordering applies (caller blocks first), and any `cache_control` markers on the caller's blocks are preserved—useful for placing explicit Anthropic prompt-cache breakpoints.
</Note>

<AccordionGroup>
  <Accordion title="Subagent prompts">
    The [prompt assembly](#prompt-assembly) overlay rules also apply to declarative [subagents](/oss/python/deepagents/subagents): each subagent re-runs profile resolution against **its own model**, then applies the resolved profile's `base_system_prompt` / `system_prompt_suffix` to its authored `system_prompt`. The subagent's `system_prompt` plays the `BASE` role; `CUSTOM` and `SUFFIX` come from the profile that matches the subagent's model (which may differ from the main agent's profile).

    | `spec["system_prompt"]` | profile `base_system_prompt` (`CUSTOM`) | profile `system_prompt_suffix` (`SUFFIX`) | Final subagent system prompt |
    | ----------------------- | :-------------------------------------: | :---------------------------------------: | ---------------------------- |
    | authored                |                    -                    |                     -                     | authored                     |
    | authored                |                    -                    |                     ✓                     | authored + `SUFFIX`          |
    | authored                |                    ✓                    |                     -                     | `CUSTOM`                     |
    | authored                |                    ✓                    |                     ✓                     | `CUSTOM` + `SUFFIX`          |

    There is no `USER` segment for subagents. The spec's authored `system_prompt` is the closest analog and stays in the `BASE` slot. A profile that ships only a `system_prompt_suffix` (the common case for built-in Anthropic / OpenAI profiles) just appends to whatever the subagent author wrote. A profile that sets `base_system_prompt` will *replace* the authored prompt outright.
  </Accordion>

  <Accordion title="General-purpose subagent prompt">
    The auto-added [general-purpose subagent](/oss/python/deepagents/subagents#the-general-purpose-subagent) follows the [prompt assembly](#prompt-assembly) overlay rules with one extra layer: the GP base prompt is resolved as **`general_purpose_subagent.system_prompt` (if set) -> `HarnessProfile.base_system_prompt` (if set) -> SDK general-purpose default**. The profile suffix layers on top either way.

    The two override fields can both carry a base-prompt replacement, but they are not interchangeable. `general_purpose_subagent.system_prompt` is general-purpose-specific configuration; `base_system_prompt` is a global override that primarily targets the main agent. When both are set, the **general-purpose-specific intent wins for the general-purpose subagent** so a user tuning both fields never sees their GP override silently dropped:

    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    register_harness_profile(
        "anthropic",
        HarnessProfile(
            base_system_prompt="You are ACME's support orchestrator.",  # main agent
            general_purpose_subagent=GeneralPurposeSubagentProfile(
                system_prompt="You are a research subagent. Cite sources.",  # GP subagent
            ),
            system_prompt_suffix="Always think step by step.",
        ),
    )
    ```

    | Stack       | Final system prompt                                     |
    | ----------- | ------------------------------------------------------- |
    | Main agent  | `"You are ACME's support orchestrator." + SUFFIX`       |
    | GP subagent | `"You are a research subagent. Cite sources." + SUFFIX` |

    If `general_purpose_subagent.system_prompt` is unset, the GP subagent falls back to `base_system_prompt` (when set) and finally to the SDK general-purpose default.
  </Accordion>
</AccordionGroup>

## Middleware

Deep Agents support any [middleware](/oss/python/langchain/middleware/overview), including the built-in middleware listed below, prebuilt middleware from LangChain, provider-specific middleware, and custom middleware you write yourself.

Pass middleware to the `middleware` argument of `create_deep_agent`. Custom middleware is appended after [`PatchToolCallsMiddleware`](https://reference.langchain.com/python/deepagents/middleware/patch_tool_calls/PatchToolCallsMiddleware) in the [default stack](#default-stack-main-agent).

By default, Deep Agents have access to the following middleware:

### Default stack (main agent)

From first to last:

1. [`TodoListMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/todo/TodoListMiddleware): Tracks and manages todo lists for organizing agent tasks and work.

2. [`SkillsMiddleware`](https://reference.langchain.com/python/deepagents/middleware/skills/SkillsMiddleware): Only when you pass `skills`. Injected **immediately after** the todo middleware and **before** filesystem middleware so skill metadata is available before file tools run.

3. [`FilesystemMiddleware`](https://reference.langchain.com/python/deepagents/middleware/filesystem/FilesystemMiddleware): Handles file system operations such as reading, writing, and navigating directories. When you pass `permissions`, filesystem permissions enforcement is included here so it can evaluate every tool the agent might call.

4. [`SubAgentMiddleware`](https://reference.langchain.com/python/deepagents/middleware/subagents/SubAgentMiddleware): Spawns and coordinates subagents for delegating tasks to specialized agents.

5. [`SummarizationMiddleware`](https://reference.langchain.com/python/langchain/agents/middleware/summarization/SummarizationMiddleware): Condenses message history to stay within context limits when conversations grow long (via [create\_summarization\_middleware](https://reference.langchain.com/python/deepagents/middleware/summarization/create_summarization_middleware)).

6. [`PatchToolCallsMiddleware`](https://reference.langchain.com/python/deepagents/middleware/patch_tool_calls/PatchToolCallsMiddleware): Repairs dangling tool calls in message history when a run resumes after an interruption or receives malformed tool-call arguments. Runs **before** Anthropic prompt caching and the tail stack below.

7. [`AsyncSubAgentMiddleware`](https://reference.langchain.com/python/deepagents/middleware/async_subagents/AsyncSubAgentMiddleware): Only when you configure async subagents.

8. **Your middleware argument**: Optional middleware you pass as the `middleware` argument is appended here (after Patch, before the tail stack).

9. **Harness profile extras**: Provider-specific middleware from the resolved model profile, if any.

10. **Excluded-tool filtering**: When the harness profile lists excluded tools, middleware removes those tools from the agent.

11. [`AnthropicPromptCachingMiddleware`](https://reference.langchain.com/python/langchain-anthropic/middleware/prompt_caching/AnthropicPromptCachingMiddleware): Always registered; it no-ops on non-Anthropic models (`unsupported_model_behavior="ignore"`). Runs **after** Patch and after your middleware so the cached prefix matches what is actually sent to the model.

12. [`MemoryMiddleware`](https://reference.langchain.com/python/deepagents/middleware/memory/MemoryMiddleware): Only when you pass `memory`.

    <Note>
      `MemoryMiddleware` is placed **after** profile extras and Anthropic prompt caching so updates to injected memory are less likely to invalidate the Anthropic cache prefix. The same ordering concern is called out in the `create_deep_agent` implementation comments.
    </Note>

13. `HumanInTheLoopMiddleware`: Only when you pass `interrupt_on`. Pauses for human approval or input at configured tool calls.

### Default stack (synchronous subagents)

The built-in **general-purpose** subagent and each declarative synchronous `SubAgent` graph use a stack that `create_deep_agent` builds in code. It matches the main agent in broad shape (todo list, filesystem, summarization, Patch, profile extras, Anthropic caching, optional permissions) but differs in two ways:

* **Skills run after** [`PatchToolCallsMiddleware`](https://reference.langchain.com/python/deepagents/middleware/patch_tool_calls/PatchToolCallsMiddleware) on these inner agents (on the main agent, skills run **before** filesystem middleware when `skills` is set).
* There is **no** [`SubAgentMiddleware`](https://reference.langchain.com/python/deepagents/middleware/subagents/SubAgentMiddleware) inside a subagent graph (only the parent agent exposes the `task` tool).

When a declarative subagent sets `interrupt_on`, that value is forwarded to `create_agent` for the subagent, which wires up human-in-the-loop handling for the configured tool calls.

### Prebuilt middleware

LangChain exposes additional prebuilt middleware that let you add-on various features, such as retries, fallbacks, or PII detection. See [Prebuilt middleware](/oss/python/langchain/middleware/built-in) for more.

The `deepagents` library also exposes [`create_summarization_tool_middleware`](https://reference.langchain.com/python/deepagents/middleware/summarization/create_summarization_tool_middleware), enabling agents to trigger summarization at opportune times—such as between tasks—instead of at fixed token intervals. For more detail, see [Summarization](/oss/python/deepagents/context-engineering#summarization).

### Provider-specific middleware

For provider-specific middleware that is optimized for specific LLM providers, see [Official integrations](/oss/python/integrations/middleware#official-integrations) and [Community integrations](/oss/python/integrations/middleware#community-integrations).

### Custom middleware

You can provide additional middleware to extend functionality, add tools, or implement custom hooks:

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents.middleware import wrap_tool_call
  from langchain.tools import tool
  from deepagents import create_deep_agent


  @tool
  def get_weather(city: str) -> str:
      """Get the weather in a city."""
      return f"The weather in {city} is sunny."


  call_count = [0]  # Use list to allow modification in nested function


  @wrap_tool_call
  def log_tool_calls(request, handler):
      """Intercept and log every tool call - demonstrates cross-cutting concern."""
      call_count[0] += 1
      tool_name = request.name if hasattr(request, "name") else str(request)

      print(f"[Middleware] Tool call #{call_count[0]}: {tool_name}")
      print(f"[Middleware] Arguments: {request.args if hasattr(request, 'args') else 'N/A'}")

      # Execute the tool call
      result = handler(request)

      # Log the result
      print(f"[Middleware] Tool call #{call_count[0]} completed")

      return result


  agent = create_deep_agent(
      model="google_genai:gemini-3.5-flash",
      tools=[get_weather],
      middleware=[log_tool_calls],
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents.middleware import wrap_tool_call
  from langchain.tools import tool
  from deepagents import create_deep_agent


  @tool
  def get_weather(city: str) -> str:
      """Get the weather in a city."""
      return f"The weather in {city} is sunny."


  call_count = [0]  # Use list to allow modification in nested function


  @wrap_tool_call
  def log_tool_calls(request, handler):
      """Intercept and log every tool call - demonstrates cross-cutting concern."""
      call_count[0] += 1
      tool_name = request.name if hasattr(request, "name") else str(request)

      print(f"[Middleware] Tool call #{call_count[0]}: {tool_name}")
      print(f"[Middleware] Arguments: {request.args if hasattr(request, 'args') else 'N/A'}")

      # Execute the tool call
      result = handler(request)

      # Log the result
      print(f"[Middleware] Tool call #{call_count[0]} completed")

      return result


  agent = create_deep_agent(
      model="openai:gpt-5.5",
      tools=[get_weather],
      middleware=[log_tool_calls],
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents.middleware import wrap_tool_call
  from langchain.tools import tool
  from deepagents import create_deep_agent


  @tool
  def get_weather(city: str) -> str:
      """Get the weather in a city."""
      return f"The weather in {city} is sunny."


  call_count = [0]  # Use list to allow modification in nested function


  @wrap_tool_call
  def log_tool_calls(request, handler):
      """Intercept and log every tool call - demonstrates cross-cutting concern."""
      call_count[0] += 1
      tool_name = request.name if hasattr(request, "name") else str(request)

      print(f"[Middleware] Tool call #{call_count[0]}: {tool_name}")
      print(f"[Middleware] Arguments: {request.args if hasattr(request, 'args') else 'N/A'}")

      # Execute the tool call
      result = handler(request)

      # Log the result
      print(f"[Middleware] Tool call #{call_count[0]} completed")

      return result


  agent = create_deep_agent(
      model="anthropic:claude-sonnet-4-6",
      tools=[get_weather],
      middleware=[log_tool_calls],
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents.middleware import wrap_tool_call
  from langchain.tools import tool
  from deepagents import create_deep_agent


  @tool
  def get_weather(city: str) -> str:
      """Get the weather in a city."""
      return f"The weather in {city} is sunny."


  call_count = [0]  # Use list to allow modification in nested function


  @wrap_tool_call
  def log_tool_calls(request, handler):
      """Intercept and log every tool call - demonstrates cross-cutting concern."""
      call_count[0] += 1
      tool_name = request.name if hasattr(request, "name") else str(request)

      print(f"[Middleware] Tool call #{call_count[0]}: {tool_name}")
      print(f"[Middleware] Arguments: {request.args if hasattr(request, 'args') else 'N/A'}")

      # Execute the tool call
      result = handler(request)

      # Log the result
      print(f"[Middleware] Tool call #{call_count[0]} completed")

      return result


  agent = create_deep_agent(
      model="openrouter:anthropic/claude-sonnet-4-6",
      tools=[get_weather],
      middleware=[log_tool_calls],
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents.middleware import wrap_tool_call
  from langchain.tools import tool
  from deepagents import create_deep_agent


  @tool
  def get_weather(city: str) -> str:
      """Get the weather in a city."""
      return f"The weather in {city} is sunny."


  call_count = [0]  # Use list to allow modification in nested function


  @wrap_tool_call
  def log_tool_calls(request, handler):
      """Intercept and log every tool call - demonstrates cross-cutting concern."""
      call_count[0] += 1
      tool_name = request.name if hasattr(request, "name") else str(request)

      print(f"[Middleware] Tool call #{call_count[0]}: {tool_name}")
      print(f"[Middleware] Arguments: {request.args if hasattr(request, 'args') else 'N/A'}")

      # Execute the tool call
      result = handler(request)

      # Log the result
      print(f"[Middleware] Tool call #{call_count[0]} completed")

      return result


  agent = create_deep_agent(
      model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
      tools=[get_weather],
      middleware=[log_tool_calls],
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents.middleware import wrap_tool_call
  from langchain.tools import tool
  from deepagents import create_deep_agent


  @tool
  def get_weather(city: str) -> str:
      """Get the weather in a city."""
      return f"The weather in {city} is sunny."


  call_count = [0]  # Use list to allow modification in nested function


  @wrap_tool_call
  def log_tool_calls(request, handler):
      """Intercept and log every tool call - demonstrates cross-cutting concern."""
      call_count[0] += 1
      tool_name = request.name if hasattr(request, "name") else str(request)

      print(f"[Middleware] Tool call #{call_count[0]}: {tool_name}")
      print(f"[Middleware] Arguments: {request.args if hasattr(request, 'args') else 'N/A'}")

      # Execute the tool call
      result = handler(request)

      # Log the result
      print(f"[Middleware] Tool call #{call_count[0]} completed")

      return result


  agent = create_deep_agent(
      model="baseten:zai-org/GLM-5",
      tools=[get_weather],
      middleware=[log_tool_calls],
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents.middleware import wrap_tool_call
  from langchain.tools import tool
  from deepagents import create_deep_agent


  @tool
  def get_weather(city: str) -> str:
      """Get the weather in a city."""
      return f"The weather in {city} is sunny."


  call_count = [0]  # Use list to allow modification in nested function


  @wrap_tool_call
  def log_tool_calls(request, handler):
      """Intercept and log every tool call - demonstrates cross-cutting concern."""
      call_count[0] += 1
      tool_name = request.name if hasattr(request, "name") else str(request)

      print(f"[Middleware] Tool call #{call_count[0]}: {tool_name}")
      print(f"[Middleware] Arguments: {request.args if hasattr(request, 'args') else 'N/A'}")

      # Execute the tool call
      result = handler(request)

      # Log the result
      print(f"[Middleware] Tool call #{call_count[0]} completed")

      return result


  agent = create_deep_agent(
      model="ollama:devstral-2",
      tools=[get_weather],
      middleware=[log_tool_calls],
  )
  ```
</CodeGroup>

<Warning>
  **Do not mutate attributes after initialization**

  If you need to track values across hook invocations (for example, counters or accumulated data), use graph state.
  Graph state is scoped to a thread by design, so updates are safe under concurrency.

  **Do this:**

  ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents.middleware import AgentMiddleware


  class CustomMiddleware(AgentMiddleware):
      def __init__(self):
          pass

      def before_agent(self, state, runtime):
          return {"x": state.get("x", 0) + 1}  # Update graph state instead
  ```

  Do **not** do this:

  ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  class CustomMiddlewareBad(AgentMiddleware):
      def __init__(self):
          self.x = 1

      def before_agent(self, state, runtime):
          self.x += 1  # Mutation causes race conditions
  ```

  Mutation in place, such as modifying `self.x` in `before_agent` or changing other shared values in hooks, can lead to subtle bugs and race conditions because many operations run concurrently (subagents, parallel tools, and parallel invocations on different threads).

  For full details on extending state with custom properties, see [Custom middleware - Custom state schema](/oss/python/langchain/middleware/custom#custom-state-schema).

  If you must use mutation in custom middleware, consider what happens when subagents, parallel tools, or concurrent agent invocations run at the same time.
</Warning>

### Interpreters

Use [interpreters](/oss/python/deepagents/interpreters) to add an `eval` tool that runs JavaScript in a scoped QuickJS runtime. Interpreters are useful when the agent needs to compose tools programmatically, batch work, handle errors in code, or transform structured data without a full shell environment.

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent
  from langchain_quickjs import CodeInterpreterMiddleware

  agent = create_deep_agent(
      model="google_genai:gemini-3.5-flash",
      middleware=[CodeInterpreterMiddleware()],
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent
  from langchain_quickjs import CodeInterpreterMiddleware

  agent = create_deep_agent(
      model="openai:gpt-5.5",
      middleware=[CodeInterpreterMiddleware()],
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent
  from langchain_quickjs import CodeInterpreterMiddleware

  agent = create_deep_agent(
      model="anthropic:claude-sonnet-4-6",
      middleware=[CodeInterpreterMiddleware()],
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent
  from langchain_quickjs import CodeInterpreterMiddleware

  agent = create_deep_agent(
      model="openrouter:anthropic/claude-sonnet-4-6",
      middleware=[CodeInterpreterMiddleware()],
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent
  from langchain_quickjs import CodeInterpreterMiddleware

  agent = create_deep_agent(
      model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
      middleware=[CodeInterpreterMiddleware()],
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent
  from langchain_quickjs import CodeInterpreterMiddleware

  agent = create_deep_agent(
      model="baseten:zai-org/GLM-5",
      middleware=[CodeInterpreterMiddleware()],
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from deepagents import create_deep_agent
  from langchain_quickjs import CodeInterpreterMiddleware

  agent = create_deep_agent(
      model="ollama:devstral-2",
      middleware=[CodeInterpreterMiddleware()],
  )
  ```
</CodeGroup>

For setup, programmatic tool calling, subagent orchestration, and limits, see [Interpreters](/oss/python/deepagents/interpreters).

## Subagents

To isolate detailed work and avoid context bloat, use subagents:

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import os
from typing import Literal

from deepagents import create_deep_agent
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "system_prompt": "You are a great researcher",
    "tools": [internet_search],
    "model": "openai:gpt-5.5",  # Optional override, defaults to main agent model
}
subagents = [research_subagent]

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    subagents=subagents,
)
```

For more information, see [Subagents](/oss/python/deepagents/subagents).

{/* ## Context - You can persist agent state between runs to store information like user IDs. */}

## Backends

Tools for a deep agent can make use of virtual file systems to store, access, and edit files. By default, deep agents use a [`StateBackend`](https://reference.langchain.com/python/deepagents/backends/state/StateBackend).

If you are using [skills](#skills) or [memory](#memory), you must add the expected skill or memory files to the backend before creating the agent.

<Tabs>
  <Tab title="StateBackend">
    A thread-scoped filesystem backend stored in `langgraph` state.

    Files persist across turns within a thread (via your checkpointer) and are not shared across threads.

    <CodeGroup>
      ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend

      # By default we provide a StateBackend
      agent = create_deep_agent(model="google_genai:gemini-3.5-flash")

      # Under the hood, it looks like
      agent2 = create_deep_agent(
          model="openai:gpt-5.5",
          backend=StateBackend(),
      )
      ```

      ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend

      # By default we provide a StateBackend
      agent = create_deep_agent(model="openai:gpt-5.5")

      # Under the hood, it looks like
      agent2 = create_deep_agent(
          model="openai:gpt-5.5",
          backend=StateBackend(),
      )
      ```

      ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend

      # By default we provide a StateBackend
      agent = create_deep_agent(model="anthropic:claude-sonnet-4-6")

      # Under the hood, it looks like
      agent2 = create_deep_agent(
          model="openai:gpt-5.5",
          backend=StateBackend(),
      )
      ```

      ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend

      # By default we provide a StateBackend
      agent = create_deep_agent(model="openrouter:anthropic/claude-sonnet-4-6")

      # Under the hood, it looks like
      agent2 = create_deep_agent(
          model="openai:gpt-5.5",
          backend=StateBackend(),
      )
      ```

      ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend

      # By default we provide a StateBackend
      agent = create_deep_agent(model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b")

      # Under the hood, it looks like
      agent2 = create_deep_agent(
          model="openai:gpt-5.5",
          backend=StateBackend(),
      )
      ```

      ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend

      # By default we provide a StateBackend
      agent = create_deep_agent(model="baseten:zai-org/GLM-5")

      # Under the hood, it looks like
      agent2 = create_deep_agent(
          model="openai:gpt-5.5",
          backend=StateBackend(),
      )
      ```

      ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend

      # By default we provide a StateBackend
      agent = create_deep_agent(model="ollama:devstral-2")

      # Under the hood, it looks like
      agent2 = create_deep_agent(
          model="openai:gpt-5.5",
          backend=StateBackend(),
      )
      ```
    </CodeGroup>
  </Tab>

  <Tab title="FilesystemBackend">
    The local machine's filesystem.

    <Warning>
      This backend grants agents direct filesystem read/write access.
      Use with caution and only in appropriate environments.
      For more information, see [`FilesystemBackend`](/oss/python/deepagents/backends#filesystembackend-local-disk).
    </Warning>

    <CodeGroup>
      ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend

      agent = create_deep_agent(
          model="google_genai:gemini-3.5-flash",
          backend=FilesystemBackend(root_dir=".", virtual_mode=True),
      )
      ```

      ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend

      agent = create_deep_agent(
          model="openai:gpt-5.5",
          backend=FilesystemBackend(root_dir=".", virtual_mode=True),
      )
      ```

      ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend

      agent = create_deep_agent(
          model="anthropic:claude-sonnet-4-6",
          backend=FilesystemBackend(root_dir=".", virtual_mode=True),
      )
      ```

      ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend

      agent = create_deep_agent(
          model="openrouter:anthropic/claude-sonnet-4-6",
          backend=FilesystemBackend(root_dir=".", virtual_mode=True),
      )
      ```

      ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend

      agent = create_deep_agent(
          model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
          backend=FilesystemBackend(root_dir=".", virtual_mode=True),
      )
      ```

      ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend

      agent = create_deep_agent(
          model="baseten:zai-org/GLM-5",
          backend=FilesystemBackend(root_dir=".", virtual_mode=True),
      )
      ```

      ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend

      agent = create_deep_agent(
          model="ollama:devstral-2",
          backend=FilesystemBackend(root_dir=".", virtual_mode=True),
      )
      ```
    </CodeGroup>

    <Tip>
      Wrap `FilesystemBackend` in a `CompositeBackend` to prevent internal agent data (offloaded tool results, conversation history) from being written to disk alongside your project files. See the [recommended pattern](/oss/python/deepagents/backends#filesystembackend-local-disk).
    </Tip>
  </Tab>

  <Tab title="LocalShellBackend">
    A filesystem with shell execution directly on the host. Provides filesystem tools plus the `execute` tool for running commands.

    <Warning>
      This backend grants agents direct filesystem read/write access **and** unrestricted shell execution on your host.
      Use with extreme caution and only in appropriate environments.
      For more information, see [`LocalShellBackend`](/oss/python/deepagents/backends#localshellbackend-local-shell).
    </Warning>

    <CodeGroup>
      ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import LocalShellBackend

      agent = create_deep_agent(
          model="google_genai:gemini-3.5-flash",
          backend=LocalShellBackend(root_dir=".", virtual_mode=True, env={"PATH": "/usr/bin:/bin"}),
      )
      ```

      ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import LocalShellBackend

      agent = create_deep_agent(
          model="openai:gpt-5.5",
          backend=LocalShellBackend(root_dir=".", virtual_mode=True, env={"PATH": "/usr/bin:/bin"}),
      )
      ```

      ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import LocalShellBackend

      agent = create_deep_agent(
          model="anthropic:claude-sonnet-4-6",
          backend=LocalShellBackend(root_dir=".", virtual_mode=True, env={"PATH": "/usr/bin:/bin"}),
      )
      ```

      ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import LocalShellBackend

      agent = create_deep_agent(
          model="openrouter:anthropic/claude-sonnet-4-6",
          backend=LocalShellBackend(root_dir=".", virtual_mode=True, env={"PATH": "/usr/bin:/bin"}),
      )
      ```

      ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import LocalShellBackend

      agent = create_deep_agent(
          model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
          backend=LocalShellBackend(root_dir=".", virtual_mode=True, env={"PATH": "/usr/bin:/bin"}),
      )
      ```

      ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import LocalShellBackend

      agent = create_deep_agent(
          model="baseten:zai-org/GLM-5",
          backend=LocalShellBackend(root_dir=".", virtual_mode=True, env={"PATH": "/usr/bin:/bin"}),
      )
      ```

      ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import LocalShellBackend

      agent = create_deep_agent(
          model="ollama:devstral-2",
          backend=LocalShellBackend(root_dir=".", virtual_mode=True, env={"PATH": "/usr/bin:/bin"}),
      )
      ```
    </CodeGroup>
  </Tab>

  <Tab title="StoreBackend">
    A filesystem that provides long-term storage that is *persisted across threads*.

    <CodeGroup>
      ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="google_genai:gemini-3.5-flash",
          backend=StoreBackend(
              namespace=lambda rt: (rt.server_info.user.identity,),
          ),
          store=InMemoryStore(),  # Good for local dev; omit for LangSmith Deployment
      )
      ```

      ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="openai:gpt-5.5",
          backend=StoreBackend(
              namespace=lambda rt: (rt.server_info.user.identity,),
          ),
          store=InMemoryStore(),  # Good for local dev; omit for LangSmith Deployment
      )
      ```

      ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="anthropic:claude-sonnet-4-6",
          backend=StoreBackend(
              namespace=lambda rt: (rt.server_info.user.identity,),
          ),
          store=InMemoryStore(),  # Good for local dev; omit for LangSmith Deployment
      )
      ```

      ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="openrouter:anthropic/claude-sonnet-4-6",
          backend=StoreBackend(
              namespace=lambda rt: (rt.server_info.user.identity,),
          ),
          store=InMemoryStore(),  # Good for local dev; omit for LangSmith Deployment
      )
      ```

      ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
          backend=StoreBackend(
              namespace=lambda rt: (rt.server_info.user.identity,),
          ),
          store=InMemoryStore(),  # Good for local dev; omit for LangSmith Deployment
      )
      ```

      ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="baseten:zai-org/GLM-5",
          backend=StoreBackend(
              namespace=lambda rt: (rt.server_info.user.identity,),
          ),
          store=InMemoryStore(),  # Good for local dev; omit for LangSmith Deployment
      )
      ```

      ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="ollama:devstral-2",
          backend=StoreBackend(
              namespace=lambda rt: (rt.server_info.user.identity,),
          ),
          store=InMemoryStore(),  # Good for local dev; omit for LangSmith Deployment
      )
      ```
    </CodeGroup>

    <Note>
      When deploying to [LangSmith Deployment](/langsmith/deployment), omit the `store` parameter. The platform automatically provisions a store for your agent.
    </Note>

    <Tip>
      The `namespace` parameter controls data isolation. For multi-user deployments, always set a [namespace factory](/oss/python/deepagents/backends#namespace-factories) to isolate data per user or tenant.
    </Tip>
  </Tab>

  <Tab title="ContextHubBackend">
    Durable filesystem storage in a LangSmith Hub repo.

    <CodeGroup>
      ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import ContextHubBackend

      agent = create_deep_agent(
          model="google_genai:gemini-3.5-flash",
          backend=ContextHubBackend("my-agent"),
      )
      ```

      ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import ContextHubBackend

      agent = create_deep_agent(
          model="openai:gpt-5.5",
          backend=ContextHubBackend("my-agent"),
      )
      ```

      ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import ContextHubBackend

      agent = create_deep_agent(
          model="anthropic:claude-sonnet-4-6",
          backend=ContextHubBackend("my-agent"),
      )
      ```

      ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import ContextHubBackend

      agent = create_deep_agent(
          model="openrouter:anthropic/claude-sonnet-4-6",
          backend=ContextHubBackend("my-agent"),
      )
      ```

      ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import ContextHubBackend

      agent = create_deep_agent(
          model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
          backend=ContextHubBackend("my-agent"),
      )
      ```

      ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import ContextHubBackend

      agent = create_deep_agent(
          model="baseten:zai-org/GLM-5",
          backend=ContextHubBackend("my-agent"),
      )
      ```

      ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import ContextHubBackend

      agent = create_deep_agent(
          model="ollama:devstral-2",
          backend=ContextHubBackend("my-agent"),
      )
      ```
    </CodeGroup>

    For more details, see [`ContextHubBackend`](/oss/python/deepagents/backends#contexthubbackend).
  </Tab>

  <Tab title="CompositeBackend">
    A flexible backend where you can specify different routes in the filesystem to point towards different backends.

    <CodeGroup>
      ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="google_genai:gemini-3.5-flash",
          backend=CompositeBackend(
              default=StateBackend(),
              routes={
                  "/memories/": StoreBackend(namespace=lambda _rt: ("memories",)),
              },
          ),
          store=InMemoryStore(),  # Store passed to create_deep_agent, not backend
      )
      ```

      ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="openai:gpt-5.5",
          backend=CompositeBackend(
              default=StateBackend(),
              routes={
                  "/memories/": StoreBackend(namespace=lambda _rt: ("memories",)),
              },
          ),
          store=InMemoryStore(),  # Store passed to create_deep_agent, not backend
      )
      ```

      ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="anthropic:claude-sonnet-4-6",
          backend=CompositeBackend(
              default=StateBackend(),
              routes={
                  "/memories/": StoreBackend(namespace=lambda _rt: ("memories",)),
              },
          ),
          store=InMemoryStore(),  # Store passed to create_deep_agent, not backend
      )
      ```

      ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="openrouter:anthropic/claude-sonnet-4-6",
          backend=CompositeBackend(
              default=StateBackend(),
              routes={
                  "/memories/": StoreBackend(namespace=lambda _rt: ("memories",)),
              },
          ),
          store=InMemoryStore(),  # Store passed to create_deep_agent, not backend
      )
      ```

      ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
          backend=CompositeBackend(
              default=StateBackend(),
              routes={
                  "/memories/": StoreBackend(namespace=lambda _rt: ("memories",)),
              },
          ),
          store=InMemoryStore(),  # Store passed to create_deep_agent, not backend
      )
      ```

      ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="baseten:zai-org/GLM-5",
          backend=CompositeBackend(
              default=StateBackend(),
              routes={
                  "/memories/": StoreBackend(namespace=lambda _rt: ("memories",)),
              },
          ),
          store=InMemoryStore(),  # Store passed to create_deep_agent, not backend
      )
      ```

      ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
      from langgraph.store.memory import InMemoryStore

      agent = create_deep_agent(
          model="ollama:devstral-2",
          backend=CompositeBackend(
              default=StateBackend(),
              routes={
                  "/memories/": StoreBackend(namespace=lambda _rt: ("memories",)),
              },
          ),
          store=InMemoryStore(),  # Store passed to create_deep_agent, not backend
      )
      ```
    </CodeGroup>
  </Tab>
</Tabs>

For more information, see [Backends](/oss/python/deepagents/backends).

### Sandboxes

Sandboxes are specialized [backends](/oss/python/deepagents/backends) that run agent code in an isolated environment with their own filesystem and an `execute` tool for shell commands.
Use a sandbox backend when you want your deep agent to write files, install dependencies, and run commands without changing anything on your local machine.

You configure sandboxes by passing a sandbox backend to `backend` when creating your deep agent:

<Tabs>
  <Tab title="LangSmith">
    <CodeGroup>
      ```bash pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      pip install "langsmith[sandbox]"
      ```

      ```bash uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      uv add "langsmith[sandbox]"
      ```
    </CodeGroup>

    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    from deepagents import create_deep_agent
    from deepagents.backends import LangSmithSandbox
    from langchain_anthropic import ChatAnthropic
    from langsmith.sandbox import SandboxClient

    client = SandboxClient()
    ls_sandbox = client.create_sandbox()
    backend = LangSmithSandbox(sandbox=ls_sandbox)

    agent = create_deep_agent(
        model=ChatAnthropic(model="claude-sonnet-4-6"),
        system_prompt="You are a Python coding assistant with sandbox access.",
        backend=backend,
    )
    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Create a small Python package and run pytest",
                    }
                ]
            }
        )
    finally:
        client.delete_sandbox(ls_sandbox.name)
    ```
  </Tab>

  <Tab title="Daytona">
    <CodeGroup>
      ```bash pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      pip install langchain-daytona
      ```

      ```bash uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      uv add langchain-daytona
      ```
    </CodeGroup>

    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    from daytona import Daytona
    from deepagents import create_deep_agent
    from langchain_anthropic import ChatAnthropic
    from langchain_daytona import DaytonaSandbox

    sandbox = Daytona().create()
    backend = DaytonaSandbox(sandbox=sandbox)

    agent = create_deep_agent(
        model=ChatAnthropic(model="claude-sonnet-4-6"),
        system_prompt="You are a Python coding assistant with sandbox access.",
        backend=backend,
    )

    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Create a small Python package and run pytest",
                    }
                ]
            }
        )
    finally:
        sandbox.stop()
    ```
  </Tab>

  <Tab title="E2B">
    <CodeGroup>
      ```bash pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      pip install langchain-e2b
      ```

      ```bash uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      uv add langchain-e2b
      ```
    </CodeGroup>

    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    from e2b import Sandbox
    from deepagents import create_deep_agent
    from langchain_anthropic import ChatAnthropic
    from langchain_e2b import E2BSandbox

    e2b_sandbox = Sandbox.create()
    backend = E2BSandbox(sandbox=e2b_sandbox)

    agent = create_deep_agent(
        model=ChatAnthropic(model="claude-sonnet-4-6"),
        system_prompt="You are a Python coding assistant with sandbox access.",
        backend=backend,
    )

    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Create a small Python package and run pytest",
                    }
                ]
            }
        )
    finally:
        e2b_sandbox.kill()
    ```
  </Tab>

  <Tab title="Modal">
    <CodeGroup>
      ```bash pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      pip install langchain-modal
      ```

      ```bash uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      uv add langchain-modal
      ```
    </CodeGroup>

    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    import modal
    from deepagents import create_deep_agent
    from langchain_anthropic import ChatAnthropic
    from langchain_modal import ModalSandbox

    app = modal.App.lookup("your-app")
    modal_sandbox = modal.Sandbox.create(app=app)
    backend = ModalSandbox(sandbox=modal_sandbox)

    agent = create_deep_agent(
        model=ChatAnthropic(model="claude-sonnet-4-6"),
        system_prompt="You are a Python coding assistant with sandbox access.",
        backend=backend,
    )
    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Create a small Python package and run pytest",
                    }
                ]
            }
        )
    finally:
        modal_sandbox.terminate()
    ```
  </Tab>

  <Tab title="Runloop">
    <CodeGroup>
      ```bash pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      pip install langchain-runloop
      ```

      ```bash uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      uv add langchain-runloop
      ```
    </CodeGroup>

    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    import os

    from deepagents import create_deep_agent
    from langchain_anthropic import ChatAnthropic
    from langchain_runloop import RunloopSandbox
    from runloop_api_client import RunloopSDK

    client = RunloopSDK(bearer_token=os.environ["RUNLOOP_API_KEY"])

    devbox = client.devbox.create()
    backend = RunloopSandbox(devbox=devbox)

    agent = create_deep_agent(
        model=ChatAnthropic(model="claude-sonnet-4-6"),
        system_prompt="You are a Python coding assistant with sandbox access.",
        backend=backend,
    )

    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Create a small Python package and run pytest",
                    }
                ]
            }
        )
    finally:
        devbox.shutdown()
    ```
  </Tab>

  <Tab title="Vercel">
    <CodeGroup>
      ```bash pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      pip install langchain-vercel-sandbox
      ```

      ```bash uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      uv add langchain-vercel-sandbox
      ```
    </CodeGroup>

    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    from deepagents import create_deep_agent
    from langchain_anthropic import ChatAnthropic
    from langchain_vercel_sandbox import VercelSandbox
    from vercel.sandbox import Sandbox

    sandbox = Sandbox.create()
    backend = VercelSandbox(sandbox=sandbox)

    agent = create_deep_agent(
        model=ChatAnthropic(model="claude-sonnet-4-6"),
        system_prompt="You are a Python coding assistant with sandbox access.",
        backend=backend,
    )

    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Create a small Python package and run pytest",
                    }
                ]
            }
        )
    finally:
        sandbox.stop()
    ```
  </Tab>
</Tabs>

For more information, see [Sandboxes](/oss/python/deepagents/sandboxes).

## Human-in-the-loop

Some tool operations may be sensitive and require human approval before execution.
You can configure the approval for each tool:

<CodeGroup>
  ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.tools import tool
  from deepagents import create_deep_agent
  from langgraph.checkpoint.memory import MemorySaver


  @tool
  def remove_file(path: str) -> str:
      """Delete a file from the filesystem."""
      return f"Deleted {path}"


  @tool
  def fetch_file(path: str) -> str:
      """Read a file from the filesystem."""
      return f"Contents of {path}"


  @tool
  def notify_email(to: str, subject: str, body: str) -> str:
      """Send an email."""
      return f"Sent email to {to}"


  # Checkpointer is REQUIRED for human-in-the-loop
  checkpointer = MemorySaver()

  agent = create_deep_agent(
      model="google_genai:gemini-3.5-flash",
      tools=[remove_file, fetch_file, notify_email],
      interrupt_on={
          "remove_file": True,  # Default: approve, edit, reject, respond
          "fetch_file": False,  # No interrupts needed
          "notify_email": {"allowed_decisions": ["approve", "reject"]},  # No editing
      },
      checkpointer=checkpointer,  # Required!
  )
  ```

  ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.tools import tool
  from deepagents import create_deep_agent
  from langgraph.checkpoint.memory import MemorySaver


  @tool
  def remove_file(path: str) -> str:
      """Delete a file from the filesystem."""
      return f"Deleted {path}"


  @tool
  def fetch_file(path: str) -> str:
      """Read a file from the filesystem."""
      return f"Contents of {path}"


  @tool
  def notify_email(to: str, subject: str, body: str) -> str:
      """Send an email."""
      return f"Sent email to {to}"


  # Checkpointer is REQUIRED for human-in-the-loop
  checkpointer = MemorySaver()

  agent = create_deep_agent(
      model="openai:gpt-5.5",
      tools=[remove_file, fetch_file, notify_email],
      interrupt_on={
          "remove_file": True,  # Default: approve, edit, reject, respond
          "fetch_file": False,  # No interrupts needed
          "notify_email": {"allowed_decisions": ["approve", "reject"]},  # No editing
      },
      checkpointer=checkpointer,  # Required!
  )
  ```

  ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.tools import tool
  from deepagents import create_deep_agent
  from langgraph.checkpoint.memory import MemorySaver


  @tool
  def remove_file(path: str) -> str:
      """Delete a file from the filesystem."""
      return f"Deleted {path}"


  @tool
  def fetch_file(path: str) -> str:
      """Read a file from the filesystem."""
      return f"Contents of {path}"


  @tool
  def notify_email(to: str, subject: str, body: str) -> str:
      """Send an email."""
      return f"Sent email to {to}"


  # Checkpointer is REQUIRED for human-in-the-loop
  checkpointer = MemorySaver()

  agent = create_deep_agent(
      model="anthropic:claude-sonnet-4-6",
      tools=[remove_file, fetch_file, notify_email],
      interrupt_on={
          "remove_file": True,  # Default: approve, edit, reject, respond
          "fetch_file": False,  # No interrupts needed
          "notify_email": {"allowed_decisions": ["approve", "reject"]},  # No editing
      },
      checkpointer=checkpointer,  # Required!
  )
  ```

  ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.tools import tool
  from deepagents import create_deep_agent
  from langgraph.checkpoint.memory import MemorySaver


  @tool
  def remove_file(path: str) -> str:
      """Delete a file from the filesystem."""
      return f"Deleted {path}"


  @tool
  def fetch_file(path: str) -> str:
      """Read a file from the filesystem."""
      return f"Contents of {path}"


  @tool
  def notify_email(to: str, subject: str, body: str) -> str:
      """Send an email."""
      return f"Sent email to {to}"


  # Checkpointer is REQUIRED for human-in-the-loop
  checkpointer = MemorySaver()

  agent = create_deep_agent(
      model="openrouter:anthropic/claude-sonnet-4-6",
      tools=[remove_file, fetch_file, notify_email],
      interrupt_on={
          "remove_file": True,  # Default: approve, edit, reject, respond
          "fetch_file": False,  # No interrupts needed
          "notify_email": {"allowed_decisions": ["approve", "reject"]},  # No editing
      },
      checkpointer=checkpointer,  # Required!
  )
  ```

  ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.tools import tool
  from deepagents import create_deep_agent
  from langgraph.checkpoint.memory import MemorySaver


  @tool
  def remove_file(path: str) -> str:
      """Delete a file from the filesystem."""
      return f"Deleted {path}"


  @tool
  def fetch_file(path: str) -> str:
      """Read a file from the filesystem."""
      return f"Contents of {path}"


  @tool
  def notify_email(to: str, subject: str, body: str) -> str:
      """Send an email."""
      return f"Sent email to {to}"


  # Checkpointer is REQUIRED for human-in-the-loop
  checkpointer = MemorySaver()

  agent = create_deep_agent(
      model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
      tools=[remove_file, fetch_file, notify_email],
      interrupt_on={
          "remove_file": True,  # Default: approve, edit, reject, respond
          "fetch_file": False,  # No interrupts needed
          "notify_email": {"allowed_decisions": ["approve", "reject"]},  # No editing
      },
      checkpointer=checkpointer,  # Required!
  )
  ```

  ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.tools import tool
  from deepagents import create_deep_agent
  from langgraph.checkpoint.memory import MemorySaver


  @tool
  def remove_file(path: str) -> str:
      """Delete a file from the filesystem."""
      return f"Deleted {path}"


  @tool
  def fetch_file(path: str) -> str:
      """Read a file from the filesystem."""
      return f"Contents of {path}"


  @tool
  def notify_email(to: str, subject: str, body: str) -> str:
      """Send an email."""
      return f"Sent email to {to}"


  # Checkpointer is REQUIRED for human-in-the-loop
  checkpointer = MemorySaver()

  agent = create_deep_agent(
      model="baseten:zai-org/GLM-5",
      tools=[remove_file, fetch_file, notify_email],
      interrupt_on={
          "remove_file": True,  # Default: approve, edit, reject, respond
          "fetch_file": False,  # No interrupts needed
          "notify_email": {"allowed_decisions": ["approve", "reject"]},  # No editing
      },
      checkpointer=checkpointer,  # Required!
  )
  ```

  ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.tools import tool
  from deepagents import create_deep_agent
  from langgraph.checkpoint.memory import MemorySaver


  @tool
  def remove_file(path: str) -> str:
      """Delete a file from the filesystem."""
      return f"Deleted {path}"


  @tool
  def fetch_file(path: str) -> str:
      """Read a file from the filesystem."""
      return f"Contents of {path}"


  @tool
  def notify_email(to: str, subject: str, body: str) -> str:
      """Send an email."""
      return f"Sent email to {to}"


  # Checkpointer is REQUIRED for human-in-the-loop
  checkpointer = MemorySaver()

  agent = create_deep_agent(
      model="ollama:devstral-2",
      tools=[remove_file, fetch_file, notify_email],
      interrupt_on={
          "remove_file": True,  # Default: approve, edit, reject, respond
          "fetch_file": False,  # No interrupts needed
          "notify_email": {"allowed_decisions": ["approve", "reject"]},  # No editing
      },
      checkpointer=checkpointer,  # Required!
  )
  ```
</CodeGroup>

You can configure interrupt for agents and subagents on tool call as well as from within tool calls.
For more information, see [Human-in-the-loop](/oss/python/deepagents/human-in-the-loop).

## Skills

You can use [skills](/oss/python/deepagents/overview) to provide your deep agent with new capabilities and expertise.
While [tools](/oss/python/deepagents/customization#tools) tend to cover lower level functionality like native file system actions or planning, skills can contain detailed instructions on how to complete tasks, reference info, and other assets, such as templates.
These files are only loaded by the agent when the agent has determined that the skill is useful for the current prompt.
This progressive disclosure reduces the amount of tokens and context the agent has to consider upon startup.

For example skills, see [Deep Agents example skills](https://github.com/langchain-ai/deepagentsjs/tree/main/examples/skills).

To add skills to your deep agent, pass them as an argument to `create_deep_agent`:

<Tabs>
  <Tab title="StateBackend">
    <CodeGroup>
      ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      checkpointer = MemorySaver()
      backend = StateBackend()

      skill_url = "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/libs/cli/examples/skills/langgraph-docs/SKILL.md"
      with urlopen(skill_url) as response:
          skill_content = response.read().decode('utf-8')

      skills_files = {
          "/skills/langgraph-docs/SKILL.md": create_file_data(skill_content),
      }

      agent = create_deep_agent(
          model="google_genai:gemini-3.5-flash",
          backend=backend,
          skills=["/skills/"],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [{"role": "user", "content": "What is langgraph?"}],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": skills_files,
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      checkpointer = MemorySaver()
      backend = StateBackend()

      skill_url = "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/libs/cli/examples/skills/langgraph-docs/SKILL.md"
      with urlopen(skill_url) as response:
          skill_content = response.read().decode('utf-8')

      skills_files = {
          "/skills/langgraph-docs/SKILL.md": create_file_data(skill_content),
      }

      agent = create_deep_agent(
          model="openai:gpt-5.5",
          backend=backend,
          skills=["/skills/"],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [{"role": "user", "content": "What is langgraph?"}],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": skills_files,
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      checkpointer = MemorySaver()
      backend = StateBackend()

      skill_url = "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/libs/cli/examples/skills/langgraph-docs/SKILL.md"
      with urlopen(skill_url) as response:
          skill_content = response.read().decode('utf-8')

      skills_files = {
          "/skills/langgraph-docs/SKILL.md": create_file_data(skill_content),
      }

      agent = create_deep_agent(
          model="anthropic:claude-sonnet-4-6",
          backend=backend,
          skills=["/skills/"],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [{"role": "user", "content": "What is langgraph?"}],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": skills_files,
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      checkpointer = MemorySaver()
      backend = StateBackend()

      skill_url = "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/libs/cli/examples/skills/langgraph-docs/SKILL.md"
      with urlopen(skill_url) as response:
          skill_content = response.read().decode('utf-8')

      skills_files = {
          "/skills/langgraph-docs/SKILL.md": create_file_data(skill_content),
      }

      agent = create_deep_agent(
          model="openrouter:anthropic/claude-sonnet-4-6",
          backend=backend,
          skills=["/skills/"],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [{"role": "user", "content": "What is langgraph?"}],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": skills_files,
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      checkpointer = MemorySaver()
      backend = StateBackend()

      skill_url = "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/libs/cli/examples/skills/langgraph-docs/SKILL.md"
      with urlopen(skill_url) as response:
          skill_content = response.read().decode('utf-8')

      skills_files = {
          "/skills/langgraph-docs/SKILL.md": create_file_data(skill_content),
      }

      agent = create_deep_agent(
          model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
          backend=backend,
          skills=["/skills/"],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [{"role": "user", "content": "What is langgraph?"}],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": skills_files,
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      checkpointer = MemorySaver()
      backend = StateBackend()

      skill_url = "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/libs/cli/examples/skills/langgraph-docs/SKILL.md"
      with urlopen(skill_url) as response:
          skill_content = response.read().decode('utf-8')

      skills_files = {
          "/skills/langgraph-docs/SKILL.md": create_file_data(skill_content),
      }

      agent = create_deep_agent(
          model="baseten:zai-org/GLM-5",
          backend=backend,
          skills=["/skills/"],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [{"role": "user", "content": "What is langgraph?"}],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": skills_files,
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen
      from deepagents import create_deep_agent
      from deepagents.backends import StateBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      checkpointer = MemorySaver()
      backend = StateBackend()

      skill_url = "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/libs/cli/examples/skills/langgraph-docs/SKILL.md"
      with urlopen(skill_url) as response:
          skill_content = response.read().decode('utf-8')

      skills_files = {
          "/skills/langgraph-docs/SKILL.md": create_file_data(skill_content),
      }

      agent = create_deep_agent(
          model="ollama:devstral-2",
          backend=backend,
          skills=["/skills/"],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [{"role": "user", "content": "What is langgraph?"}],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": skills_files,
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```
    </CodeGroup>
  </Tab>

  <Tab title="StoreBackend">
    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    from urllib.request import urlopen
    from deepagents import create_deep_agent
    from deepagents.backends import StoreBackend
    from deepagents.backends.utils import create_file_data
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    backend = StoreBackend(namespace=lambda _rt: ("filesystem",))

    skill_url = "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/libs/cli/examples/skills/langgraph-docs/SKILL.md"
    with urlopen(skill_url) as response:
        skill_content = response.read().decode('utf-8')

    store.put(
        namespace=("filesystem",),
        key="/skills/langgraph-docs/SKILL.md",
        value=create_file_data(skill_content),
    )

    agent = create_deep_agent(
        model="google_genai:gemini-3.5-flash",
        backend=backend,
        store=store,
        skills=["/skills/"],
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is langgraph?"}]},
        config={"configurable": {"thread_id": "12345"}},
    )
    ```
  </Tab>

  <Tab title="FilesystemBackend">
    ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
    from deepagents import create_deep_agent
    from deepagents.backends.filesystem import FilesystemBackend
    from langgraph.checkpoint.memory import MemorySaver

    # Checkpointer is REQUIRED for human-in-the-loop
    checkpointer = MemorySaver()
    root_dir = "/Users/user/{project}"
    backend = FilesystemBackend(root_dir=root_dir)

    agent = create_deep_agent(
        model="google_genai:gemini-3.5-flash",
        backend=backend,
        skills=[str(Path(root_dir) / "skills")],
        interrupt_on={
            "write_file": True,
            "read_file": False,
            "edit_file": True,
        },
        checkpointer=checkpointer, # Required!
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is langgraph?"}]},
        config={"configurable": {"thread_id": "12345"}},
    )
    ```
  </Tab>
</Tabs>

## Memory

Use [`AGENTS.md` files](https://agents.md/) to provide extra context to your deep agent.

You can pass one or more file paths to the `memory` parameter when creating your deep agent:

<Tabs>
  <Tab title="StateBackend">
    <CodeGroup>
      ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="google_genai:gemini-3.5-flash",
          memory=[
              "/AGENTS.md"
          ],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "123456"}},
      )
      ```

      ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="openai:gpt-5.5",
          memory=[
              "/AGENTS.md"
          ],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "123456"}},
      )
      ```

      ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="anthropic:claude-sonnet-4-6",
          memory=[
              "/AGENTS.md"
          ],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "123456"}},
      )
      ```

      ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="openrouter:anthropic/claude-sonnet-4-6",
          memory=[
              "/AGENTS.md"
          ],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "123456"}},
      )
      ```

      ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
          memory=[
              "/AGENTS.md"
          ],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "123456"}},
      )
      ```

      ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="baseten:zai-org/GLM-5",
          memory=[
              "/AGENTS.md"
          ],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "123456"}},
      )
      ```

      ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends.utils import create_file_data
      from langgraph.checkpoint.memory import MemorySaver

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="ollama:devstral-2",
          memory=[
              "/AGENTS.md"
          ],
          checkpointer=checkpointer,
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "123456"}},
      )
      ```
    </CodeGroup>
  </Tab>

  <Tab title="StoreBackend">
    <CodeGroup>
      ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.store.memory import InMemoryStore

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")

      # Create the store and add the file to it
      store = InMemoryStore()
      file_data = create_file_data(agents_md)
      store.put(
          namespace=("filesystem",),
          key="/AGENTS.md",
          value=file_data,
      )

      agent = create_deep_agent(
          model="google_genai:gemini-3.5-flash",
          backend=StoreBackend(namespace=lambda _rt: ("filesystem",)),
          store=store,
          memory=["/AGENTS.md"],
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.store.memory import InMemoryStore

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")

      # Create the store and add the file to it
      store = InMemoryStore()
      file_data = create_file_data(agents_md)
      store.put(
          namespace=("filesystem",),
          key="/AGENTS.md",
          value=file_data,
      )

      agent = create_deep_agent(
          model="openai:gpt-5.5",
          backend=StoreBackend(namespace=lambda _rt: ("filesystem",)),
          store=store,
          memory=["/AGENTS.md"],
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.store.memory import InMemoryStore

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")

      # Create the store and add the file to it
      store = InMemoryStore()
      file_data = create_file_data(agents_md)
      store.put(
          namespace=("filesystem",),
          key="/AGENTS.md",
          value=file_data,
      )

      agent = create_deep_agent(
          model="anthropic:claude-sonnet-4-6",
          backend=StoreBackend(namespace=lambda _rt: ("filesystem",)),
          store=store,
          memory=["/AGENTS.md"],
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.store.memory import InMemoryStore

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")

      # Create the store and add the file to it
      store = InMemoryStore()
      file_data = create_file_data(agents_md)
      store.put(
          namespace=("filesystem",),
          key="/AGENTS.md",
          value=file_data,
      )

      agent = create_deep_agent(
          model="openrouter:anthropic/claude-sonnet-4-6",
          backend=StoreBackend(namespace=lambda _rt: ("filesystem",)),
          store=store,
          memory=["/AGENTS.md"],
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.store.memory import InMemoryStore

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")

      # Create the store and add the file to it
      store = InMemoryStore()
      file_data = create_file_data(agents_md)
      store.put(
          namespace=("filesystem",),
          key="/AGENTS.md",
          value=file_data,
      )

      agent = create_deep_agent(
          model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
          backend=StoreBackend(namespace=lambda _rt: ("filesystem",)),
          store=store,
          memory=["/AGENTS.md"],
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.store.memory import InMemoryStore

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")

      # Create the store and add the file to it
      store = InMemoryStore()
      file_data = create_file_data(agents_md)
      store.put(
          namespace=("filesystem",),
          key="/AGENTS.md",
          value=file_data,
      )

      agent = create_deep_agent(
          model="baseten:zai-org/GLM-5",
          backend=StoreBackend(namespace=lambda _rt: ("filesystem",)),
          store=store,
          memory=["/AGENTS.md"],
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from urllib.request import urlopen

      from deepagents import create_deep_agent
      from deepagents.backends import StoreBackend
      from deepagents.backends.utils import create_file_data
      from langgraph.store.memory import InMemoryStore

      with urlopen(
          "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md"
      ) as response:
          agents_md = response.read().decode("utf-8")

      # Create the store and add the file to it
      store = InMemoryStore()
      file_data = create_file_data(agents_md)
      store.put(
          namespace=("filesystem",),
          key="/AGENTS.md",
          value=file_data,
      )

      agent = create_deep_agent(
          model="ollama:devstral-2",
          backend=StoreBackend(namespace=lambda _rt: ("filesystem",)),
          store=store,
          memory=["/AGENTS.md"],
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
              "files": {"/AGENTS.md": create_file_data(agents_md)},
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```
    </CodeGroup>
  </Tab>

  <Tab title="FilesystemBackend">
    <CodeGroup>
      ```python Google theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend
      from langgraph.checkpoint.memory import MemorySaver

      # Checkpointer is REQUIRED for human-in-the-loop
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="google_genai:gemini-3.5-flash",
          backend=FilesystemBackend(root_dir="/Users/user/{project}"),
          memory=[
              "./AGENTS.md"
          ],
          interrupt_on={
              "write_file": True,  # Default: approve, edit, reject
              "read_file": False,  # No interrupts needed
              "edit_file": True,   # Default: approve, edit, reject
          },
          checkpointer=checkpointer,  # Required!
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python OpenAI theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend
      from langgraph.checkpoint.memory import MemorySaver

      # Checkpointer is REQUIRED for human-in-the-loop
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="openai:gpt-5.5",
          backend=FilesystemBackend(root_dir="/Users/user/{project}"),
          memory=[
              "./AGENTS.md"
          ],
          interrupt_on={
              "write_file": True,  # Default: approve, edit, reject
              "read_file": False,  # No interrupts needed
              "edit_file": True,   # Default: approve, edit, reject
          },
          checkpointer=checkpointer,  # Required!
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python Anthropic theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend
      from langgraph.checkpoint.memory import MemorySaver

      # Checkpointer is REQUIRED for human-in-the-loop
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="anthropic:claude-sonnet-4-6",
          backend=FilesystemBackend(root_dir="/Users/user/{project}"),
          memory=[
              "./AGENTS.md"
          ],
          interrupt_on={
              "write_file": True,  # Default: approve, edit, reject
              "read_file": False,  # No interrupts needed
              "edit_file": True,   # Default: approve, edit, reject
          },
          checkpointer=checkpointer,  # Required!
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python OpenRouter theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend
      from langgraph.checkpoint.memory import MemorySaver

      # Checkpointer is REQUIRED for human-in-the-loop
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="openrouter:anthropic/claude-sonnet-4-6",
          backend=FilesystemBackend(root_dir="/Users/user/{project}"),
          memory=[
              "./AGENTS.md"
          ],
          interrupt_on={
              "write_file": True,  # Default: approve, edit, reject
              "read_file": False,  # No interrupts needed
              "edit_file": True,   # Default: approve, edit, reject
          },
          checkpointer=checkpointer,  # Required!
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python Fireworks theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend
      from langgraph.checkpoint.memory import MemorySaver

      # Checkpointer is REQUIRED for human-in-the-loop
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="fireworks:accounts/fireworks/models/qwen3p5-397b-a17b",
          backend=FilesystemBackend(root_dir="/Users/user/{project}"),
          memory=[
              "./AGENTS.md"
          ],
          interrupt_on={
              "write_file": True,  # Default: approve, edit, reject
              "read_file": False,  # No interrupts needed
              "edit_file": True,   # Default: approve, edit, reject
          },
          checkpointer=checkpointer,  # Required!
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python Baseten theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend
      from langgraph.checkpoint.memory import MemorySaver

      # Checkpointer is REQUIRED for human-in-the-loop
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="baseten:zai-org/GLM-5",
          backend=FilesystemBackend(root_dir="/Users/user/{project}"),
          memory=[
              "./AGENTS.md"
          ],
          interrupt_on={
              "write_file": True,  # Default: approve, edit, reject
              "read_file": False,  # No interrupts needed
              "edit_file": True,   # Default: approve, edit, reject
          },
          checkpointer=checkpointer,  # Required!
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```

      ```python Ollama theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
      from deepagents import create_deep_agent
      from deepagents.backends import FilesystemBackend
      from langgraph.checkpoint.memory import MemorySaver

      # Checkpointer is REQUIRED for human-in-the-loop
      checkpointer = MemorySaver()

      agent = create_deep_agent(
          model="ollama:devstral-2",
          backend=FilesystemBackend(root_dir="/Users/user/{project}"),
          memory=[
              "./AGENTS.md"
          ],
          interrupt_on={
              "write_file": True,  # Default: approve, edit, reject
              "read_file": False,  # No interrupts needed
              "edit_file": True,   # Default: approve, edit, reject
          },
          checkpointer=checkpointer,  # Required!
      )

      result = agent.invoke(
          {
              "messages": [
                  {
                      "role": "user",
                      "content": "Please tell me what's in your memory files.",
                  }
              ],
          },
          config={"configurable": {"thread_id": "12345"}},
      )
      ```
    </CodeGroup>
  </Tab>
</Tabs>

## Profiles

A [harness profile](/oss/python/deepagents/profiles#harness-profiles) is a reusable bundle of per-model configuration that `create_deep_agent` applies automatically when the matching model is selected. Profiles are the right tool when you want behaviour that follows the model—not the call site—such as a system prompt suffix tuned for Claude's instruction style, tool descriptions rewritten for GPT, or extra middleware that only makes sense with a specific provider.

A single profile can carry: a custom base system prompt (`base_system_prompt`), an appended suffix (`system_prompt_suffix`), tool description overrides, tools or middleware to exclude, additional middleware to inject, and edits to the auto-added general-purpose subagent.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import HarnessProfile, register_harness_profile

# Append a system-prompt suffix whenever gpt-5.5 is selected.
register_harness_profile(
    "openai:gpt-5.5",
    HarnessProfile(system_prompt_suffix="Respond in under 100 words."),
)
```

See [Profiles](/oss/python/deepagents/profiles) for registration keys, merge semantics, and plugin packaging. A narrower companion API, [provider profiles](/oss/python/deepagents/profiles#provider-profiles), packages model-construction arguments (API keys, timeouts, retry settings) for a provider.

## Structured output

Deep Agents support [structured output](/oss/python/langchain/structured-output).
You can set a desired structured output schema by passing it as the `response_format` argument to the call to `create_deep_agent()`.
When the model generates the structured data, it's captured, validated, and returned in the 'structured\_response' key of the deep agent's state.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import os
from typing import Literal

from pydantic import BaseModel, Field
from tavily import TavilyClient

from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


class WeatherReport(BaseModel):
    """A structured weather report with current conditions and forecast."""
    location: str = Field(description="The location for this weather report")
    temperature: float = Field(description="Current temperature in Celsius")
    condition: str = Field(
        description="Current weather condition (e.g., sunny, cloudy, rainy)"
    )
    humidity: int = Field(description="Humidity percentage")
    wind_speed: float = Field(description="Wind speed in km/h")
    forecast: str = Field(description="Brief forecast for the next 24 hours")


agent = create_deep_agent(
    model=model,
    response_format=WeatherReport,
    tools=[internet_search],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in San Francisco?",
            }
        ]
    }
)

print(result["structured_response"])
# location='San Francisco, California' temperature=18.3 condition='Sunny' humidity=48 wind_speed=7.6 forecast='Pleasant sunny conditions expected to continue with temperatures around 64°F (18°C) during the day, dropping to around 52°F (11°C) at night. Clear skies with minimal precipitation expected.'
```

For more information and examples, see [response format](/oss/python/langchain/structured-output#response-format).

## Advanced

`create_deep_agent` pre-assembles a middleware stack on top of [`create_agent`](https://reference.langchain.com/python/langchain/agents/factory/create_agent). To build a fully custom agent—choosing exactly which capabilities to include—see [Configure the harness](/oss/python/langchain/agents#configure-the-harness).

***

<div className="source-links">
  <Callout icon="terminal-2">
    [Connect these docs](/use-these-docs) to Claude, VSCode, and more via MCP for real-time answers.
  </Callout>

  <Callout icon="edit">
    [Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/deepagents/customization.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).
  </Callout>
</div>
