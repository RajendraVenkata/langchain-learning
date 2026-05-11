from langchain.chat_models import init_chat_model

# Initialize any chat model
model = init_chat_model("gpt-5-mini")

# Get a response
response = model.invoke("Explain photosynthesis")

# Access standard content blocks (provider-agnostic)
print("Content blocks:")
for block in response.content_blocks:
    if block["type"] == "text":
        print(f"Text: {block['text']}")
    elif block["type"] == "reasoning":
        print(f"Reasoning: {block['reasoning']}")

# The same code works for OpenAI models too
model_openai = init_chat_model("gpt-5-mini")
response_openai = model_openai.invoke("Explain photosynthesis")

# Same loop works without changes!
for block in response_openai.content_blocks:
    if block["type"] == "text":
        print(f"Text: {block['text']}")

        