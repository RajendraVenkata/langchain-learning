from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model

# Create a multimodal message
message = HumanMessage(content_blocks=[
    {"type": "text", "text": "What do you see in this image?"},
    {"type": "image", "url": "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"}
])

# Send to model
model = init_chat_model("gpt-5-mini")
response = model.invoke([message])

print(response.text)
# Output: "This image shows a brown cat..."

# Different image types
message_with_types = HumanMessage(content_blocks=[
    {"type": "text", "text": "Analyze these:"},
    {"type": "image", "url": "https://img.magnific.com/free-vector/red-arrow-going-up-with-bar-chart_1308-110320.jpg?semt=ais_hybrid&w=740&q=80", "mime_type": "image/png"},
    {"type": "image", "url": "https://example.com/diagram.jpg", "mime_type": "image/jpeg"}
])

#response = model.invoke([message_with_types])
#print(response.text)