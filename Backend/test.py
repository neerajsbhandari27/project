from google import genai
from dotenv import load_dotenv
from fast_api_upload

# Load environment variables
load_dotenv()
client = genai.Client()

result = client.models.embed_content(
        model="gemini-embedding-001",
        contents="What is the meaning of life?"
)

print(result.embeddings)