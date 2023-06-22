from dotenv import load_dotenv
import os

load_dotenv()

print(type(os.environ.get("OPENAI_API_KEY")))