
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": "Explain EnergyIQ in one sentence."
            }
        ],
        max_tokens=100
    )

    print("Groq API connected successfully!")
    print(response.choices[0].message.content)

except Exception as e:
    print("API request failed:", e)
