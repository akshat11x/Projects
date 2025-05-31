


from image_to_text import auto_text

file_path=r"D:\Raghav\EVOLUTION\GOD_DEMON_IS_BACK\MEDICINE_API\files_for_test\Medical.pdf"


report=auto_text(file_path)



import os
from cerebras.cloud.sdk import Cerebras

# === Set your Cerebras API key here or from environment ===
# It's safer to set the key as an environment variable externally and read it in Python.
# Example in terminal: set CS_API_KEY=your_key_here
API_KEY = os.environ.get("CEREBRAS_API_KEY", "csk-kjdnkhmmcrw4wfced48mjrjmpejewktm2392kx4k3vm2n56c") # expected key name

if not API_KEY:
    raise ValueError("Missing Cerebras API key. Please set the environment variable 'CS_API_KEY'.")

# === Create Cerebras client ===
client = Cerebras(api_key=API_KEY)

# === Import your medical report content ===
from image_to_text import auto_text

file_path = r"D:\Raghav\EVOLUTION\GOD_DEMON_IS_BACK\MEDICINE_API\files_for_test\Medical.pdf"
report = auto_text(file_path)

# === Prepare the user prompt ===
user_prompt = (
    "You are a senior medical consultant. Analyze the provided medical report, "
    "summarize it, and list all medications with their side effects.\n\n"
    + report
)

# === Call Cerebras LLaMA-3.1-8B model ===
try:
    chat_completion = client.chat.completions.create(
        model="llama3.1-8b",
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )
    
    # === Print the generated response ===
    print("\n=== Generated Output ===")
    print(chat_completion.choices[0].message.content)

except Exception as e:
    print("❌ Error during Cerebras API call:", str(e))
