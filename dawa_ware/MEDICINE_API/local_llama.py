from llama_cpp import Llama

# Path to your GGUF model
llm_path = r"D:\Raghav\EVOLUTION\LLM MODELS\meta-llama\CodeLlama-13b-hf\CodeLlama-13b-hf_Q4_K_M.gguf"

# Load the model
llm = Llama(
    model_path=llm_path,
    n_ctx=4096,              # Context length
    n_threads=8,             # Adjust based on your CPU
    n_gpu_layers=1           # Set >0 if using GPU acceleration
)

# Prompt template for summarizing medical reports
def summarize_medical_report(report_text):
    prompt = f"""
You are a medical expert AI. Given the following medical report, perform two tasks:

1. Provide a clear, concise summary of the medical report.
2. List potential side effects of any medicines mentioned.

Medical Report:
\"\"\"
{report_text}
\"\"\"

Respond with:
- Summary:
- Side Effects:
"""
    response = llm(prompt, max_tokens=1024, temperature=0.7, stop=["</s>"])
    return response["choices"][0]["text"].strip()

# Example usage
report_text = """
Patient Name: John Doe
Date: 2025-05-21
Diagnosis: Hypertension
Medications: Amlodipine 5mg daily, Metoprolol 50mg twice daily
Notes: Blood pressure elevated, advised dietary changes and exercise.
"""

summary = summarize_medical_report(report_text)
print(summary)




#https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
#link for LLM model used

# from huggingface_hub import notebook_login
# notebook_login()
#ACCESS_TOKEN=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


#OLD CODE

# from transformers import AutoProcessor, AutoModelForImageTextToText
# from PIL import Image
# import torch

# # Load model and processor
# model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
# processor = AutoProcessor.from_pretrained(model_id)
# model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# # Load the image
# image = Image.open(r"D:\Raghav\EVOLUTION\GOD_DEMON_IS_BACK\MEDICINE_API\files_for_test\TEXT_TEST.png").convert("RGB")

# # Create the prompt
# prompt = "Summarize this medical report and list medicine side effects."

# # Preprocess
# inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

# # Generate
# generated_ids = model.generate(**inputs, max_new_tokens=512)
# output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print("Generated Output:\n", output)