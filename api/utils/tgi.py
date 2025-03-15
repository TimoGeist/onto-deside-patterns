from huggingface_hub import InferenceClient
import requests

# -----------------------------
# TGI Setup
# -----------------------------
tgi : object = None
try:
    res = requests.get("https://llm.vse.cz/tgi/info")
    res.raise_for_status()
    tgi = InferenceClient("https://llm.vse.cz/tgi")
    print("Success connecting to Text Generation Inference API. Info:", res.json())
except Exception as e:
    print(f"Warning: Can't connect to Text Generation Inference API: {e}")

# -----------------------------
# Model routing Setup
# -----------------------------
# add at the end
if tgi is not None:
    model_provider_map = ["llama-3.1-8b-instruct(fp16)"] = "tgi"
model_names = list(model_provider_map.keys())

def call_tgi_chat(
    model_name: str, #ignored param, declared to keep the interface consistent
    prompt_text: str,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    response_type: object = None
):
    if not response_type:
        raise ValueError("Missing response type for Text Generation Inference API Call.")
    
    try:
        response = tgi.chat.completions.create(
            model="tgi",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature if temperature else 0.7,
            max_tokens=2000,
            top_p=max(min(top_p, 0.99), 0.01),
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
            # tools=tools,
            # tool_choice="auto"
        )
        data = response.choices[0].message.content.split().strip().replace("json","").replace("`","")
        return data
    except Exception as e:
        print("err", e)
        raise HTTPException(status_code=500, detail=f"Text Generation Inference API call failed: {e}")