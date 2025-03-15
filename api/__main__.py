import os
import uvicorn
import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import EventSourceResponse
from pydantic import BaseModel
from typing import List, Optional
from string import Template
from dotenv import load_dotenv
import json
import ollama
import time

load_dotenv()
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))
ORIGINS = os.getenv("ALLOWED_ORIGINS").split(",")

app = FastAPI(title="Ontology Patterns Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specifies which origins are allowed to access the API
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# do not touch
_temp_localstorage_data = {}

def _set_temp_localstorage_data(uuid: str, data: object):
    _temp_localstorage_data.setdefault(uuid, data)

def _get_temp_localstorage_data(uuid: str):
    try:
        data = _temp_localstorage_data[uuid]
        del _temp_localstorage_data[uuid]
        return data
    except:
        return None

# -----------------------------
# Ollama Setup
# -----------------------------

ollama = ollama.Client(
    host="https://llm.vse.cz/ollama"
)

# -----------------------------
# OPENAI Setup
# -----------------------------
openai_api_key = os.getenv("OPENAI_API_KEY", "")
if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found in environment!")
openai.api_key = openai_api_key

# -----------------------------
# Model routing Setup
# -----------------------------
model_provider_map = {
    "gpt-4o": "openai",
    "o1-preview": "openai",
    "o1-mini": "openai",
    "gpt-3.5-turbo": "openai",
}

def update_model_map():
    model_list = ollama.list()["models"]
    alphabetical_model_list = sorted(model_list, key=lambda m: m["model"])
    for model in alphabetical_model_list:
        name = model["model"]
        model_provider_map.setdefault(name, "ollama")

update_model_map()

class ExampleItem(BaseModel):
    A_label: str
    p_label: str
    B_label: str
    r_label: Optional[str] = None
    C_label: str
    Property: Optional[str] = None     # NEW
    Subclass: Optional[str] = None   

class Pattern1Request(BaseModel):
    A_label: str
    p_label: str
    B_label: str
    r_label: str
    C_label: str
    use_few_shot: bool
    few_shot_examples: List[ExampleItem] = []
    # Additional OpenAI params
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    #Ollama only
    repeat_penalty: float = 1.1
    pattern_name: str = "1_shortcut"
    output_schema: object = None


class Pattern2Request(BaseModel):
    A_label: str
    p_label: str
    B_label: str
    C_label: str
    use_few_shot: bool
    few_shot_examples: List[ExampleItem] = []
    # Additional OpenAI params
    model_name: str = "gpt-4o"
    temperature: float = 0.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    #Ollama only
    repeat_penalty: float = 1.1
    pattern_name: str = "2_subclass"
    output_schema: object = None


class Pattern1Response(BaseModel):
    property_name: str
    explanation: str

class Pattern2Response(BaseModel):
    class_name: str
    explanation: str

class TemporaryLocalStorageData(BaseModel):
    uuid: str
    data: object

def get_provider(model_name: str) -> str:
    return model_provider_map[model_name]

def load_text_file(filepath: str) -> str:
    """Loads a text file and returns its content as a string."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def load_output_schema(pattern_name: str) -> object:
    """Loads output schema file based on pattern name and returns the json schema as an object."""

    path = f"./prompts/{pattern_name}/output_schema.json"
    schema_text = load_text_file(path)
    try:
        schema = json.loads(schema_text)
    except Exception as e:
        print(f"Error reading output schema for {pattern_name}: {e}")
        return None
    
    return schema

def load_template(pattern_name: str, model_name: str, use_few_shot: bool) -> str:
    """Loads a prompt file based on required parameters and returns its content as a string."""

    provider = get_provider(model_name)
    technique = "few_shot" if use_few_shot else "baseline"
    path = f"./prompts/{pattern_name}/{provider}/{technique}.txt"

    return load_text_file(path)

def build_pattern1_prompt(data: Pattern1Request) -> str:
    """Build the prompt for Pattern1 (shortcut)."""

    data.output_schema = load_output_schema(data.pattern_name)

    template_content = load_template(
        pattern_name=data.pattern_name,
        model_name=data.model_name,
        use_few_shot=data.use_few_shot
    )

    if not template_content:
        raise HTTPException(status_code=500, detail="Prompt template not found (Pattern1).")

    if get_provider(data.model_name) == "ollama":
        data.output_schema = load_output_schema(data.pattern_name)# Add the output schema to data for later use

    few_shot_str = ""
    if data.use_few_shot and data.few_shot_examples:
        lines = []
        for ex in data.few_shot_examples:
            print(ex)
            r_lab = ex.r_label if ex.r_label else "..."
            snippet = f"""Input:
- Class A: {ex.A_label}
  - Property p: {ex.p_label}
- Class B: {ex.B_label}
  - Property r: {r_lab}
- Class C: {ex.C_label}

Suggested Property Name: {ex.Property}
---
"""
            lines.append(snippet.strip())
        few_shot_str = "\n\n".join(lines)

    tpl = Template(template_content)
    final_prompt = tpl.safe_substitute(
        few_shot_examples=few_shot_str,
        A_label=data.A_label,
        p_label=data.p_label,
        B_label=data.B_label,
        r_label=data.r_label,
        C_label=data.C_label,
        output_schema=json.dumps(data.output_schema)
    )
    return final_prompt

def build_pattern2_prompt(data: Pattern2Request) -> str:
    """Build the prompt for Pattern2 (subclass)."""

    template_content = load_template(
        pattern_name=data.pattern_name,
        model_name=data.model_name,
        use_few_shot=data.use_few_shot
    )

    if not template_content:
        raise HTTPException(status_code=500, detail="Prompt template not found (Pattern2).")

    if get_provider(data.model_name) == "ollama":
        data.output_schema = load_output_schema(data.pattern_name)# Add the output schema to data for later use

    few_shot_str = ""
    if data.use_few_shot and data.few_shot_examples:
        lines = []
        for ex in data.few_shot_examples:
            snippet = f"""Input:
- Class A: {ex.A_label}:
  - Property p ({ex.p_label}):
    - Domain: {ex.A_label}
    - Range: {ex.B_label}
- Class B: {ex.B_label}
- Class C: {ex.C_label}, a subclass of {ex.B_label}

Suggested Class Name: {ex.Subclass}
---
"""
            lines.append(snippet.strip())
        few_shot_str = "\n\n".join(lines)

    tpl = Template(template_content)
    final_prompt = tpl.safe_substitute(
        few_shot_examples=few_shot_str,
        A_label=data.A_label,
        p_label=data.p_label,
        B_label=data.B_label,
        C_label=data.C_label,
        output_schema=json.dumps(data.output_schema)
    )
    return final_prompt

def call_openai_chat(
    model_name: str,
    prompt_text: str,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
) -> str:
    provider = model_provider_map.get(model_name)
    if not provider:
        raise ValueError(f"Unknown model name: {model_name}")
    
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY.")
    
    if model_name in ["o1-preview", "o1-mini"]:
        temperature = 1.0
        
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        return response.choices[0].message.content.strip().replace("json","").replace("`","")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API call failed: {e}")


def call_ollama_chat(
    model_name: str,
    prompt_text: str,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    output_schema: object = None
):
    if output_schema is None:
        print("Warning: No response schema provided for the call to Ollama API." +
              "Defaulting 'format' parameter to generic 'json'")

    try:
        response: ollama.ChatResponse = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            format=output_schema if output_schema is not None else "json",
            options={
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "num_ctx": 4096
            }
        )
        return response.message.content.strip().replace("json","").replace("`","")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama API call failed: {e}")


@app.post("/generate_shortcut")
def generate_pattern1(data: Pattern1Request):
    # 1) Build the prompt
    prompt_text = build_pattern1_prompt(data)
    
    # 2) determine model provider
    provider = get_provider(data.model_name)

    # 3a) Call llm chat by provider
    if provider == "openai":
        raw_answer = call_openai_chat(
            model_name=data.model_name,
            prompt_text=prompt_text,
            temperature=data.temperature,
            top_p=data.top_p,
            frequency_penalty=data.frequency_penalty,
            presence_penalty=data.presence_penalty,
        )
    elif provider == "ollama":
        raw_answer = call_ollama_chat(
            model_name=data.model_name,
            prompt_text=prompt_text,
            temperature=data.temperature,
            top_p=data.top_p,
            repeat_penalty=data.repeat_penalty,
            output_schema=data.output_schema
        )
    # 4) Parse the LLM output as JSON
    try:
        parsed_json = json.loads(raw_answer)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="API did not return valid JSON. Raw output was:\n" + raw_answer
        )
    # 5) Extract fields from JSON
    prop_name = parsed_json.get("property_name", "UnknownProperty")
    explanation = parsed_json.get("explanation", "")
    
    return Pattern1Response(property_name=prop_name, explanation=explanation)

@app.post("/generate_subclass")
def generate_pattern2(data: Pattern2Request):
    # 1) Build the prompt
    prompt_text = build_pattern2_prompt(data)
    
    # 2) determine model provider
    provider = get_provider(data.model_name)

    # 3) Call llm chat by provider
    if provider == "openai":
        raw_answer = call_openai_chat(
            model_name=data.model_name,
            prompt_text=prompt_text,
            temperature=data.temperature,
            top_p=data.top_p,
            frequency_penalty=data.frequency_penalty,
            presence_penalty=data.presence_penalty,
        )
    elif provider == "ollama":
        raw_answer = call_ollama_chat(
            model_name=data.model_name,
            prompt_text=prompt_text,
            temperature=data.temperature,
            top_p=data.top_p,
            repeat_penalty=data.repeat_penalty,
            output_schema=data.output_schema
        )
    # 4) Parse the LLM output as JSON
    try:
        parsed_json = json.loads(raw_answer)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="API did not return valid JSON. Raw output was:\n" + raw_answer
        )
    # 5) Extract fields from JSON
    class_name = parsed_json.get("class_name", "UnknownClass")
    explanation = parsed_json.get("explanation", "")
    
    return Pattern2Response(class_name=class_name, explanation=explanation)

@app.post("/shortcut_prompt")
def prompt_pattern1(data: Pattern1Request):
    """
    Return the *complete* prompt that would be sent to API for Pattern1.
    """
    prompt_text = build_pattern1_prompt(data)
    return {"prompt": prompt_text}

@app.post("/subclass_prompt")
def prompt_pattern2(data: Pattern2Request):
    """
    Return the *complete* prompt that would be sent to API for Pattern2.
    """
    prompt_text = build_pattern2_prompt(data)
    return {"prompt": prompt_text}

@app.get("/model_provider_map")
def get_model_names():
    """
    Return the list of available model names set by the developer in model_provider_map
    """
    update_model_map()
    return model_provider_map

@app.post("/_temp_localstorage_data")
def save_temp_session_data(req: TemporaryLocalStorageData):
    data = json.loads(req.data)
    for i in range(len(data)):
        data[i] = json.loads(data[i])
    _set_temp_localstorage_data(req.uuid, data)
    return {"status": 200 }

@app.get("/_temp_localstorage_data")
def send_temp_session_data(uuid: str):
    time.sleep(2)
    data = _get_temp_localstorage_data(uuid)
    return data

if __name__ == "__main__":
    uvicorn.run(
        "__main__:app",
        host=HOST if HOST else "0.0.0.0",
        port=PORT if PORT else 8000, reload=True
    )

"""
mycomponent/index.html
<html>
  <body>
    <!-- Set up your HTML here -->
    <input id="myinput" value="" />

    <script>
      // ----------------------------------------------------
      // Just copy/paste these functions as-is:

      function sendMessageToStreamlitClient(type, data) {
        var outData = Object.assign({
          isStreamlitMessage: true,
          type: type,
        }, data);
        window.parent.postMessage(outData, "*");
      }

      function init() {
        sendMessageToStreamlitClient("streamlit:componentReady", {apiVersion: 1});
      }

      function setFrameHeight(height) {
        sendMessageToStreamlitClient("streamlit:setFrameHeight", {height: height});
      }

      // The `data` argument can be any JSON-serializable value.
      function sendDataToPython(data) {
        sendMessageToStreamlitClient("streamlit:setComponentValue", data);
      }

      // ----------------------------------------------------
      // Now modify this part of the code to fit your needs:

      var myInput = document.getElementById("myinput");

      // data is any JSON-serializable value you sent from Python,
      // and it's already deserialized for you.
      function onDataFromPython(event) {
        if (event.data.type !== "streamlit:render") return;
        myInput.value = event.data.args.my_input_value;  // Access values sent from Python here!
      }

      myInput.addEventListener("change", function() {
        sendDataToPython({
          value: myInput.value,
          dataType: "json",
        });
      })

      // Hook things up!
      window.addEventListener("message", onDataFromPython);
      init();

      // Hack to autoset the iframe height.
      window.addEventListener("load", function() {
        window.setTimeout(function() {
          setFrameHeight(document.documentElement.clientHeight)
        }, 0);
      });

      // Optionally, if the automatic height computation fails you, give this component a height manually
      // by commenting out below:
      //setFrameHeight(200);
    </script>
  </body>
</html>

mycomponent/__init__.py
import streamlit.components.v1 as components
mycomponent = components.declare_component(
    "mycomponent",
    path="./mycomponent"
)

IN APP:
import streamlit as st
from mycomponent import mycomponent
value = mycomponent(my_input_value="hello there")
st.write("Received", value)
"""

