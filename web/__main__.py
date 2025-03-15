import streamlit as st
from streamlit import session_state as sess
import requests
import pandas as pd
from typing import List
from utils import load_text_file
from dotenv import load_dotenv
import os
from streamlit.components.v1 import html
from streamlit_js import st_js, st_js_blocking
from uuid import uuid4
import json
from datetime import datetime
import time

load_dotenv()

##############################
# Must be first Streamlit command
st.set_page_config(layout="wide")
##############################

custom_css = load_text_file("./assets/custom.css")

def local_css(css_content):
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

local_css(custom_css)

# -----------------------------
# Some global placeholders
# -----------------------------
maybe_backend_url = os.getenv("BACKEND_URL")
BACKEND_URL = maybe_backend_url if maybe_backend_url else "http://localhost:8000"
PATTERN1_DEFAULT_CSV = "./fewshot_data/better_than_human.csv"
PATTERN2_DEFAULT_CSV = "./fewshot_data/better_than_human_2.csv"

# Show two logos in the sidebar
with st.sidebar:
    st.image("./assets/onto-deside-logo.png", use_column_width=True)
    st.image("./assets/kizi-logo.png", use_column_width=True)

def log_action(msg: str):
    st.write(f"[LOG] {msg}")

def login_user(username, password):
    # Example hard-coded logic
    if username in ("peter", "vojtech") and password == "VSE":
        sess["logged_in"] = True
        sess["username"] = username
        log_action(f"User '{username}' logged in successfully.")
        return True
    else:
        return False

def show_login_page():
    st.title("Please Log In")
    user = st.text_input("Username:")
    pwd = st.text_input("Password:", type="password")
    if st.button("Login"):
        if login_user(user, pwd):
            st.success("Logged in successfully!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

####################
# Few-shot CSV utilities
####################
def load_few_shot_csv_file(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        df.rename(columns=lambda x: x.replace("?", "").strip(), inplace=True)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
        return df
    except Exception as e:
        st.error(f"Error reading CSV '{filepath}': {e}")
        return pd.DataFrame()

def load_few_shot_upload(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file)
        df.rename(columns=lambda x: x.replace("?", "").strip(), inplace=True)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
        return df
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
        return pd.DataFrame()

def show_few_shot_editor(
    state_key: str, required_cols: List[str], default_csv_path: str
):
    """
    Enhanced version that loads default CSV only when the current table is empty.
    When a new file is uploaded, it immediately replaces the old table.
    """
    if state_key not in sess:
        sess[state_key] = pd.DataFrame(columns=required_cols)

    up_csv = st.file_uploader(f"Upload new CSV for {state_key}", type=["csv"])
    if up_csv is not None:
        newdf = load_few_shot_upload(up_csv)
        missing_cols = [c for c in required_cols if c not in newdf.columns]
        if missing_cols:
            st.error(f"Uploaded CSV missing columns: {missing_cols}")
        else:
            sess[state_key] = newdf.reset_index(drop=True)
            st.success("New few-shot CSV loaded successfully!")

    # If still empty, load default
    if sess[state_key].empty:
        st.info(f"Loading default few-shot CSV: {default_csv_path}")
        df_default = load_few_shot_csv_file(default_csv_path)
        missing_cols = [c for c in required_cols if c not in df_default.columns]
        if missing_cols:
            st.error(f"Default CSV missing columns: {missing_cols}")
        else:
            sess[state_key] = df_default.reset_index(drop=True)

    df = sess[state_key]
    st.write(f"**Currently {len(df)} few-shot examples** for `{state_key}`:")
    st.dataframe(df, use_container_width=True)

####################
# Backend calls
####################
def api_post(endpoint: str, payload: dict):
    url = f"{BACKEND_URL}{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"POST {endpoint} failed: {e}")
        return None

def api_get(endpoint: str, params: dict = None):
    url = f"{BACKEND_URL}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"GET {endpoint} failed: {e}")
        return None

####################
# Graphing
####################
def pattern1_graphviz(A, p, B, r, C, new_prop=None):
    dot = "digraph G {\n"
    dot += "rankdir=LR;\n"
    dot += f'A [label="{A}"];\n'
    dot += f'B [label="{B}"];\n'
    dot += f'C [label="{C}"];\n'
    dot += f'A -> B [label="{p}"];\n'
    dot += f'B -> C [label="{r}"];\n'
    if new_prop:
        dot += f'A -> C [label="{new_prop}", color="blue", penwidth=2];\n'
    dot += "}"
    return dot

def pattern2_graphviz(A, p, B, C, new_subclass=None):
    dot = "digraph G {\n"
    dot += "rankdir=LR;\n"
    dot += f'A [label="{A}"];\n'
    dot += f'B [label="{B}"];\n'
    dot += f'C [label="{C}\\n(subclass of {B})"];\n'
    dot += f'A -> B [label="{p}"];\n'
    dot += 'C -> B [label="subclassOf", style="dashed"];\n'
    if new_subclass:
        dot += f'NEW [label="{new_subclass}\\n(subclass of {A})", color="blue"];\n'
        dot += (
            'NEW -> A [label="subclassOf", color="blue", style="dashed", penwidth=2];\n'
        )
    dot += "}"
    return dot

def init(key: str, value: any):
    if key not in sess:
        sess[key] = value

####################
# MAIN TABBED APP
####################
def main():

    init("logged_in", True)

    if not sess["logged_in"]:
        show_login_page()
        return
    
    st.sidebar.title("Model Parameters")

    init("model_name", "gpt-4o")
    init("model_provider_map", api_get("/model_provider_map"))
    model_names = list(sess["model_provider_map"].keys())
    st.sidebar.selectbox(
        label="Model Name",
        options=model_names,
        key="model_name"
    )

    provider_name = sess["model_provider_map"][
        sess["model_name"]
    ]

    init("temperature", 0.0)
    init("top_p", 1.0)
    init("frequency_penalty", 0.0)
    init("presence_penalty", 0.0)
    init("repeat_penalty", 1.1)

    st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        key="temperature",
        step=0.1,
        value=0.0 if provider_name == "openai" else 0.7
    )

    sess["top_p_slider"] = st.sidebar.slider(
        "Top-p",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="top_p"
    )
    

    if (provider_name == "openai"):
        st.sidebar.slider(
            "Frequency Penalty",
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            key="frequency_penalty"
        )
        st.sidebar.slider(
            "Presence Penalty",
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            key="presence_penalty"
        )
    elif (provider_name == "ollama"):
        st.sidebar.slider(
            "Repeat Penalty",
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            key="repeat_penalty",
            value=1.1
        )

    

    st.sidebar.markdown("---")
    
    init("target_tab", None)

    tabs = st.tabs(["Pattern 1: Shortcut", "Pattern 2: Subclass", "Session History", "Documentation"])

    # if tabs and sess["initial_tab"]:
    #     html("<script>" + switch(sess["initial_tab"]) + "</script>",  f"tabSwitcher{sess["arbitrary_key"]}")
    if sess["target_tab"] is not None:
        # Generate and execute the script
        st_js(switch_tab_script(sess["target_tab"]))
        # , height=0
        # Clear the target to avoid re-execution
        sess["target_tab"] = None

    init("shortcut_A_label", "corpus_part")
    init("shortcut_B_label", "Genre")
    init("shortcut_C_label", "Music Genre")
    init("shortcut_p_label", "genre")
    init("shortcut_r_label", "has sub-genre")

    # Pattern 1 Tab
    with tabs[0]:
        st.header("Pattern 1: Object Property Chain Shortcut")

        inputs_left, inputs_right = st.columns([1, 1])
        with inputs_left:
            st.subheader("Class A label")
            st.text_input("", key="shortcut_A_label")

            st.subheader("Class B label")
            st.text_input("", key="shortcut_B_label")

            st.subheader("Class C label")
            st.text_input("", key="shortcut_C_label")

        with inputs_right:
            st.subheader("Property p label")
            st.text_input("", key="shortcut_p_label")

            st.subheader("Property r label")
            st.text_input("", key="shortcut_r_label")

        init("shortcut_use_few_shot", False)
        st.checkbox("Use Few-Shot Examples (Pattern 1)?", key="shortcut_use_few_shot")
        if sess["shortcut_use_few_shot"]:
            show_few_shot_editor(
                "shortcut_few_shot_data",
                ["A_label", "p_label", "B_label", "r_label", "C_label", "Property"],
                PATTERN1_DEFAULT_CSV,
            )

        st.markdown("**Diagram (before):**")
        dot_before = pattern1_graphviz(
            sess["shortcut_A_label"],
            sess["shortcut_p_label"],
            sess["shortcut_B_label"],
            sess["shortcut_r_label"],
            sess["shortcut_C_label"]
        )
        st.graphviz_chart(dot_before)

        init("shortcut_prompt", None)
        init("shortcut_result", None)

        shortcut_buttons_left, shortcut_buttons_right = st.columns([1, 1])
        shortcut_generate = shortcut_buttons_left.button("Generate Shortcut Property")
        if shortcut_generate:
            sess["shortcut_prompt"] = None
            shortcut_fs_examples = []
            if sess["shortcut_use_few_shot"]:
                df_few = sess["shortcut_few_shot_data"]
                for _, row in df_few.iterrows():
                    shortcut_fs_examples.append({
                        "A_label": row.get("A_label", ""),
                        "p_label": row.get("p_label", ""),
                        "B_label": row.get("B_label", ""),
                        "r_label": row.get("r_label", ""),
                        "C_label": row.get("C_label", ""),
                        "Property": row.get("Property", "")
                    })
            payload = build_payload("shortcut", shortcut_fs_examples)
            with st.spinner("Generating shortcut property..."):
                resp = api_post("/generate_shortcut", payload)
            if resp:
                sess["shortcut_result"] = resp
                flat = { "time": get_iso_time() }
                for entry in payload.items():
                    flat[entry[0]] = entry[1]
                for entry in resp.items():
                    flat[entry[0]] = entry[1]
                save_dict_localstorage(flat)

        shortcut_show_prompt = shortcut_buttons_right.button("Show Prompt (Pattern 1)")
        if shortcut_show_prompt:
            sess["shortcut_result"] = None
            shortcut_fs_examples = []
            if sess["shortcut_use_few_shot"]:
                df_few = sess["shortcut_few_shot_data"]
                for _, row in df_few.iterrows():
                    shortcut_fs_examples.append({
                        "A_label": row.get("A_label", ""),
                        "p_label": row.get("p_label", ""),
                        "B_label": row.get("B_label", ""),
                        "r_label": row.get("r_label", ""),
                        "C_label": row.get("C_label", ""),
                        "Property": row.get("Property", ""),
                    })
            payload = build_payload("shortcut", shortcut_fs_examples)
            with st.spinner("Retrieving complete prompt..."):
                shortcut_prompt_response = api_post("/shortcut_prompt", payload)
            if shortcut_prompt_response:
                sess["shortcut_prompt"] = shortcut_prompt_response["prompt"]

        # Display prompt and/or generation results below (full width)
        if sess["shortcut_prompt"]:
            st.markdown("**Complete Prompt (Pattern 1):**")
            st.json({"prompt": sess["shortcut_prompt"]})

        if sess["shortcut_result"]:
            resp = sess["shortcut_result"]
            st.subheader("Shortcut Generation Result")
            new_prop = resp.get("property_name", "(no name)")
            explanation = resp.get("explanation", "(no explanation)")
            st.markdown(f"**Suggested Property**: {new_prop}")
            st.markdown("**Explanation:**")
            st.write(explanation)

            st.markdown("**Diagram (after)**:")
            dot_after = pattern1_graphviz(
                sess["shortcut_A_label"],
                sess["shortcut_p_label"],
                sess["shortcut_B_label"],
                sess["shortcut_r_label"],
                sess["shortcut_C_label"],
                new_prop=new_prop
            )
            st.graphviz_chart(dot_after)
    
    init("subclass_A_label", "System")
    init("subclass_B_label", "Component")
    init("subclass_C_label", "Storage Device")
    init("subclass_p_label", "has component")

    # Pattern 2 Tab
    with tabs[1]:
        st.header("Pattern 2: Subclass Enrichment")
        subclass_l1, subclass_r1 = tabs[1].columns([1, 1])

        with subclass_l1:
            st.subheader("Class A label")
            st.text_input("", key="subclass_A_label")

            st.subheader("Class B label")




            st.text_input("", key="subclass_B_label")

        with subclass_r1:
            st.subheader("Property p label")
            st.text_input("", key="subclass_p_label")

            st.subheader("Class C label (subclass of B)")
            st.text_input("", key="subclass_C_label")

        init("subclass_use_few_shot", False)
        st.checkbox("Use Few-Shot Examples (Pattern 2)?", key="subclass_use_few_shot")
        if sess["subclass_use_few_shot"]:
            show_few_shot_editor(
                "subclass_few_shot_data",
                ["A_label", "p_label", "B_label", "C_label", "Subclass"],
                PATTERN2_DEFAULT_CSV,
            )

        st.markdown("**Diagram (before):**")
        dot_before2 = pattern2_graphviz(
            sess["subclass_A_label"],
            sess["subclass_p_label"],
            sess["subclass_B_label"],
            sess["subclass_C_label"]
        )
        st.graphviz_chart(dot_before2)

        init("subclass_prompt", None)
        init("subclass_result", None)

        subclass_buttons_left, subclass_buttons_right = st.columns([1, 1])
        subclass_generate = subclass_buttons_left.button("Generate Specialized Subclass")
        if subclass_generate:
            sess["subclass_prompt"] = None
            subclass_fs_examples = []
            if sess["subclass_use_few_shot"]:
                df_few2 = sess["subclass_few_shot_data"]
                for _, row in df_few2.iterrows():
                    subclass_fs_examples.append({
                        "A_label": row.get("A_label", ""),
                        "p_label": row.get("p_label", ""),
                        "B_label": row.get("B_label", ""),
                        "C_label": row.get("C_label", ""),
                        "Subclass": row.get("Subclass", ""),
                    })

            payload2 = build_payload("subclass", subclass_fs_examples)
            with st.spinner("Generating specialized subclass..."):
                resp2 = api_post("/generate_subclass", payload2)
            if resp2:
                sess["subclass_result"] = resp2
                flat = { "time": get_iso_time()  }
                for entry in payload2.items():
                    flat[entry[0]] = entry[1]
                for entry in resp2.items():
                    flat[entry[0]] = entry[1]
                save_dict_localstorage(flat)

        subclass_show_prompt = subclass_buttons_right.button("Show Prompt (Pattern 2)")
        if subclass_show_prompt:
            sess["subclass_result"] = None
            subclass_fs_examples = []
            if sess["subclass_use_few_shot"]:
                df_few2 = sess["subclass_few_shot_data"]
                for _, row in df_few2.iterrows():
                    subclass_fs_examples.append({
                        "A_label": row.get("A_label", ""),
                        "p_label": row.get("p_label", ""),
                        "B_label": row.get("B_label", ""),
                        "C_label": row.get("C_label", ""),
                        "Subclass": row.get("Subclass", ""),
                    })
            payload = build_payload("subclass", subclass_fs_examples)
            with st.spinner("Retrieving complete prompt..."):
                subclass_prompt_response = api_post("/subclass_prompt", payload)
            if subclass_prompt_response:
                sess["subclass_prompt"] = subclass_prompt_response["prompt"]

        

        # Display prompt & generation results below (full width)
        if sess["subclass_prompt"]:
            st.markdown("**Complete Prompt (Pattern 2):**")
            st.json({"prompt": sess["subclass_prompt"]})

        if sess["subclass_result"]:
            resp2 = sess["subclass_result"]
            st.subheader("Subclass Generation Result")
            new_class = resp2.get("class_name", "(no name)")
            explanation = resp2.get("explanation", "(no explanation)")
            st.markdown(f"**Suggested Class**: {new_class}")
            st.markdown("**Explanation:**")
            st.write(explanation)

            st.markdown("**Diagram (after)**:")
            dot_after2 = pattern2_graphviz(
                sess["subclass_A_label"],
                sess["subclass_p_label"],
                sess["subclass_B_label"],
                sess["subclass_C_label"]
            , new_subclass=new_class)
            st.graphviz_chart(dot_after2)

    # Session History
    with tabs[2]:
        build_session_history_tab()

    # Documentation
    with tabs[3]:
        build_documentation()      


# def go(tab: int):
#     return f"""document.addEventListener("DOMContentLoaded", function() {{
#         setTimeout(() => {{
#             const tabs = document.querySelectorAll('.stTabs [role="tab"]');
#             if (tabs.length > {tab}) {{
#                 tabs[{tab}].click();
#             }}
#         }}, 500);  // Wait 500ms to ensure elements are rendered
#     }});
# """

def transform_operation_for_export(o):
    provider_name = sess["model_provider_map"][o["model_name"]]

    return {
        **o,
        "model_provider": provider_name,
        "timestamp": get_timestamp_from(o["time"], "%d/%m/%Y at %H:%M:%S"),
        "repeat_penalty": o["repeat_penalty"] if provider_name == "ollama" else None,
        "frequency_penalty": o["frequency_penalty"] if provider_name == "openai" else None,
        "presence_penalty": o["presence_penalty"] if provider_name == "openai" else None
    }

def build_session_history_tab():
    st.header("Session History")

    operations = get_dicts_localstorage()

    if (len(operations) == 0):
        st.write("Session history is empty. Generate something first.")
        return
    
    time_format = "%d/%m/%Y at %H:%M:%S"
    operations_for_export = list(map(lambda o: transform_operation_for_export(o), operations))
    df_export = pd.DataFrame(operations_for_export)
    csv = df_export.to_csv(index=False).encode("utf-8")

    a1, a2 = st.columns([3, 9])
    a1.download_button("Export CSV", csv, "patterns_history.csv", "text/csv", key="download-csv")
    a2.button("Clear History", on_click=clear_localstorage)
    st.markdown("---")
    sorted_operations =  sorted(operations, key=lambda o: get_timestamp_from(o["time"], time_format), reverse=True)
    for o in sorted_operations:
        pattern = (
            "Shortcut" if "property_name" in o else
            "Subclass" if "class_name" in o else
            "Unknown Pattern"
        )
        provider_name = sess["model_provider_map"][o["model_name"]]
        
        columns_to_exclude1 = ['explanation', 'few_shot_examples', 'time', "property_name", "class_name"]
        openai_exclude = ["repeat_penalty"]
        ollama_exclude = ["frequency_penalty", "presence_penalty"]
        subclass = ["r_label"]

        if provider_name == "openai":
            columns_to_exclude1 += openai_exclude
        elif provider_name == "ollama":
            columns_to_exclude1 += ollama_exclude
        
        if pattern == "Subclass":
            columns_to_exclude1 += subclass

        filtered_data1 = { key: value for key, value in o.items() if key not in columns_to_exclude1 }

        result = o["property_name"] if pattern == "Shortcut" else o["class_name"]

        st.markdown(
            f"##### {pattern}:&nbsp;&nbsp;{result} \r {o['time']}"
        )

        openai_cols = (
            ["Class_A", "Property_p", "Class_B"] + (
                ["Property_r"] if pattern == "Shortcut" else []
            ) + (
                ["Class_C (⊆ B)" if pattern == "Subclass" else "Class_C", "model_name", "few_shot", "temp", "top_p", "freq_pen", "pres_pen"]
            )
        )
        ollama_cols = (
            ["Class_A", "Property_p", "Class_B"] + (
                    ["Property_r"] if pattern == "Shortcut" else []
            ) + (
                ["Class_C (⊆ B)" if pattern == "Subclass" else "Class_C", "model_name", "few_shot", "temp", "top_p", "repe_pen"]
            )
        )
        df = pd.DataFrame([filtered_data1])
        df.columns = openai_cols if provider_name == "openai" else ollama_cols 
        st.dataframe(df, hide_index=True)

        st.markdown(f"**Explanation:** {o['explanation']}")

        if o["use_few_shot"]:
            st.markdown("**Few shot examples:**")
            st.json(o["few_shot_examples"], expanded=False)

        long_pattern = f"Pattern 1: {pattern}" if pattern == "Shortcut" else f"Pattern 2: {pattern}"
        st.button(f"Load {long_pattern}", key=f"{o['time']}", on_click=lambda o=o: load(o))

        st.markdown("---")

def get_timestamp_from(input: str, time_format: str) -> float:
    return datetime.strptime(input, time_format).timestamp()

def get_iso_time():
    return datetime.now().strftime("%d/%m/%Y at %H:%M:%S")

def clear_localstorage():
    st_js(
        code='''
        Object.keys(localStorage)
            .filter(key => key.startsWith("sess_data_#_"))
            .forEach(key => localStorage.clear(key));
        ''',
    )


def load(operation):
    if "property_name" in operation:
        pattern_name = "shortcut"
    elif "class_name" in operation:
        pattern_name = "subclass"

    sess[f"{pattern_name}_A_label"] = operation["A_label"]
    if "property_name" in operation:
        sess[f"{pattern_name}_r_label"] = operation["r_label"]
        sess["property_name"] = operation["property_name"]
        sess["shortcut_result"] = {
            "property_name": operation["property_name"],
            "explanation": operation["explanation"]
        }
    if "class_name" in operation:
        sess["subclass_result"] = {
            "class_name": operation["class_name"],
            "explanation": operation["explanation"]
        }

    sess[f"{pattern_name}_B_label"] = operation["B_label"]
    sess["shortcut_p_label"] = operation["p_label"]

    sess[f"{pattern_name}_C_label"] = operation["C_label"]
    sess[f"{pattern_name}_use_few_shot"] = operation["use_few_shot"]
    # sess["few_shot_examples"] = operation["few_shot_examples"],
    sess["model_name"] = operation["model_name"]
    sess["temperature"] = operation["temperature"]
    sess["top_p"] = operation["top_p"]
    sess["frequency_penalty"] = operation["frequency_penalty"]
    sess["presence_penalty"] = operation["presence_penalty"]
    sess["repeat_penalty"] = operation["repeat_penalty"]

    if pattern_name == "shortcut":
        set_target_tab(0)
    elif pattern_name == "subclass":
        set_target_tab(1)

def switch_tab_script(tab_index):
    """Generate JavaScript to switch to the specified tab."""
    return f"""
    var tabGroup = window.parent.document.getElementsByClassName("stTabs")[0];
    if (tabGroup) {{
        var tabs = tabGroup.getElementsByTagName("button");
        if (tabs.length > {tab_index}) {{
            tabs[{tab_index}].click();
        }}
    }}
    """

# Define button actions
def set_target_tab(tab_index):
    st.session_state.target_tab = tab_index

def build_csv(operations: dict):
    headings = "pattern_name,A_label,B_label,r_label,C_label,few_shot_data,model_name,temperature,top_p,frequency_penalty,repeat_penalty,p_label,property_name,explanation"
    for o in operations:
        headings += o

def build_payload(pattern_name: str, fs_examples: dict):
    common1 = { 
        "A_label": sess[f"{pattern_name}_A_label"],
        "p_label": sess[f"{pattern_name}_p_label"],
        "B_label": sess[f"{pattern_name}_B_label"]
    }

    if pattern_name == "shortcut":
        common1.setdefault("r_label", sess[f"{pattern_name}_r_label"])
    
    common2 = {
        "C_label": sess[f"{pattern_name}_C_label"],
            "model_name": sess["model_name"],
            "use_few_shot": sess[f"{pattern_name}_use_few_shot"],
            "few_shot_examples": fs_examples,
            "temperature": sess["temperature"],
            "top_p": sess["top_p"],
            "frequency_penalty": sess["frequency_penalty"],
            "presence_penalty": sess["presence_penalty"],
            "repeat_penalty": sess["repeat_penalty"],
    }

    return { **common1, **common2 }

def save_dict_localstorage(dict: dict) -> None:
    data = json.dumps(dict)
    html('''
        <script>
            const counter = parseInt(
                localStorage.getItem("sess_data_counter")
                || "0"
            );
            localStorage.setItem("sess_data_#_" + counter,`'''+data+'''`);
            localStorage.setItem("sess_data_counter", counter + 1);
        </script>
    ''', width=0, height=0)

def get_dicts_localstorage():
    uuid = str(uuid4())
    html('''
        <script>
            counter = parseInt(localStorage.getItem("sess_data_counter"));
            const sess_data = [];
            for (let i = 0; i < counter; i++) {
                const query = "sess_data_#_" + String(i);
                const data = localStorage.getItem(query);
                sess_data.push(data);
            }
            const sess_data_serialized = JSON.stringify(sess_data)
            fetch("'''+BACKEND_URL+'''/_temp_localstorage_data", {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                uuid:"'''+uuid+'''",
                data: sess_data_serialized
              })
            });
        </script>
    ''',width=0, height=0)
    res = api_get("/_temp_localstorage_data?uuid=" + uuid)
    return res  

def build_documentation():
    md = '''## Ontology Patterns User Documentation
### Introduction  
The Ontology Patterns Backend is a tool designed to assist in ontology engineering by automating the generation of ontology patterns using large language models (LLMs). It supports two patterns:  
- **Pattern 1 (Shortcut):** Creates a direct property connecting two classes via an object property chain.  
- **Pattern 2 (Subclass):** Proposes a new subclass of an existing class based on relationships.  

                
#### **Models**  
The model selection consists of avaiable OpenAI models and models served through Ollama running on VŠE Infrastructure. The last used Ollama model will stay loaded in memory for an hour, given that nobody else loads a different model in the meantime.
                    
##### **Model Parameters**
Controls the generation parameters of the LLM.
Various parameter combinations can cause errors.
                
| Parameter          | Description                                                                |  Model availability       |
|--------------------|----------------------------------------------------------------------------|----------------------------|  
| `temperature` | Controls randomness in token selection. A value of 0.0 produces deterministic output, while higher values (up to 2.0) increase creativity and unpredictability. Values above 1.0 may lead to less coherent outputs. Hard-coded to 1.0 for the o1-preview model. | All models |  
| `top_p`            | Nucleus sampling parameter: selects tokens with cumulative probability up to the specified value. Reduces randomness compared to high temperature. Commonly used with lower temperature (e.g., 0.0-0.5) for coherent outputs.                | All models                 |  
| `frequency_penalty`| Penalizes repeated tokens based on their frequency in the generated text. Higher values (0.5-2.0) reduce repetition.                         | OpenAI models              |  
| `presence_penalty` | Penalizes tokens that have appeared at least once, encouraging the model to introduce new topics. Aids in reducing redundancy.                     | OpenAI models              |  
| `repeat_penalty`   | Ollama-specific penalty that reduces the likelihood of repeating tokens by scaling their logit scores. Higher values (e.g., 1.1-1.5) discourage immediate repetition.                           |  Ollama models  |        


&nbsp;
#### **Pattern 1: Object Property Chain Shortcut**  
  
| Input Field             | Description                                                                 |  
|-------------------|-----------------------------------------------------------------------------|  
| `Class A label`     | Label of the starting class (e.g., "corpus_part").                          |  
| `Property p label`  | Property linking Class A to Class B (e.g., "genre").                        |  
| `Class B label`     | Label of the intermediate class (e.g., "Genre").                            |  
| `Property r label`  | Property linking Class B to Class C (e.g., "has sub-genre").                |  
| `Class C label`     | Label of the target class (e.g., "Music Genre").                            |        


\\
Output: Property Name is the suggested direct property (e.g., "has_music_genre").\\
CSV Columns must include: `A_label`, `p_label`, `B_label`, `r_label`, `C_label`, `Property`.
                
#### **Pattern 2: Subclass Enrichment**  

| Input Field             | Description                                                                 |  
|-------------------|-----------------------------------------------------------------------------|  
| `Class A label`     | Label of the parent class (e.g., "System").                                 |  
| `Property p label`  | Property linking Class A to Class B (e.g., "has component").                |  
| `Class B label`     | Label of the intermediate class (e.g., "Component").                        |  
| `Class C label`     | Label of the subclass of B (e.g., "Storage Device").                        |

\\
Output: Class Name is the suggested subclass name (e.g., "StorageSystem").\\
CSV columns must include: `A_label`, `p_label`, `B_label`, `C_label`, `Subclass`.

                
#### **Session History**  

Session data is stored in the browser's local storage, never on the server.
                
- **Export CSV:** Download all session interactions.  
- **Clear History:** Remove all stored session data.  
- **Load Past Sessions:** Reuse previous inputs/results by clicking the "Load" button next to an entry.  
'''
    st.markdown(md)


if __name__ == "__main__":
    main()