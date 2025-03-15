def load_text_file(filepath: str) -> str:
    """Loads a text file and returns its content as a string."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""