# Public API

`initialize(folder_path)` scans one raw folder and prepares an in-memory structural catalog. `question(text)` returns a plain string. Calling `question` before initialization raises `RuntimeError`. A reachable OpenAI-compatible local model with strict JSON Schema support is required.
