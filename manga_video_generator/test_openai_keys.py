import os
import requests

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
keys = []
with open(ENV_PATH) as f:
    for line in f:
        if line.strip().startswith("OPENAI_KEYS_POOL"):
            pool = line.strip().split("=", 1)[1]
            for item in pool.split(","):
                if ":" in item:
                    k, owner = item.split(":", 1)
                    keys.append({"key": k.strip(), "owner": owner.strip()})

print(f"Found {len(keys)} OpenAI keys to test.\n")

for i, entry in enumerate(keys):
    key = entry["key"]
    owner = entry["owner"]
    masked = f"{key[:8]}...{key[-4:]}"
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 5,
            },
            timeout=15,
        )
        if r.status_code == 200:
            reply = r.json()["choices"][0]["message"]["content"].strip()
            print(f"[OK]    Key {i+1} ({owner})  {masked}  ->  '{reply}'")
        else:
            body = r.json()
            err = body.get("error", {})
            code = err.get("code", "")
            msg = err.get("message", "")[:100]
            print(f"[FAIL]  Key {i+1} ({owner})  {masked}  ->  HTTP {r.status_code}  {code}: {msg}")
    except Exception as e:
        print(f"[ERROR] Key {i+1} ({owner})  {masked}  ->  {e}")

print("\nDone.")