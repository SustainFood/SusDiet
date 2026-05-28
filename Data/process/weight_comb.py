import os, json, time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

SRC = "ingredient_unit_combo_with_weight.xlsx"
OUT = "ingredient_unit_combo_with_weight.xlsx"

MODEL = ""
BATCH_SIZE = 32
SLEEP = 0.05
START_ROW = 12000
END_ROW = None 
client = OpenAI(
    api_key="",
    base_url=""
)

PROMPT = """You are estimating ingredient combo weights.
Given a combo like "1 cup tomato" or "1 tsp salt", return JSON exactly:
{{"weight_g": <number in grams>}}
If unsure, return {{"weight_g": -1}}.
Combo: "{combo}"
"""

def call_llm(combo: str) -> float:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=32,
            messages=[{"role": "user", "content": PROMPT.format(combo=combo)}],
        )
        content = resp.choices[0].message.content.strip()
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return float(data.get("weight_g", 0) or 0)
        except json.JSONDecodeError:
            try:
                return float(content)
            except Exception:
                print(f"[WARN] JSON parse failed for {combo!r}, raw: {content!r}")
                return 0.0
    except Exception as e:
        print(f"[WARN] {combo} -> {e}")
        return 0.0

df = pd.read_excel(SRC)
if "weight_g" not in df.columns:
    df["weight_g"] = pd.NA

df.index = range(len(df))

combos = df["combo"].astype(str).str.strip()
in_range = df.index >= START_ROW
if END_ROW is not None:
    in_range &= df.index <= END_ROW
weight_blank = df["weight_g"].isna() | (df["weight_g"].astype(str).str.strip() == "")
pending_idx = df.index[weight_blank & in_range]

print(f"Total rows: {len(df)}, to process in range: {len(pending_idx)}")

for start in range(0, len(pending_idx), BATCH_SIZE):
    batch_idx = pending_idx[start : start + BATCH_SIZE]
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as ex:
        fut = {ex.submit(call_llm, combos[i]): i for i in batch_idx}
        for f in as_completed(fut):
            idx = fut[f]
            w = f.result()
            df.at[idx, "weight_g"] = w
            print(f"{idx}: {combos[idx]} -> {w}")
    time.sleep(SLEEP)

df.to_excel(OUT, index=False)
print(f"Done. Saved to {OUT}")
