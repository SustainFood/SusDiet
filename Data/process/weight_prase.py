import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

import pandas as pd
from openai import OpenAI

INPUT_CSV = "ingredients.csv"
OUTPUT_CSV = "ingredients_with_weight.csv"

MODEL = ""
BATCH_SIZE = 8
CHUNK_SIZE = 2_000
SLEEP_BETWEEN_BATCH = 0.05

client = OpenAI(
    api_key="",
    base_url=""
)

PROMPT = """You are a food ingredient parser.
Given an ingredient phrase, extract the ingredient name and its weight in grams.

Return JSON exactly:
{{"ingredient": "<name or 'no ingredient'>", "weight_g": <number or 0>}}

Rules:
- If no ingredient is found, use ingredient = "no ingredient" and weight_g = 0.
- weight_g should be a number (grams). If unsure, use -1.
- Ingredient name should not include quantity or unit or other words, just the food name.

Phrase: "{phrase}"
"""


def call_llm(phrase: str) -> Tuple[str, float]:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=64,
            messages=[{"role": "user", "content": PROMPT.format(phrase=phrase)}],
        )
        content = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] LLM call failed for '{phrase}': {repr(e)}")
        return "no ingredient", 0.0

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print(f"[WARN] JSON parse failed. Raw: {content!r}")
        return "no ingredient", 0.0

    ingredient = ""
    weight = 0.0

    if isinstance(data, dict):
        ingredient = str(data.get("ingredient", "")).strip() or "no ingredient"
        try:
            weight = float(data.get("weight_g", 0)) if data.get("weight_g", 0) not in ("", None) else 0.0
        except Exception:
            weight = 0.0
    else:
        ingredient = "no ingredient"
        weight = 0.0

    return ingredient, weight


def main():
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    reader = pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE)
    wrote_header = False
    total = 0

    for chunk in reader:
        if "weight_g" not in chunk.columns:
            chunk["weight_g"] = pd.NA
        if "ingredient_parsed" not in chunk.columns:
            chunk["ingredient_parsed"] = pd.NA

        targets = chunk.index[chunk["phrase"].map(lambda x: isinstance(x, str))]
        if not len(targets):
            chunk.to_csv(OUTPUT_CSV, mode="a", index=False, header=not wrote_header)
            wrote_header = True
            continue

        for start in range(0, len(targets), BATCH_SIZE):
            batch = targets[start : start + BATCH_SIZE]
            with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                futures = {executor.submit(call_llm, chunk.at[i, "phrase"]): i for i in batch}
                for future in as_completed(futures):
                    idx = futures[future]
                    ing, w = future.result()
                    chunk.at[idx, "ingredient_parsed"] = ing
                    chunk.at[idx, "weight_g"] = w
                    print(f"{idx}: {chunk.at[idx, 'phrase']} -> {ing}, {w}")
                    total += 1

            time.sleep(SLEEP_BETWEEN_BATCH)

        chunk.to_csv(OUTPUT_CSV, mode="a", index=False, header=not wrote_header)
        wrote_header = True

    print(f"Done. Saved to {OUTPUT_CSV}, processed {total} phrases.")


if __name__ == "__main__":
    main()
