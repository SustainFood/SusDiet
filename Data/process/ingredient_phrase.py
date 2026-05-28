import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI


CSV_PATH = 'ingredients.csv'

MODEL = ''

START_ROW = 0
END_ROW = None
CHUNK_SIZE = 5000
BATCH_SIZE = 10
SAVE_EVERY = 1000 
SLEEP_BETWEEN_BATCH = 0.1

client = OpenAI(
    api_key="",
    base_url=""
)

# =========================
# Prompt
# =========================
PROMPT = """Parse the ingredient phrase into JSON.

Phrase: "{phrase}"

Return EXACTLY one JSON object in one line:
{{"quantity":"","unit":"","ingredient_name":""}}

Rules:
- quantity: numeric only (e.g. "1", "0.5"); empty if missing
- unit: unit word only; empty if missing
- ingredient_name: only the name of the ingredients, no quantity, no unit, other descriptive words for ingredients.
- If unsure, return empty strings. No explanation, no markdown, no extra text
"""


def parse_phrase(phrase: str):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=128,
            messages=[{"role": "user", "content": PROMPT.format(phrase=phrase)}],
        )
        # print(PROMPT.format(phrase=phrase))
        # print(json.loads(resp.choices[0].message.content.strip()))
        return json.loads(resp.choices[0].message.content.strip())
    except Exception:
        return None



def is_empty(x) -> bool:
    return pd.isna(x) or str(x).strip() == ''

def ensure_columns(chunk: pd.DataFrame) -> None:
    for col in ['quantity', 'unit', 'ingredient_name']:
        if col not in chunk.columns:
            chunk[col] = pd.NA

start_idx = max(0, START_ROW)
end_idx = None if END_ROW is None else END_ROW + 1

tmp_path = f'{CSV_PATH}.tmp'
is_first_chunk = True



processed = 0

reader = pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE)
for chunk in reader:
    ensure_columns(chunk)

    idx_vals = chunk.index
    in_range = idx_vals >= start_idx
    if end_idx is not None:
        in_range &= idx_vals < end_idx

    phrase_is_str = chunk['phrase'].map(lambda x: isinstance(x, str))
    needs = chunk['ingredient_name'].isna() | (chunk['ingredient_name'].astype(str).str.strip() == '')
    indices = chunk.index[in_range & phrase_is_str & needs].tolist()

    for batch_start in range(0, len(indices), BATCH_SIZE):
        batch = indices[batch_start: batch_start + BATCH_SIZE]

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = {
                executor.submit(parse_phrase, chunk.at[idx, 'phrase']): idx
                for idx in batch
            }

            for future in as_completed(futures):
                idx = futures[future]
                phrase = chunk.at[idx, 'phrase']
                result = future.result()

                if result:
                    if isinstance(result, list) and len(result) > 0:
                        result = result[0]
                    if isinstance(result, dict):
                        chunk.at[idx, 'quantity'] = result.get('quantity', '')
                        chunk.at[idx, 'unit'] = result.get('unit', '')
                        chunk.at[idx, 'ingredient_name'] = result.get('ingredient_name', '')
                        print(f"row {idx}: {chunk.at[idx, 'quantity']}; {chunk.at[idx, 'unit']}; {chunk.at[idx, 'ingredient_name']}")
                    else:
                        print(f'row {idx}: FAILED -> {phrase}')
                else:
                    print(f'row {idx}: FAILED -> {phrase}')

                processed += 1

        if processed % SAVE_EVERY == 0:
            print(f'Checkpoint processed at {processed} rows (written to temp file)')

        time.sleep(SLEEP_BETWEEN_BATCH)


    chunk.to_csv(tmp_path, mode='w' if is_first_chunk else 'a', index=False, header=is_first_chunk)
    is_first_chunk = False

if is_first_chunk:
    raise RuntimeError('No data read from CSV. Check CSV_PATH or file content.')

os.replace(tmp_path, CSV_PATH)
print('Done. All data saved.')
