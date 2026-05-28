import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import pandas as pd
from openai import OpenAI
from rapidfuzz import fuzz


ING_PATH = "ingredients.csv"
ING_COL = "ingredient"
ENV_XLS = "source.xls"
ENV_SHEET = "LCA"
ENV_COL = "Product"

MODEL = os.getenv("OPENAI_MODEL", "")
BATCH_SIZE = 16
SAVE_EVERY = 240
START_ROW = 0
END_ROW = None


OUT_PATH = "ingredients_with_matches.csv"

ANIMAL_XLS = "animal.xlsx"
ANIMAL_SHEET = "animal_good"
ANIMAL_COL = "animal.prod.code"
GHG_SHEET = "GHG"
GHG_COL = "Entity"

PRICE_XLSX = "prices_csv.xlsx"
PRICE_COL = "food_description"

AISLE_SHEET = "total_env_nutri_Score"
AISLE_COL = "Aisle"
CARBON_SHEET = "Carbon_footprint"
CARBON_COL = "Food commodity ITEM"
WATER_SHEET = "Water_footprint"
WATER_COL = "Food commodity ITEM"




client = OpenAI(
    api_key="",
    base_url=""
)


def load_options(path: str, sheet: str, col: str) -> List[str]:
    df = pd.read_excel(path, sheet_name=sheet, usecols=[col])
    opts = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"": pd.NA})
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(opts)


def load_options_excel(path: str, col: str) -> List[str]:
    df = pd.read_excel(path, usecols=[col])
    opts = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace({"": pd.NA})
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(opts)


def _top_k(query: str, options: List[str], k: int = 400) -> List[str]:
    if not options:
        return []
    q = str(query).lower()
    if fuzz:
        scored = [(opt, fuzz.token_set_ratio(q, str(opt).lower())) for opt in options]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:k]]
    first = q.split()[0] if q.split() else ""
    filtered = [opt for opt in options if first and first in str(opt).lower()]
    return filtered[:k] if filtered else options[:k]


def call_llm(
    ing: str,
    products: List[str],
    animals: List[str],
    foods: List[str],
    ghg: List[str],
    aisles: List[str],
    carbons: List[str],
    waters: List[str],
) -> Tuple[str, str, str, str, str, str, str]:
    products = _top_k(ing, products)
    animals = _top_k(ing, animals)
    foods = _top_k(ing, foods)
    ghg = _top_k(ing, ghg)
    aisles = _top_k(ing, aisles)
    carbons = _top_k(ing, carbons)
    waters = _top_k(ing, waters)

    prod_text = "\n".join(f"- {p}" for p in products)
    animal_text = "\n".join(f"- {a}" for a in animals)
    food_text = "\n".join(f"- {f}" for f in foods)
    ghg_text = "\n".join(f"- {g}" for g in ghg)
    aisle_text = "\n".join(f"- {a}" for a in aisles)
    carbon_text = "\n".join(f"- {c}" for c in carbons)
    water_text = "\n".join(f"- {w}" for w in waters)

    prompt = f"""You are a classifier. Given an ingredient text, choose the closest match from the provided lists.

Ingredient: "{ing}"

Choose one from Product options (or "none"):
{prod_text}

Choose one from animal.prod.code options (or "none"):
{animal_text}

Choose one from food_description options (or "none"):
{food_text}

Choose one from GHG emissions per kilogram options (or "none"):
{ghg_text}

Choose one from Aisle options (or "none"):
{aisle_text}

Choose one from Carbon_footprint Food commodity ITEM options (or "none"):
{carbon_text}

Choose one from Water_footprint Food commodity ITEM options (or "none"):
{water_text}

Return JSON exactly:
{{
  "product": "<value from list or none>",
  "animal_code": "<value from list or none>",
  "food_description": "<value from list or none>",
  "ghg_emission": "<value from list or none>",
  "aisle": "<value from list or none>",
  "carbon_item": "<value from list or none>",
  "water_item": "<value from list or none>"
}}
"""
    # print(prompt)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            # max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] LLM call failed for '{ing}': {e}")
        return "", "", "", "", "", "", ""

    try:
        data = json.loads(content)
        prod = str(data.get("product", "")).strip()
        animal = str(data.get("animal_code", "")).strip()
        food = str(data.get("food_description", "")).strip()
        ghg_val = str(data.get("ghg_emission", "")).strip()
        aisle_val = str(data.get("aisle", "")).strip()
        carbon_val = str(data.get("carbon_item", "")).strip()
        water_val = str(data.get("water_item", "")).strip()
        return prod, animal, food, ghg_val, aisle_val, carbon_val, water_val
    except Exception:
        print(f"[WARN] JSON parse failed for '{ing}', raw: {content!r}")
        return "", "", "", "", "", "", ""


def main():
    products = load_options(ENV_XLS, ENV_SHEET, ENV_COL)
    animals = load_options(ANIMAL_XLS, ANIMAL_SHEET, ANIMAL_COL)
    foods = load_options_excel(PRICE_XLSX, PRICE_COL)
    ghg = load_options(ANIMAL_XLS, GHG_SHEET, GHG_COL)
    aisles = load_options(ENV_XLS, AISLE_SHEET, AISLE_COL)
    carbons = load_options(ENV_XLS, CARBON_SHEET, CARBON_COL)
    waters = load_options(ENV_XLS, WATER_SHEET, WATER_COL)

    df = pd.read_csv(ING_PATH)
    if ING_COL not in df.columns:
        df.rename(columns={df.columns[0]: ING_COL}, inplace=True)

    if "product_match" not in df.columns:
        df["product_match"] = ""
    if "animal_code_match" not in df.columns:
        df["animal_code_match"] = ""
    if "food_desc_match" not in df.columns:
        df["food_desc_match"] = ""
    if "ghg_match" not in df.columns:
        df["ghg_match"] = ""
    if "aisle_match" not in df.columns:
        df["aisle_match"] = ""
    if "carbon_item_match" not in df.columns:
        df["carbon_item_match"] = ""
    if "water_item_match" not in df.columns:
        df["water_item_match"] = ""

    df.index = range(len(df))
    in_range = df.index >= START_ROW
    if END_ROW is not None:
        in_range &= df.index <= END_ROW

    targets = df.index[in_range & (df[ING_COL].astype(str).str.strip() != "")]
    print(f"Total rows: {len(df)}, to process: {len(targets)}")

    processed = 0
    for start in range(0, len(targets), BATCH_SIZE):
        batch = targets[start : start + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as ex:
            futures = {
                ex.submit(call_llm, df.at[i, ING_COL], products, animals, foods, ghg, aisles, carbons, waters): i
                for i in batch
            }
            for future in as_completed(futures):
                idx = futures[future]
                prod, animal, food, ghg_val, aisle_val, carbon_val, water_val = future.result()
                df.at[idx, "product_match"] = prod
                df.at[idx, "animal_code_match"] = animal
                df.at[idx, "food_desc_match"] = food
                df.at[idx, "ghg_match"] = ghg_val
                df.at[idx, "aisle_match"] = aisle_val
                df.at[idx, "carbon_item_match"] = carbon_val
                df.at[idx, "water_item_match"] = water_val
                print(f"{idx}: {df.at[idx, ING_COL]} -> product: {prod}, animal: {animal}, food: {food}, ghg: {ghg_val}, aisle: {aisle_val}, carbon: {carbon_val}, water: {water_val}")
                processed += 1

        if processed and processed % SAVE_EVERY == 0:
            df.to_csv(OUT_PATH, index=False)
            print(f"Checkpoint saved at {processed} processed rows -> {OUT_PATH}")

    df.to_csv(OUT_PATH, index=False)
    print(f"Done. Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
