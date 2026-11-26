import csv
import json
import os
from typing import Literal
import requests
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# --- 1. READ SECRETS FROM PROTECTED PROCESS RAM ---
OPENALEX_API_KEY = os.environ["OPENALEX_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

DATA_DIR = "data"
LEDGER_PATH = os.path.join(DATA_DIR, "processed_dois.json")
CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.csv")

# --- 2. DEFINE STRICT OUTPUT CONTRACT (PYDANTIC SCHEMA) ---
class PaperDecision(BaseModel):
    index: int = Field(description="Paper number from the input list")
    keep: bool = Field(
        description=(
            "True ONLY if primary wet-lab experimental REE leaching/extraction "
            "from secondary waste with quantitative yields/parameters. "
            "False for review papers, geology/characterization-only, pure DFT/simulation, "
            "or nanomaterial synthesis."
        )
    )
    waste_type: Literal[
        "NdFeB_Magnet",
        "Coal_Fly_Ash",
        "Red_Mud",
        "Mine_Tailings",
        "Other_Secondary_Waste",
        "Not_Applicable",
    ]
    reason: str = Field(description="One concise sentence explaining why kept or rejected")


class BatchTriageResult(BaseModel):
    decisions: list[PaperDecision]


# --- 3. RECONSTRUCT OPENALEX INVERTED INDEX IN O(N) TIME ---
def reconstruct_abstract(inv_index: dict | None) -> str:
    if not inv_index:
        return ""
    max_pos = max((pos for positions in inv_index.values() for pos in positions), default=-1)
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for word, positions in inv_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)


# --- 4. FETCH FRESH OPEN-ACCESS PAPERS FROM OPENALEX ---
def fetch_openalex_papers(limit: int = 20) -> tuple[list[dict], float]:
    url = "https://api.openalex.org/works"
    params = {
        "search.title_and_abstract": (
            '("rare earth" OR neodymium OR lanthanide) AND leaching '
            'AND ("fly ash" OR "red mud" OR NdFeB OR magnet OR tailings OR "e-waste")'
        ),
        "filter": "is_oa:true,has_abstract:true,has_content.pdf:true,type:article,publication_year:>2017",
        "select": "id,doi,title,publication_year,best_oa_location,abstract_inverted_index",
        "sort": "publication_date:desc",
        "per_page": limit,
        "api_key": OPENALEX_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        print("OpenAlex Error:", resp.status_code, resp.text)
        resp.raise_for_status()
    payload = resp.json()
    cost = payload.get("meta", {}).get("cost_usd", 0.0)
    return payload.get("results", []), cost


# --- 5. BATCH TRIAGE ALL NEW PAPERS IN 1 SINGLE GEMINI CALL ---
def batch_triage_with_gemini(papers: list[dict]) -> list[PaperDecision]:
    client = genai.Client(api_key=GEMINI_API_KEY)

    formatted_items = []
    for i, p in enumerate(papers):
        formatted_items.append(
            f"[{i}] TITLE: {p['title']}\nABSTRACT: {p['abstract'][:1500]}"
        )

    prompt = (
        "You are a senior hydrometallurgy Principal Investigator filtering papers "
        "for a machine learning dataset predicting Rare Earth Element (REE) leaching "
        "efficiency (%) from secondary wastes.\n\n"
        "Classify EVERY paper below. Keep ONLY primary experimental wet-lab leaching "
        "studies reporting measurable extraction conditions and yields.\n\n"
        + "\n\n---\n\n".join(formatted_items)
    )

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchTriageResult,
            temperature=0.0,
        ),
    )

    parsed = response.parsed or BatchTriageResult.model_validate_json(response.text)
    return parsed.decisions


# --- 6. MAIN PIPELINE EXECUTION ---
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load deduplication ledger into a Hash Table (Python set)
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            processed_ids: set[str] = set(json.load(f))
    else:
        processed_ids = set()

    raw_works, query_cost = fetch_openalex_papers(limit=20)
    print(f"OpenAlex returned {len(raw_works)} works (API cost: ${query_cost:.5f} of $1.00 daily free budget)")

    new_papers = []
    for w in raw_works:
        work_id = w.get("id", "").split("/")[-1]  # e.g., W4392145873
        unique_key = (w.get("doi") or work_id).lower().strip()
        if not unique_key or unique_key in processed_ids:
            continue

        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
        if len(abstract) < 80:
            continue

        oa_loc = w.get("best_oa_location") or {}
        pdf_url = oa_loc.get("pdf_url") or ""

        new_papers.append({
            "unique_key": unique_key,
            "openalex_id": work_id,
            "doi": w.get("doi") or "",
            "year": w.get("publication_year") or "",
            "title": (w.get("title") or "").replace("\n", " ").strip(),
            "pdf_url": pdf_url,
            "abstract": abstract,
        })

    if not new_papers:
        print("No unseen papers in this batch. Deduplication ledger worked!")
        return

    print(f"Sending {len(new_papers)} unseen papers to Gemini 3.5 Flash-Lite in 1 single batch call...")
    decisions = batch_triage_with_gemini(new_papers)

    file_exists = os.path.exists(CANDIDATES_PATH)
    kept_count = 0

    with open(CANDIDATES_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["openalex_id", "doi", "year", "waste_type", "title", "pdf_url", "reason"])

        for d in decisions:
            if d.index < 0 or d.index >= len(new_papers):
                continue
            paper = new_papers[d.index]
            processed_ids.add(paper["unique_key"])

            status = "KEEP  " if d.keep else "REJECT"
            print(f"[{status}] ({d.waste_type}) {paper['title'][:75]}... -> {d.reason}")

            if d.keep:
                kept_count += 1
                writer.writerow([
                    paper["openalex_id"],
                    paper["doi"],
                    paper["year"],
                    d.waste_type,
                    paper["title"],
                    paper["pdf_url"],
                    d.reason,
                ])

    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(processed_ids), f, indent=2)

    print(f"\nDONE: Kept {kept_count}/{len(new_papers)} primary leaching papers. Total seen ledger: {len(processed_ids)}.")


if __name__ == "__main__":
    main()
