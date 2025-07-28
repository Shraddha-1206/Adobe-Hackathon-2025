import os
import json
import datetime
from utils import extract_text_chunks, get_embeddings, rank_chunks, summarize_text

def is_valid_title(title):
    if not title:
        return False
    title = title.strip().lower()
    if ":" in title or "section" in title or "untitled" in title:
        return False
    words = title.split()
    return 2 <= len(words) <= 8

def extract_section_title(text):
    lines = [l.strip("• ").strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        if line and line[0].isupper():
            candidate = line[:80].strip()
            if is_valid_title(candidate):
                return candidate
    return None

def summarize_text_safe(text):
    words = text.split()
    if len(words) < 50 or text.strip().startswith("•"):
        return text.strip("• \n")
    return summarize_text(text).strip("• \n")

INPUT_BASE = "input"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

collections = [d for d in os.listdir(INPUT_BASE) if os.path.isdir(os.path.join(INPUT_BASE, d))]

for collection in collections:
    print(f"Processing collection: {collection}")
    input_dir = os.path.join(INPUT_BASE, collection)
    persona_path = os.path.join(input_dir, "persona_job.json")

    if not os.path.exists(persona_path):
        print(f"⚠️ persona_job.json missing in {collection}")
        continue

    with open(persona_path, "r", encoding="utf-8") as f:
        persona_info = json.load(f)

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    query = persona_info["job_to_be_done"]["task"]
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    query_embedding = get_embeddings([query])[0]

    all_valid_chunks = []
    all_other_chunks = []

    for filename in pdf_files:
        filepath = os.path.join(input_dir, filename)
        print(f" Processing: {filename}")

        chunks = extract_text_chunks(filepath)
        if not chunks:
            continue

        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = get_embeddings(chunk_texts)
        ranked_chunks = rank_chunks(chunks, chunk_embeddings, query_embedding)

        for chunk in ranked_chunks:
            chunk["document"] = filename
            title = extract_section_title(chunk["text"])
            if title:
                chunk["title"] = title
                all_valid_chunks.append(chunk)
            else:
                all_other_chunks.append(chunk)

    # Sort all valid chunks by relevance
    sorted_valid = sorted(all_valid_chunks, key=lambda x: x["score"], reverse=True)

       # Take top 5 unique documents for extracted sections
    seen_docs = set()
    top_extracted = []
    for chunk in sorted_valid:
        if chunk["document"] not in seen_docs:
            seen_docs.add(chunk["document"])
            top_extracted.append(chunk)
        if len(top_extracted) == 5:
            break

    # If less than 5, allow duplicate documents to fill
    if len(top_extracted) < 5:
        remaining = [c for c in sorted_valid if c not in top_extracted]
        for chunk in remaining:
            top_extracted.append(chunk)
            if len(top_extracted) == 5:
                break

    # Fill 5 subsection_analysis with highest scored remaining chunks (excluding used pages)
    used_pages = {(c["document"], c["page"]) for c in top_extracted}
    subsection_pool = [c for c in all_valid_chunks + all_other_chunks if (c["document"], c["page"]) not in used_pages]
    top_subsections = sorted(subsection_pool, key=lambda x: x["score"], reverse=True)[:5]

    # Truncate both lists to exactly 5 entries
    top_extracted = top_extracted[:5]
    top_subsections = top_subsections[:5]

    output = {
        "metadata": {
            "input_documents": pdf_files,
            "persona": persona_info["persona"]["role"],
            "job_to_be_done": persona_info["job_to_be_done"]["task"],
            "processing_timestamp": timestamp
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for idx, sec in enumerate(top_extracted):
        output["extracted_sections"].append({
            "document": sec["document"],
            "section_title": sec["title"],
            "importance_rank": idx + 1,
            "page_number": sec["page"]
        })

    for sec in top_subsections:
        output["subsection_analysis"].append({
            "document": sec["document"],
            "refined_text": summarize_text_safe(sec["text"]),
            "page_number": sec["page"]
        })

    output_path = os.path.join(OUTPUT_DIR, f"output_{collection.replace(' ', '_').lower()}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Output saved: {output_path}")