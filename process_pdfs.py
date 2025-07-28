import os
import json
import re
import pdfplumber
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# Use recommended cache env variable
os.environ["HF_HOME"] = "/app/cache"

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

INPUT_DIR = "input"
OUTPUT_DIR = "output"
CFG_PATH = os.path.join(INPUT_DIR, "challenge1b_input.json")

def clean_section_title(title):
    title = re.sub(r'[^A-Za-z0-9\s\-–—:&]', '', title).strip()
    title = re.sub(r'\s+', ' ', title)
    return title.title()

def extract_sections(pdf_path):
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            lines = text.split("\n")
            for line in lines:
                if re.match(r"^[A-Z][A-Za-z0-9 ,:&\-–—]{3,}$", line.strip()) and len(line.strip().split()) <= 10:
                    title = clean_section_title(line.strip())
                    sections.append({
                        "title": title,
                        "text": text.strip(),
                        "page": i + 1
                    })
                    break
    return sections

def main():
    with open(CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    persona = cfg["persona"]["role"]
    task = cfg["job_to_be_done"]["task"]
    keywords = cfg.get("keywords", [])

    all_secs = []
    all_pages = []

    for doc in cfg["documents"]:
        path = os.path.join(INPUT_DIR, doc["filename"])
        sections = extract_sections(path)
        for s in sections:
            s["doc"] = doc["filename"]
            all_secs.append(s)

        with pdfplumber.open(path) as pdf:
            all_pages.append({
                "document": doc["filename"],
                "total_pages": len(pdf.pages)
            })

    generic_titles = ["introduction", "summary", "conclusion", "overview"]
    for s in all_secs:
        score = 0
        title = s["title"].lower()
        for kw in keywords:
            if kw.lower() in title:
                score += 2
            if kw.lower() in s["text"].lower():
                score += 1
        emb_section = model.encode(s["text"], convert_to_tensor=True)
        emb_task = model.encode(task, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb_section, emb_task).item()
        score += sim * 5
        s["final_score"] = score

    top5 = [
        s for s in sorted(all_secs, key=lambda s: -s["final_score"])
        if s["title"].strip().lower() not in generic_titles
    ][:5]

    output = {
        "metadata": {
            "input_documents": [os.path.basename(d["filename"]) for d in cfg["documents"]],
            "persona": persona,
            "job_to_be_done": task,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": s["doc"],
                "section_title": s["title"],
                "importance_rank": i + 1,
                "page_number": s["page"]
            } for i, s in enumerate(top5)
        ],
        "subsection_analysis": [
            {
                "document": s["doc"],
                "refined_text": s["text"],
                "page_number": s["page"]
            } for s in top5
        ],
        "all_pages_processed": all_pages
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "challenge1b_output.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
