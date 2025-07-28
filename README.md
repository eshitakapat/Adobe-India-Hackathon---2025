# Adobe-India-Hackathon---2025
# 🤖 Adobe Hackathon – Round 1B: Persona-Driven Document Intelligence

## 🧩 Objective

Design an intelligent system that extracts and ranks the most relevant sections and subsections from a collection of PDFs, based on:
- A defined Persona
- A specific Job-to-be-done

---

## 🧠 Problem Statement

Given:
- 3–10 PDFs from any domain
- A persona (e.g., Student, Researcher, Analyst)
- A concrete task related to the persona

Your system must:
1. Understand the persona’s goal
2. Identify and rank relevant sections across all documents
3. Provide meaningful subsection-level insights

---

## 🛠 Approach

Our pipeline consists of the following steps:

### 1. 📥 Input Parsing
- Accepts multiple PDFs and a JSON description of the persona and job-to-be-done
- Extracts raw text and page-wise structure

### 2. 🧠 Semantic Understanding
- Embeds persona and job definition using transformer-based models (e.g., DistilBERT/Sentence-BERT)
- Embeds each section from all PDFs
- Computes cosine similarity between job-to-be-done and document sections

### 3. 🔍 Section Selection
- Sections are ranked by similarity scores
- Top N sections are selected with importance_rank

### 4. 🪄 Subsection Extraction
- Each top section is further analyzed for key insights using:
  - Sentence scoring
  - Text summarization (if applicable)

---

## 📚 Libraries & Models Used

- PyMuPDF or pdfplumber – PDF parsing
- sentence-transformers – Semantic similarity
- nltk, spacy – NLP preprocessing
- Optional: lightweight transformer models (≤ 1GB)

✅ Entire pipeline runs offline  
✅ Fully CPU-compatible (no GPU required)  
✅ Model size kept below 1GB

---

## 📦 Output Format

Example JSON output:

### 1. Metadata
```json
{
  "documents": ["doc1.pdf", "doc2.pdf"],
  "persona": "Investment Analyst",
  "job_to_be_done": "Analyze R&D spending trends",
  "timestamp": "2025-07-28T14:33:00Z"
}