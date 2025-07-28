import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_text_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("blocks")
        for b in blocks:
            text = b[4].strip()
            if len(text.split()) >= 20:
                chunks.append({
                    "page": page_num,
                    "text": text,
                    "title": text.split('\n')[0]
                })
    return chunks

def get_embeddings(texts):
    return model.encode(texts, show_progress_bar=False)

def rank_chunks(chunks, chunk_embeddings, query_embedding):
    sims = cosine_similarity([query_embedding], chunk_embeddings)[0]
    for idx, sim in enumerate(sims):
        chunks[idx]["score"] = sim
    return sorted(chunks, key=lambda x: x["score"], reverse=True)

def summarize_text(text, num_sentences=2):
    sentences = text.replace("\n", " ").split(". ")
    summary = ". ".join(sentences[:num_sentences])
    return summary.strip() + "." if summary else ""
