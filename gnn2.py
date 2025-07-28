#!/usr/bin/env python3
# import os

# # Force model cache to a local writable path
# os.environ['TRANSFORMERS_CACHE'] = './model_cache'
# os.environ['SENTENCE_TRANSFORMERS_HOME'] = './model_cache'
# os.environ['HF_HOME'] = './model_cache'


import os
import json
import re
import datetime
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- CONFIG ----------------------------------------------------
INPUT_DIR = "."
TOP_K = 5

# --- LOAD MODELS -----------------------------------------------
# Bi-encoder for section ranking
bi_model = SentenceTransformer('all-MiniLM-L6-v2')
# T5 model for generating summaries/answers
t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
t5_model = AutoModelForSeq2SeqLM.from_pretrained('t5-base', device_map='auto')

def preprocess_text(text: str) -> str:
    return ' '.join(text.split()).strip()

def extract_sections(text: str) -> list:
    """Split text into sections by blank lines or headings."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 50]
    sections = []
    for para in paras:
        title = para.split('\n')[0][:60]
        sections.append({'title': title, 'text': para})
    return sections

# --- SCORING FUNCTIONS -----------------------------------------
def rank_sections(sections: list, context: str) -> list:
    """Return top-K sections by cosine similarity with context."""
    texts = [s['text'] for s in sections]
    emb_sections = bi_model.encode(texts, convert_to_tensor=True)
    emb_context = bi_model.encode(context, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(emb_context, emb_sections)[0].cpu().numpy()
    top_idx = np.argsort(sims)[::-1][:TOP_K]
    ranked = []
    for rank, i in enumerate(top_idx, start=1):
        sec = sections[i]
        ranked.append({
            'document': sec.get('doc', ''),
            'section_title': sec['title'],
            'page_number': sec.get('page', None),
            'importance_rank': rank,
            'text': sec['text']
        })
    return ranked

# --- GENERATION FUNCTIONS --------------------------------------
def summarize_section(text: str, max_len: int = 64) -> str:
    prompt = "summarize: " + text.replace('\n', ' ')
    inputs = t5_tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(t5_model.device)
    out = t5_model.generate(**inputs, max_new_tokens=max_len)
    return t5_tokenizer.decode(out[0], skip_special_tokens=True)

# --- PROCESS COLLECTION ----------------------------------------
def process_collection(folder: Path):
    input_path = folder / 'challenge1b_input.json'
    output_path = folder / 'challenge1b_output.json'
    pdf_dir = folder / 'PDFs'
    if not input_path.exists() or not pdf_dir.is_dir():
        return

    config = json.loads(input_path.read_text())
    persona = config['persona']['role']
    task = config['job_to_be_done']['task']
    context = f"As a {persona}, your task is: {task}."

    all_sections = []
    # extract sections from each PDF
    for doc in config.get('documents', []):
        pdf_path = pdf_dir / doc['filename']
        if not pdf_path.exists():
            continue
        reader = PyPDF2.PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ''
            sections = extract_sections(text)
            for sec in sections:
                sec['doc'] = doc['filename']
                sec['page'] = page_num
                all_sections.append(sec)

    if not all_sections:
        return

    # rank and select top sections
    ranked = rank_sections(all_sections, context)

    # build output
    metadata = {
        'collection': folder.name,
        'persona': persona,
        'job_to_be_done': task,
        'processed_at': datetime.datetime.utcnow().isoformat(),
        'model': 'all-MiniLM-L6-v2 + t5-base'
    }
    extracted_sections = []
    subsection_analysis = []

    for sec in ranked:
        extracted_sections.append({
            'document': sec['document'],
            'section_title': sec['section_title'],
            'page_number': sec['page_number'],
            'importance_rank': sec['importance_rank']
        })
        summary = summarize_section(sec['text'])
        subsection_analysis.append({
            'document': sec['document'],
            'page_number': sec['page_number'],
            'refined_text': summary
        })

    output = {
        'metadata': metadata,
        'extracted_sections': extracted_sections,
        'subsection_analysis': subsection_analysis
    }
    output_path.write_text(json.dumps(output, indent=2), encoding='utf-8')

# --- MAIN ------------------------------------------------------
def main():
    for folder in Path(INPUT_DIR).iterdir():
        if folder.is_dir():
            process_collection(folder)
            print(f"Processed {folder.name}")

if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
