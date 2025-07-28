    #!/usr/bin/env python3
    import os
    import json
    import datetime
    from pathlib import Path
    import numpy as np
    import torch
    import pdfplumber
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv, global_mean_pool

    # --- CONFIG ----------------------------------------------------
    INPUT_DIR        = "."
    MODEL_NAME_SBERT = 'all-MiniLM-L6-v2'
    MODEL_NAME_GEN   = 'google/flan-t5-small'
    GNN_HIDDEN_DIM   = 128
    TOP_K            = 5

    # --- LOAD MODELS -----------------------------------------------
    sbert = SentenceTransformer(MODEL_NAME_SBERT)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_GEN)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME_GEN, quantization_config=bnb_config, device_map='auto'
    )

    # --- UTIL ------------------------------------------------------
    def refine_text(text: str, max_tokens: int = 64) -> str:
        prompt = f"Summarize this section in a concise, clear way:\n\n{text}\n\nSummary:"
        inputs = gen_tokenizer(prompt, return_tensors='pt').to(gen_model.device)
        out = gen_model.generate(**inputs, max_new_tokens=max_tokens)
        return gen_tokenizer.decode(out[0], skip_special_tokens=True)

    # --- GRAPH-RAG COMPONENTS --------------------------------------
    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.lin   = torch.nn.Linear(hidden_channels, 1)
        def forward(self, x, edge_index, batch):
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
            x_pool = global_mean_pool(x, batch)
            return torch.sigmoid(self.lin(x_pool)).view(-1)

    # --- PROCESS COLLECTION ----------------------------------------
    def process_collection(folder: Path):
        cfg = folder / 'challenge1b_input.json'
        out = folder / 'challenge1b_output.json'
        if not cfg.exists(): return
        config = json.loads(cfg.read_text())
        persona = config['persona']['role']
        job = config['job_to_be_done']['task']
        context = f"As a {persona}, {job}."

        # extract sections
        sections, embs = [], []
        for doc in config.get('documents', []):
            pdfp = folder / 'PDFs' / doc['filename']
            if not pdfp.exists(): continue
            text = ''.join([p.extract_text() or '' for p in pdfplumber.open(pdfp).pages])
            paras = [p.strip() for p in text.split('\n\n') if len(p) > 50]
            for para in paras:
                sections.append({'document': doc['filename'], 'text': para})
                embs.append(sbert.encode(para).tolist())
        if not sections: return

        # build graph
        x = torch.tensor(embs, dtype=torch.float)
        N = len(sections)
        idx = torch.arange(N)
        src = idx.repeat_interleave(N-1); dst = torch.cat([torch.cat([idx[:i], idx[i+1:]]) for i in range(N)])
        data = Data(x=x, edge_index=torch.stack([src,dst]), batch=torch.zeros(N,dtype=torch.long))

        # score
        gnn = GraphSAGE(in_channels=x.size(1), hidden_channels=GNN_HIDDEN_DIM)
        gnn.eval()
        with torch.no_grad(): scores = gnn(data.x, data.edge_index, data.batch).cpu().numpy()

        # top-k
        top = np.argsort(scores)[::-1][:TOP_K]

        # build output
        metadata = {
            'collection': folder.name,
            'persona': persona,
            'job_to_be_done': job,
            'processed_at': datetime.datetime.utcnow().isoformat(),
            'model': 'SentenceTransformer'
        }
        extracted, analysis = [], []
        for rank, idx in enumerate(top,1):
            sec = sections[idx]
            title = sec['text'].split('\n')[0][:60]
            extracted.append({
                'document': sec['document'],
                'section_title': title,
                'page_number': None,  # page tracking omitted
                'importance_rank': rank
            })
            summary = refine_text(sec['text'])
            analysis.append({
                'document': sec['document'],
                'page_number': None,
                'refined_text': summary
            })

        out.write_text(json.dumps({
            'metadata': metadata,
            'extracted_sections': extracted,
            'subsection_analysis': analysis
        }, indent=2))

    # --- MAIN ------------------------------------------------------
    def main():
        for d in Path(INPUT_DIR).iterdir():
            if d.is_dir(): process_collection(d)

    if __name__=='__main__': main()
