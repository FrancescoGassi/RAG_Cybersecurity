import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import logging
from tqdm import tqdm

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

class LLMPredictor:
    def __init__(self, max_tokens: int = 1024):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_tokens = max_tokens

    def load(self, model_name: str = "distilgpt2", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"   Caricamento {model_name} su {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("   Modello pronto.")

    def truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def buildPrompt(self, retrieved_texts: List[str], retrieved_labels: List[int], query_text: str) -> str:
        n_examples = len(retrieved_texts) + 1
        tokens_per_example = max(50, self.max_tokens // (n_examples + 1))

        prompt = (
            "You are a cybersecurity expert. Classify Windows PE applications based on their features.\n"
            "The features are extracted from PE files using the EMBER library and LIEF tool. "
            "They represent statistical information derived from the binary.\n\n"
            "Below are similar examples that can be labeled as malware or goodware:\n"
        )
        for i, (text, label) in enumerate(zip(retrieved_texts, retrieved_labels)):
            truncated_text = self.truncate_text(text, tokens_per_example)
            label_str = "malware" if label == 1 else "goodware"
            prompt += f"{i+1}. {truncated_text}\n   Class: {label_str}\n\n"

        truncated_query = self.truncate_text(query_text, tokens_per_example)
        prompt += (
            "Based on these examples, classify the following file as either malware or goodware.\n"
            "Answer with a single word: malware or goodware.\n\n"
            f"File features:\n{truncated_query}\n\nAnswer:"
        )
        return prompt

    def predict(self, test_texts: List[str], true_labels: List[int],
                vector_index, embedding_model, k: int, output_csv: str):
        if self.model is None:
            raise ValueError("Chiamare load() prima di predict().")
        results = []
        total = len(test_texts)
        print(f"   Predizioni su {total} campioni (k={k})")
        for query, true_label in tqdm(zip(test_texts, true_labels), total=total, desc="   Progresso", unit="campione", ncols=80):
            q_emb = embedding_model.encode([query])[0]
            _, indices = vector_index.search(q_emb, k=k)
            ret_texts, ret_targets = vector_index.get_metadata_by_indices(indices)

            prompt = self.buildPrompt(ret_texts, ret_targets, query)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=self.max_tokens).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=5,
                                          pad_token_id=self.tokenizer.pad_token_id)
            gen = self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                                        skip_special_tokens=True).strip().lower()

            if "malware" in gen:
                pred = 1
                pred_type = "llm"
            elif "goodware" in gen:
                pred = 0
                pred_type = "llm"
            else:
                retrieved_labels_array = np.array(ret_targets)
                counts = np.bincount(retrieved_labels_array)
                pred = int(np.argmax(counts))
                pred_type = "mv"

            results.append({
                'prediction': pred,
                'true_label': true_label,
                'retrieved_labels': str(ret_targets),
                'pred_type': pred_type
            })

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        acc = (df['prediction'] == df['true_label']).mean()
        print(f"   Accuratezza: {acc*100:.2f}% - CSV salvato in {output_csv}")
        return df