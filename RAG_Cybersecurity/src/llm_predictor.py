import torch
import pandas as pd
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Optional
import logging
from tqdm import tqdm

from src.config import LLM_MODEL_NAME, MAX_TOKENS, USE_4BIT, MODEL_TYPE

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

class LLMPredictor:
    def __init__(self, max_tokens: int = MAX_TOKENS, model_type: str = None, debug: bool = False,
                 use_logits: bool = True):
        """
            use_logits: True = confronto logit (stabile, veloce, adatto a modelli piccoli),
            False = generazione di testo libero (meglio per LLM grandi).
        """
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_tokens = max_tokens
        self.model_type = model_type if model_type else MODEL_TYPE
        self.use_4bit = USE_4BIT
        self.debug = debug
        self.model_name = LLM_MODEL_NAME
        self.use_logits = use_logits

    def load(self, model_name: Optional[str] = None, device: Optional[str] = None, use_4bit: Optional[bool] = None):
        if model_name is None:
            model_name = LLM_MODEL_NAME
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if use_4bit is not None:
            self.use_4bit = use_4bit

        print(f"   Loading {model_name} on {self.device} (type: {self.model_type})...")
        
        quant_config = None
        if self.use_4bit and self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("   → Using 4-bit loading on GPU")
        else:
            print("   → Loading in FP32 on CPU (4-bit not supported)")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.truncation_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quant_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        print("   Modello pronto.")

    def truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def build_prompt(self, retrieved_texts: List[str], retrieved_labels: List[int], query_text: str):
        system_prompt = (
            "You are a cybersecurity expert. Classify Windows PE applications based on their features.\n"
            "The features are extracted from PE files using the EMBER library and LIEF tool. "
            "They represent statistical information derived from the binary.\n\n"
            "Below are similar examples that can be labeled as malware or goodware:\n"
        )
        
        good_examples = [(t, l) for t, l in zip(retrieved_texts, retrieved_labels) if l == 0][:3]
        mal_examples  = [(t, l) for t, l in zip(retrieved_texts, retrieved_labels) if l == 1][:3]
        
        examples_str = "Examples:\n"
        for i, (text, label) in enumerate(good_examples + mal_examples, 1):
            label_str = "goodware" if label == 0 else "malware"
            short_text = self.truncate_text(text, 300)  # truncate each example to 300 tokens
            examples_str += f"{i}. {short_text}\n   Class: {label_str}\n\n"
        
        query_trunc = self.truncate_text(query_text, 400)
        
        user_content = (
            f"{examples_str}"
            "Now classify the following file:\n"
            f"{query_trunc}\n\n"
            "Answer (single word, malware or goodware):"
        )
        
        if self.model_type == "chat":
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = system_prompt + user_content
        
        return prompt

    def _get_token_ids_for_words(self, words: List[str]) -> List[int]:
        """Returns token IDs for given words (tries with leading space for BPE)."""
        ids = []
        for w in words:
            tokenized = self.tokenizer.encode(f" {w}", add_special_tokens=False)
            if not tokenized:
                tokenized = self.tokenizer.encode(w, add_special_tokens=False)
            if tokenized:
                ids.append(tokenized[0])
        return ids

    def predict(self, test_texts: List[str], true_labels_num: List[int],
                vector_index, embedding_model, k: int, output_csv: str) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Call load() before predict().")
        
        results = []
        token_counts = []
        total = len(test_texts)
        print(f"   Predicting on {total} samples (k={k})")
        
        # Precompute token IDs for 'malware' and 'goodware'
        malware_ids = self._get_token_ids_for_words(["malware", "malware"])
        goodware_ids = self._get_token_ids_for_words(["goodware", "goodware"])
        if not malware_ids or not goodware_ids:
            print("   Warning: 'malware'/'goodware' tokens not found. Falling back to generation mode.")
            self.use_logits = False
        
        for idx, (query, true_label_num) in enumerate(tqdm(zip(test_texts, true_labels_num), total=total, desc="   Progress", unit="sample", ncols=80)):
            # Retrieve balanced neighbors (max 3 per class)
            q_emb = embedding_model.encode([query])[0]
            distances, indices = vector_index.search(q_emb, k=k*2)
            ret_texts, ret_targets = vector_index.get_metadata_by_indices(indices)
            
            good_indices = [i for i, lbl in enumerate(ret_targets) if lbl == 0]
            mal_indices   = [i for i, lbl in enumerate(ret_targets) if lbl == 1]
            selected_idx = []
            selected_idx.extend(good_indices[:3])
            selected_idx.extend(mal_indices[:3])
            
            if len(selected_idx) < k:
                remaining = [i for i in range(len(ret_targets)) if i not in selected_idx]
                needed = k - len(selected_idx)
                selected_idx.extend(remaining[:needed])
            selected_idx = selected_idx[:k]
            
            ret_texts = [ret_texts[i] for i in selected_idx]
            ret_targets = [ret_targets[i] for i in selected_idx]
            
            # Build prompt
            prompt = self.build_prompt(ret_texts, ret_targets, query)
            total_tokens_prompt = len(self.tokenizer.encode(prompt))
            token_counts.append(total_tokens_prompt)
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_tokens).to(self.device)
            
            if self.use_logits and malware_ids and goodware_ids:
                # Logits mode: compare the logits of the first generated token
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0, -1, :]
                
                malware_logit = logits[malware_ids[0]].item()
                goodware_logit = logits[goodware_ids[0]].item()
                
                if malware_logit > goodware_logit:
                    pred_str = "malware"
                else:
                    pred_str = "goodware"
                pred_type = "llm"
                
                if self.debug:
                    print(f"\n[DEBUG] Sample {idx}: malware_logit={malware_logit:.4f}, goodware_logit={goodware_logit:.4f} -> {pred_str}")
            else:
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=20,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=False,   # deterministic
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                generated = self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                if self.debug:
                    print(f"\n[DEBUG] Sample {idx}: generated = '{generated}'")
                
                match = re.search(r'\b(malware|goodware)\b', generated.lower())
                if match:
                    pred_str = match.group(1)
                    pred_type = "llm"
                else:
                    ret_labels_arr = np.array(ret_targets)
                    if len(ret_labels_arr) == 0:
                        pred_num = 1
                    else:
                        counts = np.bincount(ret_labels_arr.astype(int))
                        pred_num = int(np.argmax(counts))
                    pred_str = "malware" if pred_num == 1 else "goodware"
                    pred_type = "mv"
                    if self.debug:
                        print(f"[DEBUG] Fallback to MV, generated='{generated}'")
            
            true_label_str = "malware" if true_label_num == 1 else "goodware"
            results.append({
                'prediction': pred_str,
                'true_label': true_label_str,
                'pred_type': pred_type,
            })
        
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
        acc = (df['prediction'] == df['true_label']).mean()
        llm_percent = (df['pred_type'].str.contains('llm')).mean() * 100
        print(f"   Accuracy: {acc*100:.2f}% - CSV saved to {output_csv}")
        print(f"   LLM predictions: {llm_percent:.1f}% of cases")
        if token_counts:
            print(f"   Average prompt tokens: {np.mean(token_counts):.1f} (max {np.max(token_counts)})")
        
        return df