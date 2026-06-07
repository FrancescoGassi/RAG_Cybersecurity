import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Optional
import logging
from tqdm import tqdm

from src.config import LLM_MODEL_NAME, MAX_TOKENS, QWEN_USE_4BIT, MODEL_TYPE

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

class LLMPredictor:
    def __init__(self, max_tokens: int = MAX_TOKENS, model_type: str = None, debug: bool = False):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_tokens = max_tokens
        self.model_type = model_type if model_type else MODEL_TYPE
        self.qwen_use_4bit = QWEN_USE_4BIT
        self.debug = debug

    def load(self, model_name: Optional[str] = None, device: Optional[str] = None, qwen_use_4bit: Optional[bool] = None):
        if model_name is None:
            model_name = LLM_MODEL_NAME
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if qwen_use_4bit is not None:
            self.qwen_use_4bit = qwen_use_4bit

        print(f"   Caricamento {model_name} su {self.device} (tipo: {self.model_type})...")
        
        quant_config = None
        if self.model_type == "chat" and self.qwen_use_4bit and self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("   → Utilizzo caricamento in 4-bit")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.truncation_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            quantization_config=quant_config,
            trust_remote_code=True
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()
        print("   Modello pronto.")

    def truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def buildPrompt(self, retrieved_texts: List[str], retrieved_labels: List[int], query_text: str):
        """
        Costruisce il prompt
        """
        system_prompt = (
            "You are a cybersecurity expert. Classify Windows PE applications based on their features.\n"
            "The features are extracted from PE files using the EMBER library and LIEF tool. "
            "They represent statistical information derived from the binary.\n\n"
            "Below are similar examples that can be labeled as malware or goodware:\n"
        )
        
        # Usa al massimo 2 esempi (per limitare la lunghezza)
        max_examples = min(2, len(retrieved_texts))
        examples_str = ""
        for i in range(max_examples):
            text = self.truncate_text(retrieved_texts[i], 200)   # Ogni esempio max 200 token
            label = "malware" if retrieved_labels[i] == 1 else "goodware"
            examples_str += f"{i+1}. {text}\n   Class: {label}\n\n"
        
        query_trunc = self.truncate_text(query_text, 300)        # Query max 300 token
        
        user_content = (
            f"{examples_str}"
            "Based on these examples, classify the following file as either malware or goodware.\n"
            "Answer with a single word: malware or goodware.\n\n"
            f"File features:\n{query_trunc}\n\nAnswer:"
        )
        
        if self.model_type == "chat":
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_content}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = system_prompt + user_content
        
        total_tokens = len(self.tokenizer.encode(prompt))
        return prompt, total_tokens

    def predict(self, test_texts: List[str], true_labels_num: List[int],
                vector_index, embedding_model, k: int, output_csv: str):
        if self.model is None:
            raise ValueError("Chiamare load() prima di predict().")
        
        results = []
        token_counts = []
        total = len(test_texts)
        print(f"   Predizioni su {total} campioni (k={k})")
        
        for idx, (query, true_label_num) in enumerate(tqdm(zip(test_texts, true_labels_num), total=total, desc="   Progresso", unit="campione", ncols=80)):
            q_emb = embedding_model.encode([query])[0]
            _, indices = vector_index.search(q_emb, k=k)
            ret_texts, ret_targets = vector_index.get_metadata_by_indices(indices)
            
            prompt, total_tokens = self.buildPrompt(ret_texts, ret_targets, query)
            token_counts.append(total_tokens)
            
            if total_tokens > self.max_tokens:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_tokens).to(self.device)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            generated = self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            if not generated:
                generated = ""
            
            if self.debug:
                print(f"\n[DEBUG] Sample {idx}: generated = '{generated}'")
            
            first_word = generated.split()[0].lower().strip('.,!?') if generated else ""
            if first_word == "malware":
                pred_str = "malware"
                pred_type = "llm"
            elif first_word == "goodware":
                pred_str = "goodware"
                pred_type = "llm"
            else:
                # Fallback al majority voting
                ret_labels_arr = np.array(ret_targets)
                counts = np.bincount(ret_labels_arr)
                pred_num = int(np.argmax(counts))
                pred_str = "malware" if pred_num == 1 else "goodware"
                pred_type = "mv"
                if self.debug:
                    print(f"[DEBUG] Fallback to MV, generated='{generated}', first_word='{first_word}'")
            
            true_label_str = "malware" if true_label_num == 1 else "goodware"
            
            results.append({
                'prediction': pred_str,
                'true_label': true_label_str,
                'pred_type': pred_type,
            })
        
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
        acc = (df['prediction'] == df['true_label']).mean()
        llm_percent = (df['pred_type'] == 'llm').mean() * 100
        print(f"   Accuratezza: {acc*100:.2f}% - CSV salvato in {output_csv}")
        print(f"   Predizioni LLM: {llm_percent:.1f}% dei casi")
        if token_counts:
            print(f"   Token medi nel prompt: {np.mean(token_counts):.1f} (max {np.max(token_counts)})")
        
        return df