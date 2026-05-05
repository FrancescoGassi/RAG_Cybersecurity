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
    def __init__(self, max_tokens: int = MAX_TOKENS, model_type: str = None):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_tokens = max_tokens
        self.model_type = model_type if model_type else MODEL_TYPE   # "causal" o "chat"
        self.qwen_use_4bit = QWEN_USE_4BIT   # valore predefinito da config

    def load(self, model_name: Optional[str] = None, device: Optional[str] = None, qwen_use_4bit: Optional[bool] = None):
        if model_name is None:
            model_name = LLM_MODEL_NAME
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if qwen_use_4bit is not None:
            self.qwen_use_4bit = qwen_use_4bit

        print(f"   Caricamento {model_name} su {self.device} (tipo forzato: {self.model_type})...")
        
        # Configurazione quantizzazione solo per modelli chat (es. Qwen) se richiesta
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
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("   Modello pronto.")

    def truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def buildPrompt(self, retrieved_texts: List[str], retrieved_labels: List[int], query_text: str):
        """
        Costruisce il prompt secondo il tipo di modello.
        Restituisce (prompt_string, lista_tokens_per_vicino, total_tokens_prompt)
        """
        n_examples = len(retrieved_texts)
        tokens_per_example = max(30, (self.max_tokens - 200) // max(1, n_examples))
        
        neighbor_tokens_list = []
        
        system_prompt = (
            "You are a cybersecurity expert. Classify Windows PE applications based on their features.\n"
            "The features are extracted from PE files using the EMBER library and LIEF tool. "
            "They represent statistical information derived from the binary.\n\n"
            "Below are similar examples that can be labeled as malware or goodware:\n"
        )
        
        if self.model_type == "chat":
            # Per modelli chat (Qwen, Llama, Mistral, ecc.)
            messages = [{"role": "system", "content": system_prompt.strip()}]
            fewshot_text = ""
            for i, (text, label) in enumerate(zip(retrieved_texts, retrieved_labels)):
                truncated_text = self.truncate_text(text, tokens_per_example)
                token_count = len(self.tokenizer.encode(truncated_text))
                neighbor_tokens_list.append(token_count)
                label_str = "malware" if label == 1 else "goodware"
                fewshot_text += f"{i+1}. {truncated_text}\n   Class: {label_str}\n\n"
            truncated_query = self.truncate_text(query_text, tokens_per_example)
            user_content = (
                f"{fewshot_text}"
                "Based on these examples, classify the following file as either malware or goodware.\n"
                "Answer with a single word: malware or goodware.\n\n"
                f"File features:\n{truncated_query}\n\nAnswer:"
            )
            messages.append({"role": "user", "content": user_content})
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = system_prompt
            for i, (text, label) in enumerate(zip(retrieved_texts, retrieved_labels)):
                truncated_text = self.truncate_text(text, tokens_per_example)
                token_count = len(self.tokenizer.encode(truncated_text))
                neighbor_tokens_list.append(token_count)
                label_str = "malware" if label == 1 else "goodware"
                prompt += f"{i+1}. {truncated_text}\n   Class: {label_str}\n\n"
            truncated_query = self.truncate_text(query_text, tokens_per_example)
            prompt += (
                "Based on these examples, classify the following file as either malware or goodware.\n"
                "Answer with a single word: malware or goodware.\n\n"
                f"File features:\n{truncated_query}\n\nAnswer:"
            )
        
        total_tokens = len(self.tokenizer.encode(prompt))
        return prompt, neighbor_tokens_list, total_tokens

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

            ret_acc = sum(1 for t in ret_targets if t == true_label) / len(ret_targets)

            prompt, neighbor_tokens, total_tokens = self.buildPrompt(ret_texts, ret_targets, query)
            
            truncated_flag = False
            if total_tokens > self.max_tokens:
                prompt_tokens = self.tokenizer.encode(prompt, truncation=True, max_length=self.max_tokens)
                prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                total_tokens = len(prompt_tokens)
                truncated_flag = True
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                    max_length=self.max_tokens).to(self.device)
            with torch.no_grad():
                if self.model_type == "chat":
                    out = self.model.generate(**inputs, max_new_tokens=10,
                                              pad_token_id=self.tokenizer.pad_token_id,
                                              do_sample=False)
                else:
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
                ret_labels_arr = np.array(ret_targets)
                counts = np.bincount(ret_labels_arr)
                pred = int(np.argmax(counts))
                pred_type = "mv"

            results.append({
                'prediction': pred,
                'true_label': true_label,
                'retrieved_labels': str(ret_targets),
                'retrieval_accuracy': ret_acc,
                'pred_type': pred_type,
                'total_prompt_tokens': total_tokens,
                'neighbor_tokens': str(neighbor_tokens),
                'truncated': truncated_flag
            })

        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        acc = (df['prediction'] == df['true_label']).mean()
        print(f"   Accuratezza: {acc*100:.2f}% - CSV salvato in {output_csv}")
        return df