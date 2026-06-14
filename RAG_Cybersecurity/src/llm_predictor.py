from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    DEBUG_LLM,
    K_NEIGHBORS,
    LLM_MODEL_NAME,
    MAX_EXAMPLES_IN_PROMPT,
    MAX_FEATURES_IN_PROMPT,
    MAX_PROMPT_TOKENS,
    MIN_CONSISTENT_LLM_VOTES,
    MIN_FEATURES_IN_PROMPT,
    PROMPT_VARIANTS,
    STRONG_MV_MIN_MARGIN,
    USE_4BIT,
    USE_STRONG_MV_GUARD,
)
from src.feature_names import BODMASFeatureNames

if TYPE_CHECKING:
    from src.faiss_rag_index import FaissRAGIndex, Neighbor

CLASS_NAME = {0: "goodware", 1: "malware"}

@dataclass(frozen=True)
class PromptVariant:
    name: str
    reverse_examples: bool = False

_VARIANTS = [
    PromptVariant("normal", reverse_examples=False),
    PromptVariant("reversed", reverse_examples=True),
]

class LLMPredictor:
    def __init__(self, model_name: str = LLM_MODEL_NAME, debug: bool = DEBUG_LLM, use_4bit: bool = USE_4BIT) -> None:
        self.model_name = model_name
        self.debug = bool(debug)
        self.use_4bit = bool(use_4bit)
        self.model = None
        self.tokenizer = None
        self.input_device = None
        self._torch = None
        self.feature_descriptions = BODMASFeatureNames.get_feature_descriptions()

    def load(self, device: Optional[str] = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        quantized = self.use_4bit and runtime_device == "cuda"
        print(f"   Caricamento LLM {self.model_name} su {runtime_device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, object] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if quantized:
            model_kwargs.update(
                device_map="auto",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                ),
                dtype="auto",
            )
        elif runtime_device == "cuda":
            model_kwargs.update(device_map="auto", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        else:
            model_kwargs["dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if runtime_device == "cpu":
            self.model.to("cpu")
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        self._torch = torch
        print("   LLM pronto: sampling con temperatura 0.3, output numerico 0/1, descrizioni feature.")

    def _format_vector(self, vector: np.ndarray, feature_names: List[str], max_features: int) -> str:
        limit = min(len(vector), len(feature_names), max_features)
        parts = []
        for i in range(limit):
            name = feature_names[i]
            desc = self.feature_descriptions.get(name, name)
            parts.append(f"{desc[:40]}={float(vector[i]):+.2f}")
        return ", ".join(parts)

    @staticmethod
    def _parse_output(raw_output: str) -> Optional[int]:
        text = raw_output.strip().upper()
        if text == "0":
            return 0
        if text == "1":
            return 1
        if text == "GOODWARE":
            return 0
        if text == "MALWARE":
            return 1
        match = re.search(r"\b([01])\b", text)
        if match:
            return int(match.group(1))
        return None

    def _render(self, messages: list[dict[str, str]]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer non caricato.")
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"

    def _balance_examples(self, examples: list["Neighbor"], max_examples: int) -> list["Neighbor"]:
        goodware = [e for e in examples if e.label == 0]
        malware = [e for e in examples if e.label == 1]
        half = max_examples // 2
        selected = []
        selected.extend(goodware[:half])
        selected.extend(malware[:half])
        if len(selected) < max_examples:
            remaining = max_examples - len(selected)
            if len(goodware) > half:
                selected.extend(goodware[half:half+remaining])
            else:
                selected.extend(malware[half:half+remaining])
        return selected[:max_examples]

    def _build_messages(
        self,
        query_z: np.ndarray,
        examples: list["Neighbor"],
        feature_names: list[str],
        max_features: int,
        variant: PromptVariant,
    ) -> list[dict[str, str]]:
        balanced_examples = self._balance_examples(examples, MAX_EXAMPLES_IN_PROMPT)
        if variant.reverse_examples:
            balanced_examples = list(reversed(balanced_examples))

        goodware_count = sum(1 for e in balanced_examples if e.label == 0)
        malware_count = sum(1 for e in balanced_examples if e.label == 1)

        user_content_lines = [
            "Classify the QUERY Windows PE file into exactly one of two classes.",
            "",
            "The two possible classes are:",
            "- 0 = GOODWARE (benign)",
            "- 1 = MALWARE (malicious)",
            "",
            "Do not prefer either class. Base your decision only on the supplied FAISS neighbors.",
            "",
            f"Retrieved neighbors used in the prompt: {len(balanced_examples)}.",
            f"GOODWARE neighbors (class 0): {goodware_count}.",
            f"MALWARE neighbors (class 1): {malware_count}.",
            "",
            "FAISS NEAREST EXAMPLES (with feature descriptions):",
        ]

        for idx, ex in enumerate(balanced_examples, start=1):
            values = self._format_vector(ex.z_vector, feature_names, max_features)
            user_content_lines.append(
                f"Example {idx}: label={ex.label}; similarity={ex.similarity:+.4f}; features=[{values}]"
            )

        query_values = self._format_vector(query_z, feature_names, max_features)
        user_content_lines.extend(
            [
                "",
                f"QUERY features=[{query_values}]",
                "",
                "Return exactly one digit: 0 for GOODWARE or 1 for MALWARE.",
                "Do not provide explanations, punctuation, or additional text.",
            ]
        )

        system_content = (
            "You are a neutral binary classifier for Windows PE files. "
            "GOODWARE (0) and MALWARE (1) are equally possible. "
            "Here are two concrete examples:\n"
            "Example A: features=[...] -> 0\n"
            "Example B: features=[...] -> 1\n"
            "Now classify the following query based on its FAISS neighbors."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "\n".join(user_content_lines)},
        ]

    def _prepare_prompt(
        self,
        query_z: np.ndarray,
        examples: list["Neighbor"],
        feature_names: list[str],
        variant: PromptVariant,
    ) -> tuple[str, int, int]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer non caricato.")
        feature_options = list(range(MAX_FEATURES_IN_PROMPT, MIN_FEATURES_IN_PROMPT - 1, -1))
        for max_features in feature_options:
            messages = self._build_messages(query_z, examples, feature_names, max_features, variant)
            prompt = self._render(messages)
            token_count = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if token_count <= MAX_PROMPT_TOKENS:
                return prompt, token_count, max_features
        messages = self._build_messages(query_z, examples, feature_names, MIN_FEATURES_IN_PROMPT, variant)
        prompt = self._render(messages)
        token_count = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        return prompt, token_count, MIN_FEATURES_IN_PROMPT

    def _generate_one(
        self,
        query_z: np.ndarray,
        examples: list["Neighbor"],
        feature_names: list[str],
        variant: PromptVariant,
    ) -> tuple[Optional[int], str, int]:
        if self.model is None or self.tokenizer is None or self._torch is None:
            raise RuntimeError("Chiamare load() prima di predict().")

        prompt, token_count, feature_count = self._prepare_prompt(query_z, examples, feature_names, variant)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS)
        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}
        input_length = int(inputs["input_ids"].shape[1])

        with self._torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[0, input_length:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        prediction = self._parse_output(raw_output)

        if self.debug:
            print(f"      {variant.name}: raw={raw_output!r}, parsed={prediction}, tokens={token_count}, features={feature_count}")
        return prediction, raw_output, token_count

    def predict(self, test_features: pd.DataFrame, true_labels: list[int], index: "FaissRAGIndex", output_csv: str) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Chiamare load() prima di predict().")
        if len(test_features) != len(true_labels):
            raise ValueError("Feature e label di test hanno lunghezze diverse.")

        variants = _VARIANTS[: max(1, min(PROMPT_VARIANTS, len(_VARIANTS)))]
        full_results: list[dict[str, object]] = []
        csv_rows: list[dict[str, str]] = []

        iterator = zip(test_features.itertuples(index=False, name=None), true_labels)
        for sample_idx, (row_values, true_label) in enumerate(tqdm(iterator, total=len(test_features), desc="RAG prediction", unit="sample")):
            row = np.asarray(row_values, dtype=np.float64)
            mv_prediction, mv_counts, natural_neighbors = index.majority_vote(row, k=K_NEIGHBORS)
            context = natural_neighbors
            query_z, _ = index.transform_query(row)

            if self.debug:
                print(
                    f"\n[Sample {sample_idx}] true={true_label}, MV={mv_prediction}, "
                    f"counts={mv_counts}, neighbors={[n.label for n in natural_neighbors]}"
                )

            valid_votes: list[int] = []
            raw_outputs: dict[str, str] = {}
            for variant in variants:
                vote, raw_output, _ = self._generate_one(query_z[0], context, index.feature_names, variant)
                raw_outputs[variant.name] = raw_output
                if vote is not None:
                    valid_votes.append(vote)

            vote_counts = Counter(valid_votes)
            llm_candidate: Optional[int] = None
            if vote_counts:
                label, votes = vote_counts.most_common(1)[0]
                if votes >= MIN_CONSISTENT_LLM_VOTES:
                    llm_candidate = int(label)

            mv_margin = abs(mv_counts[0] - mv_counts[1])
            if llm_candidate is None:
                final_prediction = mv_prediction
                pred_type = "MV"
                decision_source = "fallback_majority_voting_invalid_llm"
            elif USE_STRONG_MV_GUARD and llm_candidate != mv_prediction and mv_margin >= STRONG_MV_MIN_MARGIN:
                final_prediction = mv_prediction
                pred_type = "MV"
                decision_source = "fallback_majority_voting_strong_disagreement"
            else:
                final_prediction = llm_candidate
                pred_type = "LLM"
                decision_source = "llm_rag_consensus"

            csv_rows.append(
                {
                    "prediction": CLASS_NAME[int(final_prediction)],
                    "true_label": CLASS_NAME[int(true_label)],
                    "pred_type": pred_type,
                }
            )
            full_results.append(
                {
                    "prediction": CLASS_NAME[int(final_prediction)],
                    "true_label": CLASS_NAME[int(true_label)],
                    "pred_type": pred_type,
                    "prediction_num": int(final_prediction),
                    "true_label_num": int(true_label),
                    "decision_source": decision_source,
                    "llm_prediction": CLASS_NAME[llm_candidate] if llm_candidate is not None else "invalid",
                    "mv_prediction": CLASS_NAME[int(mv_prediction)],
                    "mv_goodware_votes": int(mv_counts[0]),
                    "mv_malware_votes": int(mv_counts[1]),
                    "neighbor_labels": [int(n.label) for n in natural_neighbors],
                    "llm_raw_outputs": raw_outputs,
                }
            )

        pd.DataFrame(csv_rows, columns=["prediction", "true_label", "pred_type"]).to_csv(output_csv, index=False)
        return pd.DataFrame(full_results)