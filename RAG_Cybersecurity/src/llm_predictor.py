from __future__ import annotations

import importlib.util
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    DEBUG_LLM,
    K_NEIGHBORS,
    LLM_MODEL_NAME,
    MAX_EXAMPLES_IN_PROMPT,
    MAX_FEATURES_IN_PROMPT,
    MAX_NEW_TOKENS,
    MAX_PROMPT_TOKENS,
    MIN_CONSISTENT_LLM_VOTES,
    MIN_FEATURES_IN_PROMPT,
    PROMPT_VARIANTS,
    USE_4BIT,
)
from src.feature_names import BODMASFeatureNames
from src.faiss_rag_index import FaissRAGIndex, Neighbor


CLASS_NAME = {
    0: "goodware",
    1: "malware",
}

# Se True, l'LLM prevale sempre (quando valido). Se False, usa la logica restrittiva.
LLM_ALWAYS_OVERRIDE = False

# Numero minimo di varianti LLM che devono concordare per poter sovrascrivere MV.
# Con LLM_ALWAYS_OVERRIDE=False, si usa ancora MIN_VARIANTS_FOR_LLM_OVERRIDE.
MIN_VARIANTS_FOR_LLM_OVERRIDE = 2 


@dataclass(frozen=True)
class PromptVariant:
    """Configurazione di una variante semantica del prompt."""

    name: str
    reverse_examples: bool
    compact: bool


_VARIANTS = [
    PromptVariant(
        name="normal",
        reverse_examples=False,
        compact=False,
    ),
    PromptVariant(
        name="reversed",
        reverse_examples=True,
        compact=False,
    ),
    PromptVariant(
        name="compact",
        reverse_examples=False,
        compact=True,
    ),
]


class LLMPredictor:
    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        debug: bool = DEBUG_LLM,
        use_4bit: bool = USE_4BIT,
    ) -> None:
        self.model_name = model_name
        self.debug = bool(debug)
        self.use_4bit = bool(use_4bit)
        self.model = None
        self.tokenizer = None
        self.input_device = None
        self._torch = None
        self.feature_descriptions = (
            BODMASFeatureNames.get_feature_descriptions()
        )

    def load(self, device: Optional[str] = None) -> None:
        """Carica tokenizer e modello sul dispositivo disponibile."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        runtime_device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if runtime_device not in {"cpu", "cuda"}:
            raise ValueError(
                "device deve essere 'cpu', 'cuda' oppure None."
            )

        if runtime_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "È stato richiesto CUDA, ma CUDA non è disponibile."
            )

        bitsandbytes_available = (
            importlib.util.find_spec("bitsandbytes") is not None
        )
        quantized = (
            self.use_4bit
            and runtime_device == "cuda"
            and bitsandbytes_available
        )

        if (
            self.use_4bit
            and runtime_device == "cuda"
            and not bitsandbytes_available
        ):
            print(
                "   AVVISO: bitsandbytes non disponibile; "
                "caricamento GPU senza quantizzazione 4-bit."
            )

        print(
            f"   Caricamento LLM {self.model_name} "
            f"su {runtime_device}..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.eos_token_id is None:
            raise RuntimeError(
                "Il tokenizer non definisce eos_token_id."
            )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if quantized:
            model_kwargs.update(
                device_map="auto",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=(
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float16
                    ),
                ),
                dtype="auto",
            )
        elif runtime_device == "cuda":
            model_kwargs.update(
                device_map="auto",
                dtype=(
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
            )
        else:
            model_kwargs["dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        if runtime_device == "cpu":
            self.model.to("cpu")

        self.model.eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        self.input_device = (
            self.model.get_input_embeddings().weight.device
        )
        self._torch = torch

        print(
            "   LLM pronto: modello Instruct, generazione deterministica, "
            "output GOODWARE/MALWARE."
        )

    def _format_vector(
        self,
        vector: np.ndarray,
        feature_names: list[str],
        max_features: int,
    ) -> str:
        """Formatta un sottoinsieme delle feature standardizzate."""
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        limit = min(
            len(values),
            len(feature_names),
            max(0, int(max_features)),
        )
        parts: list[str] = []

        for index in range(limit):
            name = feature_names[index]
            description = self.feature_descriptions.get(name, name)
            parts.append(
                f"{name} ({description})="
                f"{float(values[index]):+.3f}"
            )

        return ", ".join(parts)

    @staticmethod
    def _parse_output(raw_output: str) -> Optional[int]:
        """Converte una risposta semantica in 0=goodware o 1=malware.

        Sono accettate parole con maiuscole/minuscole arbitrarie e una breve
        frase contenente una sola classe. Se compaiono entrambe le classi o
        nessuna classe, la risposta viene considerata non valida.
        """
        if not isinstance(raw_output, str) or not raw_output.strip():
            return None

        matches = re.findall(
            r"\b(goodware|malware)\b",
            raw_output,
            flags=re.IGNORECASE,
        )

        if not matches:
            return None

        parsed_labels = {
            0 if match.lower() == "goodware" else 1
            for match in matches
        }

        if len(parsed_labels) != 1:
            return None

        return parsed_labels.pop()

    def _render(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Applica il chat template del tokenizer, se disponibile."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer non caricato.")

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return (
            "\n\n".join(
                f"{message['role'].upper()}: {message['content']}"
                for message in messages
            )
            + "\nASSISTANT:"
        )

    @staticmethod
    def _similarity_weight(similarity: float) -> float:
        """Converte la cosine similarity [-1, 1] in un peso [0, 1]."""
        return float(np.clip((float(similarity) + 1.0) / 2.0, 0.0, 1.0))

    def _build_messages(
        self,
        query_z: np.ndarray,
        examples: list[Neighbor],
        feature_names: list[str],
        max_features: int,
        max_examples: int,
        variant: PromptVariant,
    ) -> list[dict[str, str]]:
        """Costruisce un prompt breve e coerente con l'evidenza FAISS."""
        if not examples:
            raise ValueError(
                "Nessun esempio FAISS disponibile per il prompt."
            )

        query_vector = np.asarray(
            query_z,
            dtype=np.float32,
        ).reshape(-1)

        if len(query_vector) != len(feature_names):
            raise ValueError(
                "Il numero di feature della query non coincide con "
                "feature_names."
            )

        # Ordina per similarità decrescente
        ordered_examples = sorted(
            examples,
            key=lambda example: example.similarity,
            reverse=True,
        )

        # Verifica che i vettori degli esempi siano della stessa lunghezza
        for example in ordered_examples:
            if len(example.vector) != len(query_vector):
                raise ValueError(
                    "Query ed esempio FAISS hanno dimensioni differenti."
                )

        effective_max_examples = (
            min(int(max_examples), 3)
            if variant.compact
            else int(max_examples)
        )
        prompt_examples = ordered_examples[:effective_max_examples]

        if variant.reverse_examples:
            prompt_examples = list(reversed(prompt_examples))

        all_goodware = [
            example for example in ordered_examples if example.label == 0
        ]
        all_malware = [
            example for example in ordered_examples if example.label == 1
        ]

        goodware_count = len(all_goodware)
        malware_count = len(all_malware)

        mean_goodware_similarity = (
            float(np.mean([ex.similarity for ex in all_goodware]))
            if all_goodware
            else 0.0
        )
        mean_malware_similarity = (
            float(np.mean([ex.similarity for ex in all_malware]))
            if all_malware
            else 0.0
        )

        weighted_goodware_score = float(
            sum(
                self._similarity_weight(example.similarity)
                for example in all_goodware
            )
        )
        weighted_malware_score = float(
            sum(
                self._similarity_weight(example.similarity)
                for example in all_malware
            )
        )

        nearest_neighbor = ordered_examples[0]
        nearest_neighbor_class = CLASS_NAME[
            int(nearest_neighbor.label)
        ].upper()

        if goodware_count == malware_count:
            retrieval_majority = nearest_neighbor_class
        elif goodware_count > malware_count:
            retrieval_majority = "GOODWARE"
        else:
            retrieval_majority = "MALWARE"

        system_content = (
            "You are a binary classifier for Windows PE files.\n"
            "The class meanings are fixed and must never be inverted:\n"
            "- GOODWARE means benign, clean, or safe software.\n"
            "- MALWARE means malicious or harmful software.\n"
            "Use only the retrieved numerical evidence supplied by the user.\n"
            "Cosine similarity closer to 1 indicates a more similar example.\n"
            "A smaller mean absolute standardized-feature distance indicates "
            "a closer example.\n"
            "Return exactly one word: GOODWARE or MALWARE."
        )

        summary_lines = [
            "RETRIEVAL SUMMARY OVER ALL NEIGHBORS:",
            f"- GOODWARE count: {goodware_count}",
            f"- MALWARE count: {malware_count}",
            f"- Label-count majority: {retrieval_majority}",
            (
                "- Mean GOODWARE cosine similarity: "
                f"{mean_goodware_similarity:+.4f}"
            ),
            (
                "- Mean MALWARE cosine similarity: "
                f"{mean_malware_similarity:+.4f}"
            ),
            (
                "- Weighted GOODWARE evidence: "
                f"{weighted_goodware_score:.4f}"
            ),
            (
                "- Weighted MALWARE evidence: "
                f"{weighted_malware_score:.4f}"
            ),
            f"- Closest-neighbor class: {nearest_neighbor_class}",
        ]

        if variant.compact:
            user_lines = [
                *summary_lines,
                "",
                "CLOSEST LABELED EXAMPLES:",
            ]
        else:
            user_lines = [
                "DECISION RULES:",
                "- Preserve the fixed GOODWARE/MALWARE meanings.",
                "- Consider all-neighbor label agreement.",
                "- Give more importance to higher cosine similarity.",
                "- Use full-vector distance as supporting evidence.",
                "- Do not infer a class from the cybersecurity topic alone.",
                "",
                *summary_lines,
                "",
                "LABELED RETRIEVED EXAMPLES:",
            ]

        feature_limit = (
            min(int(max_features), 3)
            if variant.compact
            else int(max_features)
        )

        for number, example in enumerate(prompt_examples, start=1):
            example_vector = np.asarray(
                example.vector,
                dtype=np.float32,
            ).reshape(-1)

            # La distanza media assoluta su tutte le feature
            mean_feature_distance = float(
                np.mean(np.abs(query_vector - example_vector))
            )

            displayed_values = self._format_vector(
                example_vector,
                feature_names,
                feature_limit,
            )

            user_lines.append(
                f"Example {number}: "
                f"label={CLASS_NAME[int(example.label)].upper()}; "
                f"cosine_similarity={float(example.similarity):+.4f}; "
                f"full_vector_mean_abs_distance="
                f"{mean_feature_distance:.4f}; "
                f"displayed_features=[{displayed_values}]"
            )

        query_values = self._format_vector(
            query_vector,
            feature_names,
            feature_limit,
        )

        user_lines.extend(
            [
                "",
                "UNLABELED QUERY:",
                f"displayed_features=[{query_values}]",
                "",
                "Classify the query from the retrieved evidence.",
                "Answer with exactly GOODWARE or MALWARE.",
            ]
        )

        return [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": "\n".join(user_lines),
            },
        ]

    def _prepare_prompt(
        self,
        query_z: np.ndarray,
        examples: list[Neighbor],
        feature_names: list[str],
        variant: PromptVariant,
    ) -> tuple[str, int, int, int]:
        """Riduce esempi/feature finché il prompt rispetta il limite token."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer non caricato.")

        maximum_examples = min(
            MAX_EXAMPLES_IN_PROMPT,
            len(examples),
        )

        if maximum_examples <= 0:
            raise ValueError("Nessun esempio disponibile per il prompt.")

        if MIN_FEATURES_IN_PROMPT <= 0:
            raise ValueError(
                "MIN_FEATURES_IN_PROMPT deve essere maggiore di zero."
            )

        if MAX_FEATURES_IN_PROMPT < MIN_FEATURES_IN_PROMPT:
            raise ValueError(
                "MAX_FEATURES_IN_PROMPT deve essere maggiore o uguale "
                "a MIN_FEATURES_IN_PROMPT."
            )

        for max_examples in range(maximum_examples, 0, -1):
            for max_features in range(
                MAX_FEATURES_IN_PROMPT,
                MIN_FEATURES_IN_PROMPT - 1,
                -1,
            ):
                messages = self._build_messages(
                    query_z=query_z,
                    examples=examples,
                    feature_names=feature_names,
                    max_features=max_features,
                    max_examples=max_examples,
                    variant=variant,
                )
                prompt = self._render(messages)
                token_count = len(
                    self.tokenizer(
                        prompt,
                        add_special_tokens=False,
                    )["input_ids"]
                )

                if token_count <= MAX_PROMPT_TOKENS:
                    effective_examples = (
                        min(max_examples, 3)
                        if variant.compact
                        else max_examples
                    )
                    effective_features = (
                        min(max_features, 3)
                        if variant.compact
                        else max_features
                    )
                    return (
                        prompt,
                        token_count,
                        effective_features,
                        effective_examples,
                    )

        raise ValueError(
            "Impossibile costruire un prompt entro "
            f"{MAX_PROMPT_TOKENS} token. Ridurre il testo del prompt "
            "oppure aumentare MAX_PROMPT_TOKENS."
        )

    def _generate_one(
        self,
        query_z: np.ndarray,
        examples: list[Neighbor],
        feature_names: list[str],
        variant: PromptVariant,
    ) -> tuple[Optional[int], str, int]:
        """Esegue una singola generazione deterministica."""
        if (
            self.model is None
            or self.tokenizer is None
            or self._torch is None
            or self.input_device is None
        ):
            raise RuntimeError(
                "Chiamare load() prima di predict()."
            )

        (
            prompt,
            token_count,
            feature_count,
            example_count,
        ) = self._prepare_prompt(
            query_z,
            examples,
            feature_names,
            variant,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {
            key: value.to(self.input_device)
            for key, value in inputs.items()
        }
        input_length = int(inputs["input_ids"].shape[1])

        with self._torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[0, input_length:]
        raw_output = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()
        prediction = self._parse_output(raw_output)

        if self.debug:
            print(
                f"      {variant.name}: "
                f"raw={raw_output!r}, "
                f"parsed={prediction}, "
                f"tokens={token_count}, "
                f"features={feature_count}, "
                f"examples={example_count}"
            )

        return prediction, raw_output, token_count

    @staticmethod
    def _select_llm_candidate(
        llm_votes: list[int],
        required_votes: int,
    ) -> tuple[Optional[int], int, Counter[int]]:
        """Seleziona una classe LLM solo in presenza di un vincitore unico."""
        vote_counts: Counter[int] = Counter(llm_votes)

        if not vote_counts:
            return None, 0, vote_counts

        highest_count = max(vote_counts.values())
        winners = [
            label
            for label, count in vote_counts.items()
            if count == highest_count
        ]

        if len(winners) != 1 or highest_count < required_votes:
            return None, int(highest_count), vote_counts

        return int(winners[0]), int(highest_count), vote_counts

    @staticmethod
    def _decide_final_prediction(
        mv_prediction: int,
        mv_counts: dict[int, int],
        llm_candidate: Optional[int],
        llm_votes: list[int],
        variant_count: int,
    ) -> tuple[int, str, str, bool, bool, int]:
        """Combina LLM e MV dando priorità all'LLM quando produce una risposta valida.

        Restituisce:
            final_prediction,
            pred_type,
            decision_source,
            llm_override_applied,
            llm_unanimous,
            mv_margin
        """
        mv_prediction = int(mv_prediction)

        if mv_prediction not in CLASS_NAME:
            raise ValueError("Predizione MV non binaria.")

        count_goodware = int(mv_counts.get(0, 0))
        count_malware = int(mv_counts.get(1, 0))
        mv_margin = abs(count_goodware - count_malware)

        # Determina se le varianti LLM sono unanimi (tutte valide e stessa classe)
        llm_unanimous = (
            len(llm_votes) == variant_count
            and len(set(llm_votes)) == 1
        )

        # Se non c'è alcun candidato LLM valido, fallback a MV
        if llm_candidate is None:
            return (
                mv_prediction,
                "MV",
                "mv_fallback_no_llm",
                False,
                llm_unanimous,
                mv_margin,
            )

        llm_candidate = int(llm_candidate)

        if LLM_ALWAYS_OVERRIDE:
            return (
                llm_candidate,
                "LLM",
                "llm_always_overrides",
                True,
                llm_unanimous,
                mv_margin,
            )

        if llm_candidate == mv_prediction:
            return (
                mv_prediction,
                "LLM",
                "llm_agrees_with_mv",
                False,
                llm_unanimous,
                mv_margin,
            )

        if mv_margin <= 1 and llm_unanimous:
            return (
                llm_candidate,
                "LLM",
                "unanimous_llm_overrides_uncertain_mv",
                True,
                True,
                mv_margin,
            )

        return (
            mv_prediction,
            "MV",
            "mv_retained",
            False,
            llm_unanimous,
            mv_margin,
        )

    def predict(
        self,
        test_features: pd.DataFrame,
        true_labels: list[int],
        index: FaissRAGIndex,
        output_csv: str,
        k: int = K_NEIGHBORS,
    ) -> pd.DataFrame:
        """Predice le classi e salva il CSV compatto richiesto dal progetto."""
        if self.model is None:
            raise RuntimeError(
                "Chiamare load() prima di predict()."
            )

        if len(test_features) != len(true_labels):
            raise ValueError(
                "Feature e label di test hanno lunghezze diverse."
            )

        if test_features.empty:
            raise ValueError("Il dataset di test è vuoto.")

        invalid_labels = sorted(
            {
                int(label)
                for label in true_labels
                if int(label) not in CLASS_NAME
            }
        )
        if invalid_labels:
            raise ValueError(
                f"Etichette di test non binarie: {invalid_labels}"
            )

        if not index.feature_names:
            raise RuntimeError(
                "L'indice FAISS non contiene i nomi delle feature."
            )

        missing_features = [
            name
            for name in index.feature_names
            if name not in test_features.columns
        ]
        if missing_features:
            raise ValueError(
                "Feature mancanti nel test: "
                f"{missing_features[:10]}"
            )

        # Garantisce lo stesso ordine di feature usato dall'indice FAISS.
        ordered_test_features = test_features.loc[
            :,
            index.feature_names,
        ].copy()

        variants = _VARIANTS[
            : max(
                1,
                min(int(PROMPT_VARIANTS), len(_VARIANTS)),
            )
        ]
        variant_count = len(variants)

        required_consensus_votes = min(
            max(1, int(MIN_CONSISTENT_LLM_VOTES)),
            variant_count,
        )

        full_results: list[dict[str, object]] = []
        csv_rows: list[dict[str, str]] = []

        iterator = zip(
            ordered_test_features.itertuples(
                index=False,
                name=None,
            ),
            true_labels,
        )

        for sample_index, (row_values, true_label) in enumerate(
            tqdm(
                iterator,
                total=len(ordered_test_features),
                desc="RAG prediction",
                unit="sample",
            )
        ):
            row = np.asarray(row_values, dtype=np.float64)
            true_label = int(true_label)

            # Ottieni predizione MV e vicini
            mv_prediction, mv_counts, natural_neighbors = index.majority_vote(
                row, k=k
            )

            # Ottieni il vettore standardizzato della query
            query_z = index.transform_vector(row)

            if self.debug:
                print(
                    f"\n[Sample {sample_index}] "
                    f"true={true_label}, "
                    f"MV={mv_prediction}, "
                    f"counts={mv_counts}, "
                    f"neighbors="
                    f"{[neighbor.label for neighbor in natural_neighbors]}"
                )

            llm_votes: list[int] = []
            raw_outputs: dict[str, str] = {}
            prompt_tokens: dict[str, int] = {}

            for variant in variants:
                vote, raw_output, token_count = self._generate_one(
                    query_z,
                    natural_neighbors,
                    index.feature_names,
                    variant,
                )
                raw_outputs[variant.name] = raw_output
                prompt_tokens[variant.name] = int(token_count)

                if vote is not None:
                    llm_votes.append(int(vote))

            (
                llm_candidate,
                llm_consensus_votes,
                vote_counts,
            ) = self._select_llm_candidate(
                llm_votes,
                required_consensus_votes,
            )

            (
                final_prediction,
                pred_type,
                decision_source,
                llm_override_applied,
                llm_unanimous,
                mv_margin,
            ) = self._decide_final_prediction(
                mv_prediction=mv_prediction,
                mv_counts=mv_counts,
                llm_candidate=llm_candidate,
                llm_votes=llm_votes,
                variant_count=variant_count,
            )

            if self.debug:
                print(
                    "      decision: "
                    f"LLM={llm_candidate}, "
                    f"unanimous={llm_unanimous}, "
                    f"MV_margin={mv_margin}, "
                    f"final={final_prediction}, "
                    f"source={decision_source}"
                )

            csv_rows.append(
                {
                    "prediction": CLASS_NAME[int(final_prediction)],
                    "true_label": CLASS_NAME[true_label],
                    "pred_type": pred_type,
                }
            )

            full_results.append(
                {
                    "prediction": CLASS_NAME[int(final_prediction)],
                    "true_label": CLASS_NAME[true_label],
                    "pred_type": pred_type,
                    "prediction_num": int(final_prediction),
                    "true_label_num": true_label,
                    "decision_source": decision_source,
                    "llm_prediction": (
                        CLASS_NAME[int(llm_candidate)]
                        if llm_candidate is not None
                        else "invalid"
                    ),
                    "llm_prediction_num": (
                        int(llm_candidate)
                        if llm_candidate is not None
                        else None
                    ),
                    "llm_valid_votes": int(len(llm_votes)),
                    "llm_consensus_votes": int(llm_consensus_votes),
                    "llm_goodware_votes": int(vote_counts.get(0, 0)),
                    "llm_malware_votes": int(vote_counts.get(1, 0)),
                    "llm_unanimous": bool(llm_unanimous),
                    "llm_override_applied": bool(llm_override_applied),
                    "mv_prediction": CLASS_NAME[int(mv_prediction)],
                    "mv_prediction_num": int(mv_prediction),
                    "mv_goodware_votes": int(mv_counts.get(0, 0)),
                    "mv_malware_votes": int(mv_counts.get(1, 0)),
                    "mv_margin": int(mv_margin),
                    "neighbor_labels": [
                        int(neighbor.label)
                        for neighbor in natural_neighbors
                    ],
                    "neighbor_similarities": [
                        float(neighbor.similarity)
                        for neighbor in natural_neighbors
                    ],
                    "llm_votes": list(llm_votes),
                    "llm_raw_outputs": raw_outputs,
                    "prompt_tokens": prompt_tokens,
                }
            )

        if not csv_rows:
            raise RuntimeError("Nessuna predizione prodotta.")

        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            csv_rows,
            columns=[
                "prediction",
                "true_label",
                "pred_type",
            ],
        ).to_csv(
            output_path,
            index=False,
        )

        return pd.DataFrame(full_results)