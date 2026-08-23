"""
backends/hf_llm.py — Free local LLM backend using HuggingFace Transformers
============================================================================

No API key required. Models run entirely on your GPU (or CPU).

Supported models (set in config.py → GenerationConfig.model_key):

  "phi3"    → microsoft/phi-3-mini-4k-instruct  (recommended, ~8 GB VRAM)
  "mistral" → mistralai/Mistral-7B-Instruct-v0.2 (best quality, ~14 GB VRAM)
  "flan"    → google/flan-t5-large               (lightweight seq2seq, ~3 GB)
  "smollm"  → HuggingFaceTB/SmolLM2-1.7B-Instruct (~4 GB, fastest)

Usage
-----
    from backends.hf_llm import HFLLMBackend
    llm = HFLLMBackend("microsoft/phi-3-mini-4k-instruct", device="cuda")
    answer = llm.generate(prompt="What is the termination clause?",
                          context="...retrieved passages...",
                          max_new_tokens=256)
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a legal document assistant. Answer the question using only the "
    "provided passages. If multiple passages conflict, the later passage takes "
    "precedence (document-order authority rule). Be concise and precise. "
    "If the answer is not in the passages, reply 'Not found'."
)


class HFLLMBackend:
    """Wrapper around a HuggingFace causal-LM or seq2seq model."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
    ):
        self.model_name     = model_name
        self.device         = device
        self.load_in_4bit   = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self._pipeline      = None   # lazy-loaded

    # ── Lazy load ─────────────────────────────────────────────────────────────

    def _load(self):
        if self._pipeline is not None:
            return

        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch

        is_flan = "flan" in self.model_name.lower()
        task    = "text2text-generation" if is_flan else "text-generation"

        kwargs: dict = {
            "model":            self.model_name,
            "device_map":       "auto" if self.device == "cuda" else self.device,
            "torch_dtype":      torch.float16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": True,
        }

        # 4-bit quantisation — keeps Mistral-7B inside T4's 16 GB
        if self.load_in_4bit and self.device == "cuda" and not is_flan:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("4-bit quantisation enabled for %s", self.model_name)
            except ImportError:
                logger.warning("bitsandbytes not found — loading in fp16.")

        logger.info("Loading LLM: %s", self.model_name)
        self._pipeline = pipeline(task, **kwargs)
        self._is_flan  = is_flan
        logger.info("✓ LLM loaded.")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        context:  str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        self._load()
        max_new_tokens = max_new_tokens or self.max_new_tokens

        if self._is_flan:
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Passages:\n{context}\n\n"
                f"Question: {question}\nAnswer:"
            )
            out = self._pipeline(prompt, max_new_tokens=max_new_tokens)
            return out[0]["generated_text"].strip()

        # Chat-template models (Phi-3, Mistral, SmolLM)
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Passages:\n{context}\n\nQuestion: {question}"},
        ]
        tokenizer = self._pipeline.tokenizer
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        out = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Strip the prompt from the output
        full_text = out[0]["generated_text"]
        return full_text[len(prompt):].strip()

    def batch_generate(
        self,
        items: list[dict],          # each: {"question": str, "context": str}
        max_new_tokens: Optional[int] = None,
    ) -> list[str]:
        """Generate answers for a list of {question, context} dicts."""
        return [
            self.generate(i["question"], i["context"], max_new_tokens)
            for i in items
        ]
