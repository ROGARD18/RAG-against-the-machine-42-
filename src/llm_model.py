from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from src.models import MinimalSource
import re


class LlmModel:
    def __init__(self):
        self.model_id = "Qwen/Qwen3-0.6B"
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

    def generate(self, query: str, chunks: List) -> str:
        self._load_model()

        context = "\n\n".join(
            [f"[Document {i}]\n{self._get_chunk_text(chunk)}"
             for i, chunk in enumerate(chunks)]
        )

        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a strict, expert technical assistant. "
                    "You must answer the user's question using "
                    "ONLY the provided Context. "
                    "Do not use markdown, code blocks, lists, or "
                    "any formatting symbols. "
                    "Answer in a single plain text paragraph. "
                    "If the answer cannot be confidently deduced "
                    "from the Context, you must reply exactly with:"
                    " 'Information not found in context.' "
                    "Be highly concise and direct."
                )
            },
            {
                "role": "user",
                "content": "Where can I find information about "
                "using generative models in vLLM?"
            },
            {
                "role": "assistant",
                "content": "Information about using generative"
                " models can be found on the generative models "
                "page as referenced in the supported models documentation."
            },
            {
                "role": "user",
                "content": "What conditions must be met for ModelRunner "
                "to use CUDA graphs?"
            },
            {
                "role": "assistant",
                "content": "Two conditions must be met. First, prefill_meta"
                " must be None. Second, decode_meta.use_cuda_graph must be "
                "True. When both are satisfied, the ModelRunner uses the "
                "virtual engine graph runners instead of the regular model."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        input_length = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            repetition_penalty=1.15,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_tokens = outputs[0][input_length:]
        raw_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        clean_answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()
        return clean_answer

    def _get_chunk_text(self, chunk) -> str:
        with open(chunk.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(content[chunk.first_character_index:chunk.last_character_index])
        return content[chunk.first_character_index:chunk.last_character_index]
