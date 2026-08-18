from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from src.models import MinimalSource


class LlmModel:

    def __init__(self):
        self.model_id = "Qwen/Qwen3-0.6B"
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

    def _get_chunk_text(self, source: MinimalSource) -> str:
        with open(source.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return content[source.first_character_index:source.last_character_index]

    def generate(self, query: str, chunks: List) -> str:
        self._load_model()

        context: str = "\n".join(
            [self._get_chunk_text(chunk) for chunk in chunks])

        prompt = ("You are a strict, expert technical assistant. "
                  f"You must answer the user's question: {query} using ONLY the "
                  f"provided context:\n{context}\n\n. "
                  "Do not use markdown, code blocks, lists, or any "
                  "formatting symbols. Answer in a single plain text "
                  "paragraph. "
                  "If the answer cannot be confidently deduced from the "
                  "Context, you must reply exactly with: "
                  "'Information not found in context.' "
                  "Be highly concise and direct.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=200)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
