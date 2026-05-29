"""Grounded answer generation via the OpenAI API.

The LLM is instructed to answer *only* from the retrieved passages and to cite
them as ``[1]``, ``[2]`` ... matching the passage order it was given. If no API
key is configured the pipeline still works as a retrieval-only system; this
module simply reports that generation is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

SYSTEM_PROMPT = (
    "You are a meticulous financial analyst assistant. Answer the user's "
    "question using ONLY the numbered context passages provided. Cite every "
    "claim with bracketed passage numbers like [1] or [2][3]. If the passages "
    "do not contain the answer, say you cannot find it in the provided "
    "documents — do not use outside knowledge or guess. Be concise and precise "
    "with figures, units, and time periods."
)


@dataclass
class Passage:
    """A passage handed to the LLM as context."""

    doc_id: str
    text: str
    title: str = ""


@dataclass
class GenerationResult:
    answer: str
    used: bool = True                 # whether the LLM actually ran
    model: str = ""
    error: str = ""
    citations: List[int] = field(default_factory=list)


def build_context(passages: List[Passage], max_chars: int = 2000) -> str:
    """Render passages into a numbered context block for the prompt."""
    blocks = []
    for i, p in enumerate(passages, start=1):
        header = f"[{i}] (doc_id: {p.doc_id}"
        if p.title:
            header += f", title: {p.title}"
        header += ")"
        blocks.append(f"{header}\n{p.text[:max_chars]}")
    return "\n\n".join(blocks)


def build_messages(question: str, passages: List[Passage], max_chars: int = 2000) -> List[dict]:
    """Construct the chat messages (pure function, unit-testable)."""
    context = build_context(passages, max_chars=max_chars)
    user = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer (with citations):"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class OpenAIGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.use_generation and self.cfg.openai_api_key)

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI  # lazy

            self._client = OpenAI(api_key=self.cfg.openai_api_key)
        return self._client

    def generate(self, question: str, passages: List[Passage], temperature: float = 0.0) -> GenerationResult:
        if not self.cfg.use_generation:
            return GenerationResult(answer="", used=False, error="generation disabled")
        if not self.cfg.openai_api_key:
            return GenerationResult(
                answer="",
                used=False,
                error="No OPENAI_API_KEY set — showing retrieved passages only.",
            )
        if not passages:
            return GenerationResult(
                answer="No relevant passages were retrieved for this question.",
                used=False,
            )

        messages = build_messages(question, passages, max_chars=self.cfg.max_passage_chars)
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.llm_model,
                messages=messages,
                temperature=temperature,
            )
            answer = resp.choices[0].message.content.strip()
            return GenerationResult(answer=answer, used=True, model=self.cfg.llm_model)
        except Exception as exc:  # network / auth / quota
            return GenerationResult(answer="", used=False, error=f"LLM call failed: {exc}")
