from __future__ import annotations

from src.embeddings.vertex_mock import MockGenerativeModel


class QueryExpander:

    _PROMPT_TEMPLATE = (
        "Rewrite and expand this search query into a comma-separated "
        "list of semantically related technical terms for better "
        "embedding search: {query}"
    )

    def __init__(self, model: MockGenerativeModel) -> None:
        self._model = model

    def expand(self, query: str) -> str:
        prompt = self._PROMPT_TEMPLATE.format(query=query)
        response = self._model.generate_content(prompt)
        return response.text
