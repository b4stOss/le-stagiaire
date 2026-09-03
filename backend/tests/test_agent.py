from types import SimpleNamespace

from app.agent import ToolCallAccumulator, content_to_text, resolve_citations
from app.retrieval import RetrievedChunk


def _chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        company="TotalEnergies",
        title="URD",
        fiscal_year=2025,
        page_start=10 + chunk_id,
        page_end=10 + chunk_id,
        section_path="Debt",
        content=f"chunk {chunk_id}",
        score=0.5,
    )


SEEN = {1: _chunk(1), 2: _chunk(2), 3: _chunk(3)}


def test_citations_are_unique_and_in_order_of_first_appearance():
    citations = resolve_citations("Debt fell [c2]. Cash rose [c1, c2] and [c3].", SEEN)
    assert [c["chunk_id"] for c in citations] == [2, 1, 3]
    assert citations[0]["page_start"] == 12
    assert citations[0]["quote"] == "chunk 2"


def test_citations_the_model_never_received_are_dropped():
    citations = resolve_citations("Invented [c999] and real [c1].", SEEN)
    assert [c["chunk_id"] for c in citations] == [1]


def test_answer_without_markers_has_no_citations():
    assert resolve_citations("The information is not in the reports.", SEEN) == []


def test_content_to_text_flattens_mistral_reference_parts():
    parts = [
        SimpleNamespace(type="text", text="Net debt was 12 bn"),
        SimpleNamespace(type="reference", reference_ids=["c7", "[c8]"]),
        SimpleNamespace(type="text", text="."),
    ]
    assert content_to_text(parts) == "Net debt was 12 bn [c7][c8]."
    assert content_to_text("plain [c1]") == "plain [c1]"
    assert content_to_text(None) == ""


def _delta(index, id="", name="", arguments: str | dict = ""):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


def test_tool_call_accumulator_reassembles_streamed_fragments():
    acc = ToolCallAccumulator()
    acc.add([_delta(0, id="call_a", name="search_filings", arguments='{"query": "net')])
    acc.add([_delta(1, id="call_b", name="search_filings", arguments={"query": "capex"})])
    acc.add([_delta(0, arguments=' debt"}')])
    calls = acc.calls()
    assert [c["id"] for c in calls] == ["call_a", "call_b"]
    assert calls[0]["function"]["arguments"] == '{"query": "net debt"}'
    assert calls[1]["function"]["arguments"] == '{"query": "capex"}'
