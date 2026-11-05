from copilot import agent
from copilot.agent import Copilot, normalize_citations, is_german
from copilot.tools import Tools
from eval.run import facts_ok, hit_rank


class FakeRetriever:
    def search(self, query, k, mode, sources=None):
        c = {"doc": "UR3e_User_Manual_en_SW5.24.pdf", "page": 60, "title": "7.7. Freedrive", "id": "x",
             "text": "To use the 3PE TP button: rapidly light-press, release, light-press again and hold."}
        return [(c, 1.0)]


def test_error_lookup_exact_and_group():
    t = Tools(FakeRetriever())
    _, src = t.lookup_error_code("c153a1")
    assert src[0]["id"] == "err:C153A1" and "Check payload" in src[0]["text"]
    note, src = t.lookup_error_code("C153A7")
    assert "not listed" in note and src[0]["id"] == "err:C153"
    note, src = t.lookup_error_code("C9999A1")
    assert src == [] and "not in" in note


def test_sql_is_read_only():
    t = Tools(FakeRetriever())
    for bad in ("DROP TABLE trials", "DELETE FROM trials", "SELECT 1; DROP TABLE trials", "PRAGMA table_info(x)"):
        note, src = t.query_trials(bad)
        assert src == [] and "read-only" in note
    _, src = t.query_trials("SELECT COUNT(*) FROM trials WHERE grasp_success='True'")
    assert "\n1" in src[0]["text"]


def test_parameter_lookup_never_guesses():
    t = Tools(FakeRetriever())
    _, src = t.get_parameter("max_guide_force")
    assert src[0]["text"].startswith("safety_monitor.max_guide_force = 150.0")
    note, src = t.get_parameter("max_guide_velocity")  # the name a model guesses for max_guide_speed
    # a near miss returns no value (cycle 1: a guessed 150 N was reported for UR manual limits), only names
    assert src == [] and "safety_monitor.max_guide_speed" in note


def test_citation_normalisation_and_language():
    assert normalize_citations("Hold it [1, 2].") == "Hold it [1][2]."
    assert normalize_citations("See [1-3].") == "See [1][2][3]."
    assert is_german("Wie aktiviere ich Freedrive am Teach Pendant?")
    assert not is_german("How do I enable Freedrive?")


def _fake_chat(replies):
    it = iter(replies)
    return lambda msgs, model, tools=None, num_ctx=8192: {"message": next(it)}


def test_guard_removes_invented_citations_and_refuses_uncited(monkeypatch):
    cp = Copilot(FakeRetriever(), pipeline="rag")
    monkeypatch.setattr(agent, "chat", _fake_chat([{"content": "Light-press and hold [1][7]."}]))
    r = cp.ask("How do I enable Freedrive?")
    assert r["answer"] == "Light-press and hold [1]." and r["cited"] == [1]
    assert any(f.startswith("invalid_citations") for f in r["flags"])
    monkeypatch.setattr(agent, "chat", _fake_chat([{"content": "Just press it."}, {"content": "Press it."}]))
    r = cp.ask("How do I enable Freedrive?")
    assert r["refused"] and "uncited_refused" in r["flags"]


def test_question_is_always_retrieved_first(monkeypatch):
    cp = Copilot(FakeRetriever(), pipeline="agent")
    monkeypatch.setattr(agent, "chat", _fake_chat([{"content": "Light-press, release, light-press and hold [1]."}]))
    r = cp.ask("How do I enable Freedrive?")
    assert r["tool_calls"][0]["first"] and r["cited"] == [1] and not r["refused"]


def test_error_lookup_is_gated_to_codes_in_the_question(monkeypatch):
    cp = Copilot(FakeRetriever(), pipeline="agent")
    monkeypatch.setattr(agent, "chat", _fake_chat([
        {"content": "", "tool_calls": [{"function": {"name": "lookup_error_code", "arguments": {"code": "C153A1"}}}]},
        {"content": "Check the connections [2]."}]))
    r = cp.ask("Our tool output keeps switching off. What does the error directory say?")
    assert r["tool_calls"][1]["gated"] == "code not in question" and r["tool_calls"][2]["ran"] == "search_docs"


def test_relevance_gate_refuses_when_nothing_matches(monkeypatch):
    class Far(FakeRetriever):
        def search(self, *a, **k):
            return [(c, -9.0) for c, _ in super().search(*a, **k)]
    cp = Copilot(Far(), pipeline="rag")
    monkeypatch.setattr(agent, "RELEVANCE", 0.24)  # off by default since cycle 4; the mechanism is still tested
    monkeypatch.setattr(agent, "chat", _fake_chat([{"content": "The KUKA connects like this [1]."}]))
    r = cp.ask("How do I connect the UR3e to a KUKA controller?")
    assert r["refused"] and r["flags"][0].startswith("relevance_gate")


def test_refusal_detection_is_strict():
    from copilot.agent import is_refusal
    assert is_refusal("I could not find this in the documentation I have. [1]")
    assert is_refusal("Das konnte ich in der vorliegenden Dokumentation nicht finden.")
    assert is_refusal("Das kann ich in der vorliegenden Dokumentation nicht finden. [1]")
    assert not is_refusal("Allow 30 seconds [2]. The manual gives no information on faster discharge.")


def test_agent_runs_tool_calls(monkeypatch):
    cp = Copilot(FakeRetriever(), pipeline="agent")
    monkeypatch.setattr(agent, "chat", _fake_chat([
        {"content": "", "tool_calls": [{"function": {"name": "lookup_error_code", "arguments": {"code": "C153A1"}}}]},
        {"content": "The shoulder joint could not follow the path [1]; check the payload [1]."}]))
    r = cp.ask("What does C153A1 mean?")
    assert r["tool_calls"][1]["tool"] == "lookup_error_code" and r["sources"][1]["page"] == 34


def test_grading_helpers():
    ok, _ = facts_ok("Tighten four M6 bolts to 9 Nm.", [["m6"], ["9 nm", "9nm"]])
    assert ok
    ok, missing = facts_ok("Das Kabel ist 4,5 m lang.", [["4.5 m"], ["teach"]])
    assert not ok and missing == [["teach"]]
    hits = [({"doc": "a", "page": 1}, 1.0), ({"doc": "b", "page": 2}, 0.5)]
    assert hit_rank(hits, [["b", 2]]) == 2 and hit_rank(hits, [["c", 0]]) is None


def test_other_robot_is_refused_before_retrieval(monkeypatch):
    cp = Copilot(FakeRetriever(), pipeline="agent")
    monkeypatch.setattr(agent, "chat", _fake_chat([{"content": "The UR3e is IP54 [1]."}]))
    r = cp.ask("What is the IP rating of the KUKA LBR iiwa?")
    assert r["refused"] and r["flags"] == ["other_robot"] and r["tool_calls"] == []
    r = cp.ask("How do I connect the UR3e to a KUKA controller?")  # names the UR3e: not caught by this rule
    assert "other_robot" not in r["flags"]


def test_brackets_in_passages_are_not_citations(monkeypatch):
    class Ranged(FakeRetriever):
        def search(self, *a, **k):
            c = {"doc": "ROS_INTERFACE.md", "page": 6, "title": "servoj_gain", "id": "y", "text": "Range: [100 - 3000]"}
            return [(c, 1.0)]
    seen = []
    def fake(messages, model, tools=None, num_ctx=8192):
        seen.append(messages[-1]["content"])
        return {"message": {"content": "The range is 100 to 3000 [1]."}}
    monkeypatch.setattr(agent, "chat", fake)
    Copilot(Ranged(), pipeline="rag").ask("What is the range of servoj_gain?")
    assert "Range: (100 - 3000)" in seen[0] and "[100" not in seen[0]


def test_bare_citation_is_retried(monkeypatch):
    cp = Copilot(FakeRetriever(), pipeline="rag")
    monkeypatch.setattr(agent, "chat", _fake_chat([{"content": "[1]"}, {"content": "Light-press and hold [1]."}]))
    r = cp.ask("How do I enable Freedrive?")
    assert "empty_retry" in r["flags"] and r["answer"] == "Light-press and hold [1]."
