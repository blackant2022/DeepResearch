"""
tests/test_smoke.py — 冒烟测试（不需要真实API Key也能测的部分）
用 pytest 运行：pytest tests/ -v
"""
from src.tools.builtin_tools import CalculatorTool
from src.tools.base import registry
from src.middleware.pipeline import pipeline
from src.memory.working_memory import WorkingMemory


def test_calculator_ok():
    r = CalculatorTool()(expression="(3+5)*2")
    assert r.ok and r.output == 16


def test_calculator_guardrail_blocks_danger():
    tool = registry.get("calculator")
    r = pipeline.invoke(tool, expression="__import__('os')")
    assert not r.ok and r.error_type == "guardrail"


def test_calculator_invalid_args_no_retry():
    tool = registry.get("calculator")
    r = pipeline.invoke(tool)  # 缺 expression
    assert not r.ok and r.error_type == "invalid_args"


def test_working_memory_budget_keeps_facts():
    wm = WorkingMemory(max_chars=50)
    wm.add_fact("t", "F" * 40)
    for i in range(10):
        wm.add("t", "S" * 20, kind="scratch")
    # fact 不应被淘汰
    assert any(it.kind == "fact" for it in wm._items)


def test_registry_manifest_lists_tools():
    m = registry.manifest()
    assert "knowledge_search" in m and "calculator" in m and "kb_overview" in m


def test_kb_catalog_vs_summary():
    from src.rag.kb_utils import is_kb_catalog_question, is_kb_knowledge_summary

    assert is_kb_catalog_question("说一说现有知识库里的内容")
    assert not is_kb_catalog_question("总结我已经上传到知识库中的核心知识")

    assert is_kb_knowledge_summary("总结我已经上传到知识库中的核心知识")
    assert is_kb_knowledge_summary("总结知识库里关于高光谱的核心知识")
    assert not is_kb_knowledge_summary("3+5等于多少")


def test_chitchat_detected():
    from src.orchestrator.router import is_chitchat

    assert is_chitchat("你是谁？")
    assert is_chitchat("你好")
    assert not is_chitchat("高光谱如何估算玉米氮含量")
    from src.rag.kb_utils import is_kb_catalog_question, is_kb_knowledge_summary

    assert is_kb_catalog_question("说一说现有知识库里的内容")
    assert not is_kb_catalog_question("总结我已经上传到知识库中的核心知识")

    assert is_kb_knowledge_summary("总结我已经上传到知识库中的核心知识")
    assert is_kb_knowledge_summary("总结知识库里关于高光谱的核心知识")
    assert not is_kb_knowledge_summary("3+5等于多少")


def test_tools_registered_on_import():
    import src.tools  # noqa: F401
    from src.tools.base import registry
    for name in ("knowledge_search", "kb_overview", "calculator"):
        assert registry.get(name) is not None, f"工具 {name} 未注册"


def test_build_kb_overview_has_docs():
    from src.rag.kb_utils import build_kb_overview_answer
    ans = build_kb_overview_answer()
    if "为空" not in ans:
        assert "篇文档" in ans
        assert "文档列表" in ans


def test_settings_paths_are_absolute():
    from config.settings import PROJECT_ROOT, settings
    assert settings.RAG_CHROMA_DIR.startswith(str(PROJECT_ROOT))
    assert settings.DOCS_DIR.startswith(str(PROJECT_ROOT))


def test_openai_tool_schema():
    from src.tools.builtin_tools import CalculatorTool, KnowledgeSearchTool
    calc = CalculatorTool().to_openai_schema()
    assert calc["function"]["name"] == "calculator"
    assert "expression" in calc["function"]["parameters"]["properties"]
    kb = KnowledgeSearchTool().to_openai_schema()
    assert kb["function"]["name"] == "knowledge_search"


def test_execute_tool_calls():
    from src.orchestrator.nodes.tool_node import execute_tool_calls
    from src.memory.working_memory import WorkingMemory

    wm = WorkingMemory()
    msgs, trace = execute_tool_calls([{
        "id": "call_test",
        "type": "function",
        "function": {"name": "calculator", "arguments": '{"expression": "2+3"}'},
    }], wm)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "tool"
    assert "5" in msgs[0]["content"]
    assert trace[0]["step"] == "tool"


def test_route_maps_search_to_super():
    from src.orchestrator.graph import route_by_agent
    assert route_by_agent({"route": "search"}) == "super"
    assert route_by_agent({"route": "super"}) == "super"


def test_super_agent_initial_messages():
    from src.agents.super_agent import _SUPER_SYSTEM

    assert "超级智能体" in _SUPER_SYSTEM
    assert "knowledge_search" in _SUPER_SYSTEM


def test_working_memory_snapshot_roundtrip():
    wm = WorkingMemory()
    wm.add_fact("search", "test fact")
    wm.add("tool", "obs", kind="observation", tool="calc")
    restored = WorkingMemory.from_snapshot(wm.export_snapshot())
    assert len(restored.facts()) == 1
    assert len(restored.observations()) == 1


def test_build_dispatch_graph_compiles():
    from src.orchestrator.graph import build_dispatch_graph
    app = build_dispatch_graph()
    assert app is not None


def test_retriever_sees_ingested_kb_from_any_cwd(monkeypatch):
    """模拟从 frontend/ 子目录启动时，路径仍指向项目根下的 chroma。"""
    from pathlib import Path
    frontend = Path(__file__).resolve().parent.parent / "frontend"
    monkeypatch.chdir(frontend)
    from config.settings import PROJECT_ROOT, _abs_path
    chroma = _abs_path("data/chroma")
    assert chroma == str((PROJECT_ROOT / "data/chroma").resolve())
