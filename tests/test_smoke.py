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


def test_build_turn_input_resets_checkpoint_channels():
    from langgraph.types import Overwrite
    from src.orchestrator.graph import _build_turn_input

    init = _build_turn_input("新问题", [])
    assert isinstance(init["messages"], Overwrite)
    assert init["messages"].value == []
    assert isinstance(init["trace"], Overwrite)
    assert init["react_iteration"] == 0
    assert init["final_answer"] == ""
    assert init["attachments"] == []


def test_query_rewrite_skips_chitchat(monkeypatch):
    from src.rag.query_rewrite import improve_query

    def _fail(*_a, **_k):
        raise AssertionError("chitchat should not call LLM")

    monkeypatch.setattr("src.rag.query_rewrite.llm.chat_json", _fail)
    r = improve_query("你好")
    assert r["skipped"] is True
    assert r["rewritten"] == "你好"


def test_query_rewrite_preserves_attachments():
    from src.rag.query_rewrite import apply_query_rewrite, _preserve_attachments

    full = "玉米氮含量怎么估\n\n【用户本次上传的文档内容】\n一些正文"
    merged = _preserve_attachments(full, "高光谱遥感估算玉米叶片氮含量方法")
    assert "【用户本次上传的文档内容】" in merged
    assert "一些正文" in merged


def test_mcp_schema_from_json():
    from src.mcp.client import _schema_from_json

    schema = _schema_from_json({
        "type": "object",
        "properties": {"query": {"type": "string", "description": "检索词"}},
        "required": ["query"],
    })
    assert schema["query"]["required"] is True
    assert schema["query"]["type"] == "str"


def test_register_mcp_tools_disabled(monkeypatch):
    from config.settings import settings
    from src.mcp.client import register_mcp_tools

    monkeypatch.setattr(settings, "MCP_CLIENT_ENABLED", False)
    assert register_mcp_tools() == 0


def test_mcp_server_module_imports():
    from src.mcp.client import mcp_available
    if not mcp_available():
        import pytest
        pytest.skip("mcp 未安装")
    from src.mcp import server as mcp_server
    assert mcp_server.mcp.name == "DeepResearch"


def test_repair_tool_message_chain_fills_missing_tool():
    from src.llm.messages import repair_tool_message_chain

    broken = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}},
        ]},
    ]
    fixed = repair_tool_message_chain(broken)
    assert len(fixed) == 2
    assert fixed[1]["role"] == "tool"
    assert fixed[1]["tool_call_id"] == "call_1"


def test_build_dispatch_graph_compiles():
    from src.orchestrator.graph import build_dispatch_graph
    app = build_dispatch_graph()
    assert app is not None


def test_web_search_mock_mode(monkeypatch):
    from config.settings import settings
    from src.tools.web_search import search_web

    monkeypatch.setattr(settings, "WEB_SEARCH_PROVIDER", "mock")
    results = search_web("高光谱遥感", max_results=2)
    assert len(results) == 2
    assert results[0]["title"]
    assert results[0]["url"]


def test_web_search_tool_registered():
    tool = registry.get("web_search")
    assert tool is not None
    assert "max_results" in tool.schema


def test_parse_upload_image_and_doc():
    from src.multimodal.attachments import build_enhanced_question, has_images, parse_upload

    img = parse_upload("test.png", b"\x89PNG\r\n\x1a\n" + b"x" * 40)
    assert img["kind"] == "image"
    assert has_images([img])

    doc = parse_upload("note.txt", "高光谱遥感".encode("utf-8"))
    assert doc["kind"] == "document"
    assert "高光谱" in doc["text"]

    q = build_enhanced_question("请总结", [doc])
    assert "用户本次上传的文档内容" in q


def test_rag_dual_layer_eval():
    from src.evaluation.rag_eval import evaluate_rag_dual_layer

    dual = evaluate_rag_dual_layer({
        "trace": [{"step": "tool", "meta": {"retrieval_scores": [0.72, 0.68]}}],
        "grounding": {"support_rate": 0.83, "claims_total": 6, "grounded": True},
    })
    assert dual["mode"] == "dual"
    assert dual["layer1"]["pass"] is True
    assert dual["layer2"]["pass"] is True
    assert dual["overall_pass"] is True


def test_build_run_metrics_grounding():
    from src.evaluation.run_metrics import build_run_metrics

    m = build_run_metrics({
        "route": "deep_research",
        "grounding": {"support_rate": 0.83, "grounded": True, "claims_total": 6},
        "trace": [{"step": "tool", "meta": {"retrieval_scores": [0.7, 0.75]}}],
        "react_iterations": 0,
    })
    assert m["primary_name"] == "事实支撑率"
    assert m["primary_display"] == "83%"
    assert m["primary_pass"] is True
    assert m["rag_eval"]["mode"] == "dual"
    assert m["rag_dual_pass"] is True


def test_build_run_metrics_retrieval():
    from src.evaluation.run_metrics import build_run_metrics

    m = build_run_metrics({
        "route": "super",
        "grounding": {},
        "trace": [{"step": "tool", "meta": {"retrieval_scores": [0.72, 0.68, 0.81]}}],
        "react_iterations": 2,
    })
    assert m["primary_name"] == "检索相关度"
    assert m["retrieval_avg_score"] == 0.737


def test_retriever_sees_ingested_kb_from_any_cwd(monkeypatch):
    """模拟从 frontend/ 子目录启动时，路径仍指向项目根下的 chroma。"""
    from pathlib import Path
    frontend = Path(__file__).resolve().parent.parent / "frontend"
    monkeypatch.chdir(frontend)
    from config.settings import PROJECT_ROOT, _abs_path
    chroma = _abs_path("data/chroma")
    assert chroma == str((PROJECT_ROOT / "data/chroma").resolve())
