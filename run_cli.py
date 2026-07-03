"""
run_cli.py — 命令行入口（不依赖前端，方便快速验证整条链路）
用法：python run_cli.py "你的问题"
"""
import sys
from src.orchestrator.graph import run_agent


def main() -> None:
    q = sys.argv[1] if len(sys.argv) > 1 else "请总结知识库中的核心内容"
    result = run_agent(q)
    print("\n" + "=" * 60)
    print("最终答案：\n")
    print(result["answer"])
    print("\n" + "-" * 60)
    print("执行时间线：")
    for s in result["trace"]:
        print(f"  [{s['step']}] {s['detail']}")
    if result.get("react_iterations"):
        print(f"\nReAct 轮次: {result['react_iterations']}")
    if result.get("thread_id"):
        print(f"会话 thread_id: {result['thread_id']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
