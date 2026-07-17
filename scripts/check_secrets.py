"""
scripts/check_secrets.py — 推送 / 提交前扫描疑似密钥泄露

用法（仓库根目录）：
  python scripts/check_secrets.py
  python scripts/check_secrets.py --staged   # 只扫暂存区

退出码：发现疑似真实密钥时为 1。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 占位符放行；疑似真实密钥拦截
_PLACEHOLDER = re.compile(
    r"(your[-_]?key|xxxx|example|placeholder|changeme|dummy|test[-_]?key)",
    re.I,
)

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI/DeepSeek-like key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("Tavily key", re.compile(r"\btvly-[A-Za-z0-9_\-]{10,}\b")),
    ("Generic API assignment", re.compile(
        r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"
    )),
]

_SKIP_DIR = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    "data", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}
_SKIP_SUFFIX = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".bin", ".sqlite3"}
_ALLOW_FILES = {".env.example"}  # 模板允许 placeholder


def _iter_files(staged: bool) -> list[Path]:
    if staged:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        return [ROOT / p for p in out.splitlines() if p.strip()]

    files: list[Path] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIR for part in p.parts):
            continue
        if p.suffix.lower() in _SKIP_SUFFIX:
            continue
        if p.name == ".env" or p.name.startswith(".env."):
            # .env 本身应被 ignore；若出现在扫描结果里也跳过内容（不读）
            if p.name != ".env.example":
                continue
        files.append(p)
    return files


def _is_placeholder(text: str) -> bool:
    return bool(_PLACEHOLDER.search(text))


def scan(staged: bool = False) -> list[str]:
    hits: list[str] = []
    for path in _iter_files(staged):
        rel = path.relative_to(ROOT).as_posix()
        if path.name in _ALLOW_FILES or rel.endswith(".env.example"):
            # 仍检查是否混入过长真实 key
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for label, pat in _PATTERNS[:2]:
                for m in pat.finditer(text):
                    token = m.group(0)
                    if _is_placeholder(token) or "xxxx" in token.lower() or "your-key" in token.lower():
                        continue
                    if len(token) >= 24 and not _is_placeholder(token):
                        hits.append(f"{rel}: {label} → {token[:8]}…（疑似真实密钥）")
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for label, pat in _PATTERNS:
            for m in pat.finditer(text):
                token = m.group(0)
                if _is_placeholder(token):
                    continue
                # settings 空 SecretStr / 注释放过短占位
                if "SecretStr(\"\")" in text and "sk-" not in token:
                    continue
                if token.startswith("sk-your") or token.startswith("tvly-xxxx"):
                    continue
                hits.append(f"{rel}: {label} → {token[:12]}…")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="扫描仓库中的疑似 API Key")
    parser.add_argument("--staged", action="store_true", help="仅扫描 git 暂存区")
    args = parser.parse_args()

    # 硬规则：.env 绝不能被跟踪
    tracked = subprocess.check_output(
        ["git", "ls-files", ".env", ".env.local", ".env.production"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    ).strip()
    if tracked:
        print("ERROR: 敏感文件已被 Git 跟踪：")
        print(tracked)
        print("请执行: git rm --cached .env  并确认 .gitignore 已忽略 .env")
        return 1

    hits = scan(staged=args.staged)
    if hits:
        print("ERROR: 发现疑似敏感信息，禁止提交/推送：")
        for h in hits:
            print(" -", h)
        print("\n请改为写入本地 .env（已 gitignore），并轮换已泄露的 Key。")
        return 1

    print("OK: 未发现疑似真实 API Key（.env 未被跟踪）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
