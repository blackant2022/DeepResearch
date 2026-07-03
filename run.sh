#!/bin/bash
# run.sh — 一键启动（Linux/Mac）；Windows 见 README 的 PowerShell 命令
set -e
[ ! -f .env ] && cp .env.example .env && echo "已生成 .env，请填入 DEEPSEEK_API_KEY 后重跑" && exit 1
pip install -r requirements.txt -q
mkdir -p data/docs data/chroma data/ltm_chroma
echo "① 构建知识库…" && python -m src.rag.ingest ./data/docs || true
echo "② 启动前端 → http://localhost:8501"
streamlit run frontend/streamlit_app.py
