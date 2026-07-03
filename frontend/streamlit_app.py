"""
frontend/streamlit_app.py — DeepResearch 前端
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import streamlit as st  # noqa: E402

# ---- Design tokens：统一浅色主题，避免与 Streamlit 默认深色栏冲突 ----
THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --color-primary: #1E3A5F;
    --color-on-primary: #FFFFFF;
    --color-secondary: #2563EB;
    --color-accent: #059669;
    --color-background: #F1F5F9;
    --color-foreground: #0F172A;
    --color-muted: #E2E8F0;
    --color-border: #CBD5E1;
    --color-destructive: #DC2626;
    --color-muted-text: #475569;
    --color-surface: #FFFFFF;
    --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.08);
    --shadow-md: 0 4px 14px rgba(15, 23, 42, 0.1);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --font-heading: 'Inter', system-ui, sans-serif;
    --font-body: 'Inter', system-ui, sans-serif;
    --transition: 180ms ease;
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

html, body, [class*="css"] {
    font-family: var(--font-body);
    color: var(--color-foreground);
}

.stApp {
    background: var(--color-background) !important;
}

/* 统一顶栏 / 底栏为浅色，去掉黑条 */
header[data-testid="stHeader"] {
    background: var(--color-surface) !important;
    border-bottom: 1px solid var(--color-border);
}

[data-testid="stToolbar"] {
    background: transparent !important;
}

[data-testid="stBottom"] {
    background: var(--color-background) !important;
    border-top: 1px solid var(--color-border) !important;
    padding-bottom: 0.5rem;
}

[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
}

.stChatFloatingInputContainer {
    background: transparent !important;
    padding-bottom: 0.5rem;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 780px;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: var(--color-surface);
    border-right: 1px solid var(--color-border);
}

[data-testid="stSidebar"] [data-testid="stMarkdown"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdown"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {
    font-family: var(--font-heading);
    color: var(--color-primary);
    font-weight: 600;
    letter-spacing: -0.01em;
}

[data-testid="stSidebar"] .stMetric {
    background: var(--color-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: 0.75rem 1rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow var(--transition), border-color var(--transition);
}

[data-testid="stSidebar"] .stMetric:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--color-secondary);
}

[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    font-size: 0.8rem;
    color: var(--color-muted-text);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-family: var(--font-heading);
    font-size: 1.75rem;
    color: var(--color-secondary);
    font-weight: 700;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--color-muted-text) !important;
}

/* ---- Buttons ---- */
div[data-testid="stButton"] > button {
    font-family: var(--font-body);
    font-weight: 600;
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border);
    background: var(--color-surface);
    color: var(--color-foreground);
    padding: 0.55rem 1rem;
    transition: background var(--transition), border-color var(--transition),
                color var(--transition), box-shadow var(--transition);
    cursor: pointer;
}

div[data-testid="stButton"] > button:hover:not(:disabled) {
    background: var(--color-muted);
    border-color: var(--color-secondary);
    box-shadow: var(--shadow-sm);
}

div[data-testid="stButton"] > button:focus-visible {
    outline: 2px solid var(--color-secondary);
    outline-offset: 2px;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: var(--color-primary) !important;
    color: var(--color-on-primary) !important;
    border-color: var(--color-primary) !important;
}

div[data-testid="stButton"] > button[kind="primary"]:hover:not(:disabled) {
    background: #152a45 !important;
    border-color: #152a45 !important;
}

div[data-testid="stButton"] > button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
}

/* 侧边栏：最后一个按钮 = 清空对话（次要/危险） */
[data-testid="stSidebar"] div[data-testid="stButton"]:last-of-type > button {
    background: var(--color-surface) !important;
    color: var(--color-muted-text) !important;
    border: 1px solid var(--color-border) !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"]:last-of-type > button:hover:not(:disabled) {
    background: #FEF2F2 !important;
    color: var(--color-destructive) !important;
    border-color: var(--color-destructive) !important;
}

/* ---- Chat ---- */
div[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: 0.25rem 0;
}

div[data-testid="stChatMessage"][data-testid*="user"],
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse;
}

div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: 0.85rem 1.1rem;
    box-shadow: var(--shadow-sm);
    line-height: 1.65;
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
    background: var(--color-primary);
    color: var(--color-on-primary);
    border-color: var(--color-primary);
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] p,
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] li {
    color: var(--color-on-primary);
}

[data-testid="stChatInput"] {
    background: transparent !important;
    border-top: none !important;
    padding-top: 0.5rem;
}

[data-testid="stChatInput"] > div {
    background: var(--color-surface) !important;
    border: 1px solid var(--color-border) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm);
}

[data-testid="stChatInput"] textarea {
    font-family: var(--font-body);
    color: var(--color-foreground) !important;
    background: var(--color-surface) !important;
    border: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--color-muted-text) !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] textarea:focus {
    box-shadow: none !important;
}

[data-testid="stChatInput"] button {
    background: var(--color-primary) !important;
    color: white !important;
}

/* ---- File uploader ---- */
[data-testid="stFileUploader"] section {
    border: 2px dashed var(--color-border);
    border-radius: var(--radius-md);
    background: #F8FAFC;
    padding: 1.25rem 1rem;
    transition: border-color var(--transition), background var(--transition);
}

[data-testid="stFileUploader"] section:hover {
    border-color: var(--color-secondary);
    background: var(--color-surface);
}

[data-testid="stFileUploader"] section * {
    color: var(--color-foreground) !important;
}

[data-testid="stFileUploader"] small {
    color: var(--color-muted-text) !important;
    font-size: 0.85rem !important;
}

[data-testid="stFileUploader"] button {
    background: var(--color-primary) !important;
    color: var(--color-on-primary) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #152a45 !important;
}

/* ---- Alerts & expanders ---- */
[data-testid="stAlert"] {
    border-radius: var(--radius-md);
    border: 1px solid var(--color-border);
}

details[data-testid="stExpander"] {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-muted);
}

details[data-testid="stExpander"] summary {
    font-weight: 600;
    color: var(--color-muted-text);
    font-size: 0.875rem;
}

/* ---- Custom components ---- */
.dr-header {
    margin-bottom: 1.75rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--color-border);
}

.dr-header-top {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.35rem;
}

.dr-logo {
    width: 40px;
    height: 40px;
    border-radius: var(--radius-sm);
    background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: var(--shadow-sm);
}

.dr-logo svg {
    width: 22px;
    height: 22px;
    fill: white;
}

.dr-title {
    font-family: var(--font-heading);
    font-size: 2rem;
    font-weight: 700;
    color: var(--color-primary);
    margin: 0;
    line-height: 1.2;
    letter-spacing: -0.02em;
}

.dr-subtitle {
    color: var(--color-muted-text);
    font-size: 0.95rem;
    margin: 0;
    padding-left: 3.25rem;
    line-height: 1.5;
    font-weight: 500;
}

.dr-empty {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: 2rem 1.5rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    margin: 1rem 0 1.5rem;
}

.dr-empty-icon {
    width: 48px;
    height: 48px;
    margin: 0 auto 1rem;
    border-radius: 50%;
    background: var(--color-muted);
    display: flex;
    align-items: center;
    justify-content: center;
}

.dr-empty-icon svg {
    width: 24px;
    height: 24px;
    stroke: var(--color-secondary);
}

.dr-empty h3 {
    font-family: var(--font-heading);
    font-size: 1.25rem;
    color: var(--color-foreground);
    margin: 0 0 0.5rem;
    font-weight: 600;
}

.dr-empty p {
    color: var(--color-muted-text);
    margin: 0;
    font-size: 0.9rem;
    line-height: 1.6;
}

.dr-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin-top: 1.25rem;
}

.dr-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    background: var(--color-muted);
    color: var(--color-muted-text);
    border: 1px solid var(--color-border);
}

.dr-tag-accent {
    background: #EFF6FF;
    color: var(--color-secondary);
    border-color: #BFDBFE;
}

.dr-sidebar-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--color-muted-text);
    margin-bottom: 0.5rem;
}

.dr-divider {
    height: 1px;
    background: var(--color-border);
    margin: 1.25rem 0;
}

/* Hide Streamlit branding footer */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
"""

HEADER_HTML = """
<div class="dr-header">
  <div class="dr-header-top">
    <div class="dr-logo" aria-hidden="true">
      <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M9.5 3A6.5 6.5 0 0 1 16 9.5c0 1.61-.59 3.09-1.56 4.23l.27.27h.79l5 5-1.5 1.5-5-5v-.79l-.27-.27A6.516 6.516 0 0 1 9.5 16 6.5 6.5 0 0 1 3 9.5 6.5 6.5 0 0 1 9.5 3m0 2C7 5 5 7 5 9.5S7 14 9.5 14 14 12 14 9.5 12 5 9.5 5z"/>
      </svg>
    </div>
    <h1 class="dr-title">DeepResearch</h1>
  </div>
  <p class="dr-subtitle">学术文献超级智能体 · 自动路由：日常对话 / ReAct 工具调用 / 深度研究</p>
</div>
"""

EMPTY_STATE_HTML = """
<div class="dr-empty">
  <div class="dr-empty-icon" aria-hidden="true">
    <svg fill="none" viewBox="0 0 24 24" stroke-width="1.5" xmlns="http://www.w3.org/2000/svg">
      <path stroke-linecap="round" stroke-linejoin="round"
            d="M12 6.042A8.967 8.967 0 0 0 6 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 0 1 6 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 0 1 6-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0 0 18 18a8.967 8.967 0 0 0-6 2.292m0-14.25v14.25"/>
    </svg>
  </div>
  <h3>开始你的研究</h3>
  <p>在左侧上传 PDF / Word / TXT 文献后提问，<br>也可以直接进行日常对话。</p>
  <div class="dr-tags">
    <span class="dr-tag dr-tag-accent">文献检索</span>
    <span class="dr-tag">深度研究</span>
    <span class="dr-tag">日常对话</span>
  </div>
</div>
"""

st.set_page_config(
    page_title="DeepResearch",
    page_icon=":material/science:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)

# ---- Session（最先初始化，避免刷新闪烁）----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "ingest_notice" not in st.session_state:
    st.session_state.ingest_notice = ""


@st.cache_data(ttl=15, show_spinner=False)
def _kb_stats() -> tuple[int, int]:
    from src.rag.retriever import retriever
    return retriever.stats()


@st.cache_resource(show_spinner=False)
def _load():
    from src.orchestrator.graph import run_agent
    return run_agent


def _ingest_files(uploaded_files, progress_bar, status) -> list[dict]:
    from config.settings import settings
    from src.rag.ingest import ingest_file

    docs_dir = Path(settings.DOCS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    total = len(uploaded_files)
    for i, f in enumerate(uploaded_files):
        name = Path(f.name).name
        status.markdown(f"**({i + 1}/{total})** `{name}`")
        progress_bar.progress(i / total)
        dest = docs_dir / name
        dest.write_bytes(f.getvalue())
        results.append(ingest_file(dest))
        progress_bar.progress((i + 1) / total)
    progress_bar.progress(1.0)
    return results


def _submit(text: str) -> None:
    text = text.strip()
    if text:
        st.session_state.messages.append({"role": "user", "content": text})
        st.session_state.generating = True


def _render_trace(trace: list[dict]) -> None:
    with st.expander("执行详情", expanded=False):
        for step in trace:
            icon = "✓" if step["step"] != "error" else "✗"
            color = "var(--color-accent)" if step["step"] != "error" else "var(--color-destructive)"
            st.markdown(
                f'<div style="padding:0.35rem 0;border-bottom:1px solid var(--color-border);'
                f'font-size:0.875rem"><span style="color:{color};font-weight:700;margin-right:0.4rem">'
                f'{icon}</span><strong>{step["step"]}</strong> · {step["detail"]}</div>',
                unsafe_allow_html=True,
            )


# ---- 侧边栏 ----
with st.sidebar:
    st.markdown("### 知识库")

    try:
        chunks, docs = _kb_stats()
    except Exception:
        chunks, docs = 0, 0

    c1, c2 = st.columns(2)
    c1.metric("文档", docs)
    c2.metric("文本块", chunks)

    if st.session_state.ingest_notice:
        st.success(st.session_state.ingest_notice)

    st.markdown('<div class="dr-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="dr-sidebar-label">上传文献</p>', unsafe_allow_html=True)
    st.caption("支持 PDF · Word · TXT · Markdown")

    uploads = st.file_uploader(
        "上传文献",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )

    if st.button("确认入库", use_container_width=True, disabled=not uploads, type="primary"):
        progress = st.progress(0)
        status = st.empty()
        results = _ingest_files(uploads, progress, status)
        _kb_stats.clear()
        ok = [r for r in results if r["ok"]]
        fail = [r for r in results if not r["ok"]]
        if ok:
            added = sum(r["chunks"] for r in ok)
            st.session_state.ingest_notice = f"已入库 {len(ok)} 个文件，新增 {added} 块"
            st.toast(st.session_state.ingest_notice)
        else:
            st.session_state.ingest_notice = ""
        for r in fail:
            st.error(f"{r['filename']}：{r['error']}")
        st.session_state.uploader_key += 1
        progress.empty()
        status.empty()
        st.rerun()

    st.markdown('<div class="dr-divider"></div>', unsafe_allow_html=True)
    if st.button("清空对话", use_container_width=True, key="btn_clear"):
        st.session_state.messages = []
        st.session_state.ingest_notice = ""
        st.session_state.generating = False
        st.session_state.thread_id = None
        st.rerun()

# ---- 主区域 ----
st.markdown(HEADER_HTML, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace"):
            _render_trace(msg["trace"])

if not st.session_state.messages:
    st.markdown(EMPTY_STATE_HTML, unsafe_allow_html=True)

if prompt := st.chat_input("输入研究问题或日常对话…"):
    _submit(prompt)
    st.rerun()

if st.session_state.get("generating") and st.session_state.messages:
    if st.session_state.messages[-1]["role"] == "user":
        question = st.session_state.messages[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("正在分析…"):
                try:
                    result = _load()(question, thread_id=st.session_state.thread_id)
                    st.session_state.thread_id = result.get("thread_id")
                    answer = result["answer"]
                    trace = result.get("trace", [])
                except Exception as e:
                    answer = f"处理失败：{e}"
                    trace = [{"step": "error", "detail": str(e)}]
            st.markdown(answer)
            if trace:
                _render_trace(trace)
        st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})
        st.session_state.generating = False
