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

/* ---- Chat：左助手 / 右用户 ---- */
div[data-testid="stChatMessage"] {
    display: flex !important;
    align-items: flex-start;
    gap: 0.65rem;
    background: transparent !important;
    border: none !important;
    padding: 0.35rem 0 !important;
    width: 100% !important;
}

/* 助手：靠左 */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    flex-direction: row !important;
    justify-content: flex-start !important;
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:nth-child(2) {
    flex: 0 1 auto;
    max-width: min(92%, 720px);
}

/* 用户：靠右 */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
    justify-content: flex-start !important;
    margin-left: auto !important;
    max-width: min(88%, 680px);
    width: fit-content !important;
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:nth-child(2) {
    flex: 0 1 auto;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
}

/* 气泡基础样式 */
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    padding: 0.75rem 1rem;
    box-shadow: var(--shadow-sm);
    line-height: 1.65;
    width: 100%;
}

/* 助手气泡：左下小圆角 */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
    border-radius: 4px var(--radius-md) var(--radius-md) var(--radius-md);
}

/* 用户气泡：右下小圆角 + 主色 */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
    background: var(--color-primary);
    color: var(--color-on-primary);
    border-color: var(--color-primary);
    border-radius: var(--radius-md) var(--radius-md) 4px var(--radius-md);
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] p,
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] li,
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] span {
    color: var(--color-on-primary) !important;
}

/* 头像缩小 */
div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    min-width: 2rem;
    width: 2rem;
    height: 2rem;
    font-size: 1rem;
}

/* 执行详情（助手侧） */
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) details[data-testid="stExpander"] {
    margin-top: 0.5rem;
    max-width: 100%;
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

/* 输入框内附件按钮与预览区 */
[data-testid="stChatInput"] [data-testid="stChatInputFileUploadButton"] button {
    background: transparent !important;
    color: var(--color-muted-text) !important;
    border: none !important;
}

[data-testid="stChatInput"] [data-testid="stChatInputFileUploadButton"] button:hover {
    color: var(--color-secondary) !important;
    background: var(--color-muted) !important;
}

[data-testid="stChatInputFileUploadDropzone"] {
    border-color: var(--color-secondary) !important;
    background: #EFF6FF !important;
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
    position: relative;
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: 2.5rem 1.75rem 2rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    margin: 2rem 0 2.5rem;
    overflow: hidden;
}

.dr-empty-glow {
    position: absolute;
    top: -40%;
    left: 50%;
    transform: translateX(-50%);
    width: 280px;
    height: 160px;
    background: radial-gradient(ellipse, rgba(37, 99, 235, 0.08) 0%, transparent 70%);
    pointer-events: none;
}

.dr-empty-icon {
    position: relative;
    width: 56px;
    height: 56px;
    margin: 0 auto 1.25rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.12);
}

.dr-empty-icon svg {
    width: 28px;
    height: 28px;
    stroke: var(--color-secondary);
}

.dr-empty h3 {
    font-family: var(--font-heading);
    font-size: 1.35rem;
    color: var(--color-foreground);
    margin: 0 0 1.25rem;
    font-weight: 600;
    letter-spacing: -0.01em;
}

.dr-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin-top: 0;
}

.dr-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: var(--color-surface);
    color: var(--color-muted-text);
    border: 1px solid var(--color-border);
    transition: border-color var(--transition), box-shadow var(--transition);
}

.dr-tag:hover {
    border-color: #BFDBFE;
    box-shadow: 0 1px 4px rgba(37, 99, 235, 0.08);
}

.dr-tag-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--color-muted-text);
    opacity: 0.5;
    flex-shrink: 0;
}

.dr-tag-accent {
    background: #EFF6FF;
    color: var(--color-secondary);
    border-color: #BFDBFE;
}

.dr-tag-accent .dr-tag-dot {
    background: var(--color-secondary);
    opacity: 1;
}

.dr-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.5rem 0 0.75rem;
    align-items: center;
}

.dr-metric-primary {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid var(--color-border);
    background: var(--color-surface);
}

.dr-metric-primary.pass {
    background: #ECFDF5;
    border-color: #6EE7B7;
    color: #047857;
}

.dr-metric-primary.warn {
    background: #FFFBEB;
    border-color: #FCD34D;
    color: #B45309;
}

.dr-metric-chip {
    display: inline-flex;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    color: var(--color-muted-text);
    background: var(--color-muted);
    border: 1px solid var(--color-border);
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
  <p class="dr-subtitle">学术文献超级智能体 · 智能路由</p>
</div>
"""

EMPTY_STATE_HTML = """
<div class="dr-empty">
  <div class="dr-empty-glow" aria-hidden="true"></div>
  <div class="dr-empty-icon" aria-hidden="true">
    <svg fill="none" viewBox="0 0 24 24" stroke-width="1.5" xmlns="http://www.w3.org/2000/svg">
      <path stroke-linecap="round" stroke-linejoin="round"
            d="M12 6.042A8.967 8.967 0 0 0 6 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 0 1 6 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 0 1 6-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0 0 18 18a8.967 8.967 0 0 0-6 2.292m0-14.25v14.25"/>
    </svg>
  </div>
  <h3>开始你的研究</h3>
  <div class="dr-tags">
    <span class="dr-tag dr-tag-accent"><span class="dr-tag-dot"></span>文献检索</span>
    <span class="dr-tag"><span class="dr-tag-dot"></span>深度研究</span>
    <span class="dr-tag"><span class="dr-tag-dot"></span>日常对话</span>
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
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "ingest_notice" not in st.session_state:
    st.session_state.ingest_notice = ""
if "generating" not in st.session_state:
    st.session_state.generating = False
if "agent_started" not in st.session_state:
    st.session_state.agent_started = False


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


def _submit(text: str, attachments: list[dict] | None = None) -> None:
    text = text.strip()
    att = attachments or []
    if text or att:
        st.session_state.messages.append({
            "role": "user",
            "content": text or "（请根据附件回答）",
            "attachments": att,
        })
        st.session_state.generating = True


def _render_user_attachments(attachments: list[dict] | None) -> None:
    if not attachments:
        return
    cols = st.columns(min(len(attachments), 4))
    for i, att in enumerate(attachments):
        with cols[i % len(cols)]:
            if att.get("kind") == "image":
                import base64
                st.image(base64.b64decode(att["b64"]), caption=att.get("name", "图片"), use_container_width=True)
            elif att.get("kind") == "document":
                st.caption(f"文档 {att.get('name')}（{att.get('chars', 0)} 字）")


def _render_metrics(metrics: dict | None) -> None:
    if not metrics:
        return
    name = metrics.get("primary_name", "指标")
    display = metrics.get("primary_display", "—")
    hint = metrics.get("primary_hint", "")
    passed = metrics.get("primary_pass", True)
    status_cls = "pass" if passed else "warn"

    chips = []
    if metrics.get("tool_calls"):
        chips.append(f"工具 {metrics['tool_calls']} 次")
    if metrics.get("react_iterations"):
        chips.append(f"ReAct {metrics['react_iterations']} 轮")
    rag = metrics.get("rag_eval") or {}
    if rag.get("mode") == "dual":
        l1, l2 = rag.get("layer1", {}), rag.get("layer2", {})
        chips.append(
            f"检索层 {'✓' if l1.get('pass') else '✗'} {l1.get('avg_score', 0):.2f}"
            if l1.get("avg_score") is not None else "检索层 —"
        )
        chips.append(
            f"生成层 {'✓' if l2.get('pass') else '✗'} {l2.get('support_rate', 0):.0%}"
            if l2.get("support_rate") is not None else "生成层 —"
        )
    elif metrics.get("retrieval_hits"):
        chips.append(f"检索 {metrics['retrieval_hits']} 条")

    chip_html = "".join(f'<span class="dr-metric-chip">{c}</span>' for c in chips)
    st.markdown(
        f'<div class="dr-metrics">'
        f'<span class="dr-metric-primary {status_cls}">{name} {display}</span>'
        f'{chip_html}'
        f'<span class="dr-metric-chip">{hint}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


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
        st.session_state.agent_started = False
        st.rerun()

# ---- 主区域 ----
st.markdown(HEADER_HTML, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg.get("attachments"):
            _render_user_attachments(msg["attachments"])
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("metrics"):
                _render_metrics(msg["metrics"])
            if msg.get("trace"):
                _render_trace(msg["trace"])

if not st.session_state.messages:
    st.markdown(EMPTY_STATE_HTML, unsafe_allow_html=True)

CHAT_FILE_TYPES = ["png", "jpg", "jpeg", "webp", "gif", "bmp", "pdf", "docx", "txt", "md", "csv"]
prompt = st.chat_input(
    "输入研究问题或日常对话…",
    accept_file="multiple",
    file_type=CHAT_FILE_TYPES,
)
if prompt:
    text = (prompt.text or "").strip()
    attachments: list[dict] = []
    if prompt.files:
        from config.settings import settings
        from src.multimodal.attachments import parse_upload

        for f in prompt.files:
            try:
                attachments.append(
                    parse_upload(f.name, f.getvalue(), max_doc_chars=settings.CHAT_ATTACHMENT_MAX_DOC_CHARS)
                )
            except ValueError as e:
                st.warning(str(e))
    if text or attachments:
        _submit(text, attachments)
        st.rerun()

# 先快速刷新一帧，让用户消息立刻显示（避免与 Agent 阻塞同跑导致界面空白）
if st.session_state.get("generating") and not st.session_state.get("agent_started"):
    st.session_state.agent_started = True
    with st.chat_message("assistant"):
        st.markdown("_正在分析…_")
    st.rerun()

if st.session_state.get("generating") and st.session_state.get("agent_started"):
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        question = st.session_state.messages[-1]["content"]
        attachments = st.session_state.messages[-1].get("attachments") or []
        with st.chat_message("assistant"):
            with st.spinner("正在分析…"):
                try:
                    result = _load()(question, attachments=attachments)
                    answer = result["answer"]
                    trace = result.get("trace", [])
                    metrics = result.get("metrics")
                except Exception as e:
                    answer = f"处理失败：{e}"
                    trace = [{"step": "error", "detail": str(e)}]
                    metrics = None
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "trace": trace,
            "metrics": metrics,
        })
        st.session_state.generating = False
        st.session_state.agent_started = False
        st.rerun()
