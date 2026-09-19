import html
import time
import uuid
import datetime

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graph import graph_builder

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SauravGPT",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# LAYOUT CSS
#
# Architecture (see ui.md):
#   viewport lock      -> html/body/stApp/stMain/blockContainer = 100vh, overflow hidden
#   flex chain         -> every ancestor of a scroll region is display:flex + min-height:0
#   flex-none regions  -> .st-key-sb_top / sb_bottom / chat_dock / tel_top / tel_bottom
#   flex-1  regions    -> .st-key-sb_list / chat_scroll / tel_trace
#   bottom gutter      -> --dock-gap (5rem) applied identically to sidebar + main
#
# `min-height: 0` is the load-bearing rule: without it a flex child refuses to
# shrink below its content height and the overflow escapes to the page.
# =========================================================
st.markdown(
    """
<style>
:root {
    /* Bottom gutter shared by all three columns: profile, chat dock, memory box.
       Single source of truth — changing it keeps all three aligned. */
    --dock-gap: 2.5rem;
    --top-gap: 1.15rem;
    --side-gap: 1.5rem;

    --bg-app: #0b0e14;
    --bg-sidebar: #0c1017;
    --bg-card: #10151f;
    --bg-card-alt: #131823;
    --bg-input: #121721;
    --line: rgba(255, 255, 255, 0.07);
    --line-strong: rgba(255, 255, 255, 0.12);
    --txt: #e6edf3;
    --txt-dim: #7e8c9e;
    --txt-faint: #556274;
    --accent: #00e599;
}

/* ---- Strip Streamlit chrome ------------------------------------------- */
#MainMenu, header, footer { visibility: hidden; height: 0; }
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarHeader"],
[data-testid="stChatInputInstructions"],
button[kind="header"] { display: none !important; }

/* Streamlit gives stMarkdownContainer `margin-bottom: -1rem` to cancel out the
   bottom margin of the <p> that normal markdown produces. Raw-HTML blocks have
   no such <p>, so the negative margin is never cancelled and every custom card
   reports 16px less height than it draws — which is what makes stacked cards
   overlap and pushes fixed regions out of alignment. Cancel it for top-level
   markdown only (button/label markdown is not inside [data-testid=stMarkdown]). */
[data-testid="stMarkdown"] [data-testid="stMarkdownContainer"] {
    margin-bottom: 0 !important;
}

/* Boot script iframe: kept in the DOM (so it still runs) but given zero size. */
.st-key-boot_script {
    position: absolute !important;
    height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* =======================================================================
   1. GLOBAL VIEWPORT LOCK — the page itself must never scroll
   ======================================================================= */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    height: 100vh !important;
    max-height: 100vh !important;
    min-height: 100vh !important;
    overflow: hidden !important;
    margin: 0 !important;
    box-sizing: border-box !important;
}

body, [data-testid="stApp"] {
    background-color: var(--bg-app) !important;
    color: var(--txt) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif !important;
}

/* The block container becomes the flex root of the main area.
   Its padding-bottom is the 5rem dock gutter. */
[data-testid="stMainBlockContainer"] {
    height: 100vh !important;
    max-height: 100vh !important;
    min-height: 0 !important;
    max-width: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    padding: var(--top-gap) var(--side-gap) var(--dock-gap) var(--side-gap) !important;
    box-sizing: border-box !important;
}

/* =======================================================================
   2. FLEX CHAIN — root block -> columns row -> column -> column body
   ======================================================================= */

/* Streamlit wraps every child *block* in a <div data-testid="stLayoutWrapper">
   with `flex: 0 1 auto; min-height: auto`. That wrapper — not the container
   itself — is the real flex child, so sizing applied only to `.st-key-*`
   lands one level too deep and the region never grows or clamps.
   Every wrapper is therefore made transparent to the flex chain here, and
   sized through `:has()` in section 3. */
[data-testid="stLayoutWrapper"] {
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}

[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
    flex: 1 1 auto !important;
    height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    gap: 0 !important;
}

/* The wrapper holding the columns row takes all remaining vertical space. */
[data-testid="stMainBlockContainer"] [data-testid="stLayoutWrapper"]:has(> [data-testid="stHorizontalBlock"]) {
    flex: 1 1 auto !important;
    height: 100% !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] {
    flex: 1 1 auto !important;
    height: 100% !important;
    min-height: 0 !important;
    align-items: stretch !important;
    overflow: hidden !important;
}

[data-testid="stColumn"] {
    height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}

[data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    flex: 1 1 auto !important;
    height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    gap: 0 !important;
}

/* Kill the implicit top margin Streamlit adds to the first element of a column. */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"]
    > [data-testid="stElementContainer"]:first-of-type > * {
    margin-top: 0 !important;
}

/* =======================================================================
   3. LAYOUT UTILITIES — the only two behaviours any region is allowed
   ======================================================================= */

/* flex-none: locked headers / footers. Solid background so scrolling
   content disappears cleanly behind them instead of bleeding through.
   Applied to the layout wrapper *and* the container it holds. */
[data-testid="stLayoutWrapper"]:has(> .st-key-sb_top),
[data-testid="stLayoutWrapper"]:has(> .st-key-sb_bottom),
[data-testid="stLayoutWrapper"]:has(> .st-key-chat_dock),
[data-testid="stLayoutWrapper"]:has(> .st-key-tel_top),
[data-testid="stLayoutWrapper"]:has(> .st-key-tel_bottom),
.st-key-sb_top,
.st-key-sb_bottom,
.st-key-chat_dock,
.st-key-tel_top,
.st-key-tel_bottom {
    flex: 0 0 auto !important;
    flex-shrink: 0 !important;
    flex-grow: 0 !important;
    min-height: 0 !important;
    overflow: visible !important;
    position: relative !important;
    z-index: 20 !important;
    gap: 0 !important;
}

.st-key-sb_top, .st-key-sb_bottom { background-color: var(--bg-sidebar) !important; }
.st-key-chat_dock, .st-key-tel_top, .st-key-tel_bottom { background-color: var(--bg-app) !important; }

/* flex-1: the wrapper claims the leftover space, the container clamps to it. */
[data-testid="stLayoutWrapper"]:has(> .st-key-sb_list),
[data-testid="stLayoutWrapper"]:has(> .st-key-chat_scroll),
[data-testid="stLayoutWrapper"]:has(> .st-key-tel_trace) {
    flex: 1 1 auto !important;
    height: 100% !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

/* The only surfaces in the whole app permitted to scroll. */
.st-key-sb_list,
.st-key-chat_scroll {
    flex: 1 1 auto !important;
    height: 100% !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
}

/* Execution trace: the CONTAINER is fixed (flex-1, no scroll); only its
   inner .trace-scroll region scrolls. */
.st-key-tel_trace {
    flex: 1 1 auto !important;
    height: 100% !important;
    min-height: 0 !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
    margin: 0.7rem 0 !important;
}

/* Every markdown wrapper inside the trace container must inherit full height so
   the card can run its own header / scroll-body flex split. Streamlit nests an
   unnamed emotion div between stMarkdown and stMarkdownContainer, so the `> div`
   step below is required — without it the chain breaks and .trace-scroll grows
   to its content height instead of clamping. */
.st-key-tel_trace > [data-testid="stElementContainer"],
.st-key-tel_trace [data-testid="stMarkdown"],
.st-key-tel_trace [data-testid="stMarkdown"] > div,
.st-key-tel_trace [data-testid="stMarkdownContainer"] {
    flex: 1 1 auto !important;
    height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Scrollbar styling for the three permitted scroll surfaces. */
.st-key-sb_list::-webkit-scrollbar,
.st-key-chat_scroll::-webkit-scrollbar,
.trace-scroll::-webkit-scrollbar,
.memory-scroll::-webkit-scrollbar { width: 6px; }
.st-key-sb_list::-webkit-scrollbar-thumb,
.st-key-chat_scroll::-webkit-scrollbar-thumb,
.trace-scroll::-webkit-scrollbar-thumb,
.memory-scroll::-webkit-scrollbar-thumb {
    background-color: rgba(255, 255, 255, 0.12);
    border-radius: 3px;
}
.st-key-sb_list::-webkit-scrollbar-track,
.st-key-chat_scroll::-webkit-scrollbar-track,
.trace-scroll::-webkit-scrollbar-track,
.memory-scroll::-webkit-scrollbar-track { background: transparent; }

/* =======================================================================
   4. COLUMN 1 — SIDEBAR
   ======================================================================= */
[data-testid="stSidebar"] {
    min-width: 275px !important;
    max-width: 275px !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

[data-testid="stSidebarContent"] {
    height: 100vh !important;
    max-height: 100vh !important;
    min-height: 0 !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Same 5rem bottom gutter as the main area, so the profile card lines up
   with the chat dock and the memory box. */
[data-testid="stSidebarUserContent"] {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    padding: var(--top-gap) 0.8rem var(--dock-gap) 0.8rem !important;
    box-sizing: border-box !important;
}

/* Streamlit puts a plain, untagged <div> between stSidebarUserContent and the
   root vertical block; it has to join the flex chain too or the column's
   height stays content-sized and the profile card drifts off screen. */
[data-testid="stSidebarUserContent"] > div,
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {
    flex: 1 1 auto !important;
    height: 100% !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    gap: 0 !important;
}

.sidebar-brand-title {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.3px;
    padding-bottom: 0.9rem;
}
.sparkle-icon { color: var(--accent); font-size: 1.35rem; }

.st-key-sb_top .stButton > button {
    width: 100% !important;
    background-color: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #f0f4fc !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    padding: 0.5rem 0.75rem !important;
    border-radius: 8px !important;
    display: flex !important;
    justify-content: flex-start !important;
}
.st-key-sb_top .stButton > button:hover {
    background-color: rgba(255, 255, 255, 0.08) !important;
    border-color: var(--line-strong) !important;
}

[data-testid="stSidebar"] .stTextInput input {
    background-color: #131822 !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    color: #d1d9e5 !important;
    font-size: 0.82rem !important;
    padding: 0.4rem 0.65rem !important;
}
[data-testid="stSidebar"] .stTextInput { margin-top: 0.5rem !important; }

.sidebar-section-label {
    font-size: 0.76rem;
    font-weight: 600;
    color: #6a7686;
    margin: 0.85rem 0 0.15rem 0;
    padding-left: 0.3rem;
    letter-spacing: 0.3px;
}

/* Chat history rows — each is one element container, locked to its own
   height so the list scrolls instead of the rows stretching. */
.st-key-sb_list { padding-right: 0.3rem !important; padding-bottom: 0.5rem !important; }
.st-key-sb_list > [data-testid="stElementContainer"] {
    flex: 0 0 auto !important;
    min-height: 0 !important;
}
.st-key-sb_list .stButton > button {
    width: 100% !important;
    background-color: transparent !important;
    color: #c9d3e0 !important;
    border: 1px solid transparent !important;
    text-align: left !important;
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    padding: 0.45rem 0.65rem !important;
    font-size: 0.84rem !important;
    font-weight: 400 !important;
    border-radius: 6px !important;
    margin-bottom: 0.15rem !important;
    overflow: hidden !important;
    white-space: nowrap !important;
}
/* Keep the label left-aligned; primary buttons centre theirs by default. */
.st-key-sb_list .stButton > button > div,
.st-key-sb_list .stButton > button p {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.st-key-sb_list .stButton > button:hover {
    background-color: #171e2b !important;
    color: #ffffff !important;
}
/* Active thread = primary button */
.st-key-sb_list [data-testid="stBaseButton-primary"] {
    background-color: #18202d !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
.st-key-sb_list [data-testid="stBaseButton-primary"] p { color: #ffffff !important; }
.sidebar-empty-note {
    font-size: 0.78rem;
    color: var(--txt-faint);
    padding: 0.6rem 0.4rem;
}

/* Bottom profile card — locked 5rem above the viewport floor */
.sidebar-user-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background-color: #111621;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 0.55rem 0.75rem;
    height: 52px;
    box-sizing: border-box;
    margin-top: 0.6rem;
}
.user-info-wrap { display: flex; align-items: center; gap: 0.65rem; }
.user-avatar-circle {
    width: 32px; height: 32px; border-radius: 50%;
    background: linear-gradient(135deg, #2b394f, #1a2333);
    border: 1px solid rgba(255, 255, 255, 0.15);
    display: flex; align-items: center; justify-content: center;
    color: #ffffff; font-size: 0.82rem; font-weight: 600;
}
.user-name { font-size: 0.84rem; font-weight: 600; color: #ffffff; }
.user-settings-icon { color: #6c7889; font-size: 1.05rem; cursor: pointer; }
.user-settings-icon:hover { color: #ffffff; }

/* =======================================================================
   5. COLUMN 2 — MAIN CHAT
   ======================================================================= */
.st-key-chat_scroll { padding-right: 0.5rem !important; }

/* Empty state: centre the hero in the scroll area. The hero sits inside an
   stElementContainer, so `margin:auto` on the hero itself has no flex parent
   to centre against — the scroll container has to do it. */
.st-key-chat_scroll:has(.hero-container) { justify-content: center !important; }

.hero-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    margin: auto 0;              /* vertically centred inside the scroll area */
    padding: 1.5rem 1rem;
    width: 100%;
}
.hero-icon-badge {
    width: 48px; height: 48px; border-radius: 12px;
    background-color: #131b26;
    border: 1px solid rgba(0, 229, 153, 0.25);
    display: flex; align-items: center; justify-content: center;
    color: var(--accent); font-size: 1.4rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 0 20px rgba(0, 229, 153, 0.12);
}
.hero-title {
    font-size: 2.6rem; font-weight: 500; color: #ffffff;
    letter-spacing: -0.5px; margin-bottom: 0.9rem;
}
.hero-subtitle {
    font-size: 0.96rem; color: var(--txt-dim);
    max-width: 530px; line-height: 1.55;
}

.msg-user-box {
    background-color: #111620;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1.2rem;
}
.msg-meta-row {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 0.78rem; margin-bottom: 0.45rem;
}
.msg-user-avatar {
    width: 22px; height: 22px; border-radius: 50%;
    background-color: #273142; color: #cfd8e5;
    font-size: 0.65rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.msg-author-title { font-weight: 600; color: #ffffff; }
.msg-time { color: #637082; font-size: 0.72rem; }
.msg-user-content { font-size: 0.93rem; color: #e2e8f0; line-height: 1.5; }

.msg-agent-box { background-color: transparent; margin-bottom: 1.4rem; }
.msg-agent-header {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 0.82rem; font-weight: 600; color: #ffffff;
    margin-bottom: 0.5rem;
}
.verified-pill {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background-color: rgba(0, 229, 153, 0.1);
    border: 1px solid rgba(0, 229, 153, 0.25);
    color: var(--accent);
    font-size: 0.68rem; font-weight: 600;
    padding: 1px 7px; border-radius: 9999px;
}
.reasoning-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background-color: #141a24;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px; padding: 4px 9px;
    font-size: 0.74rem; color: #d97757;
    margin-bottom: 0.8rem;
}
.msg-agent-text {
    font-size: 0.93rem; color: #cbd5e1; line-height: 1.6;
}

/* Chat dock — flex-none, solid background, never moves */
.st-key-chat_dock { padding-top: 0.6rem !important; }

.st-key-chat_dock [data-testid="stChatInput"] {
    flex: 0 0 auto !important;
    width: 100% !important;
    background-color: var(--bg-input) !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 12px !important;
}
.st-key-chat_dock [data-testid="stChatInput"] > div,
.st-key-chat_dock [data-testid="stChatInputTextArea"] {
    background-color: transparent !important;
    color: var(--txt) !important;
}

/* =======================================================================
   6. COLUMN 3 — CONTEXT OBSERVABILITY
   ======================================================================= */
[data-testid="stColumn"]:last-of-type {
    border-left: 1px solid var(--line) !important;
    padding-left: 1.1rem !important;
}

.telemetry-header-bar {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.75rem; padding-bottom: 0.35rem;
}
.telemetry-title-group { display: flex; align-items: center; gap: 0.55rem; }
.telemetry-title {
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.8px;
    color: #e2e8f0; text-transform: uppercase;
}
.telemetry-badge-idle {
    background-color: #171d27;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px; padding: 1px 6px;
    font-size: 0.65rem; color: #798799; font-weight: 600;
}
.telemetry-badge-live {
    background-color: rgba(0, 229, 153, 0.1);
    border: 1px solid rgba(0, 229, 153, 0.3);
    border-radius: 4px; padding: 1px 6px;
    font-size: 0.65rem; color: var(--accent); font-weight: 700;
}
.telemetry-cluster-status {
    display: flex; align-items: center; gap: 0.4rem;
    font-size: 0.74rem; color: var(--txt-dim);
}
.pulse-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background-color: var(--accent);
    box-shadow: 0 0 6px rgba(0, 229, 153, 0.8);
}

.telemetry-metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.55rem;
}
.telemetry-metric-card {
    background-color: var(--bg-card);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.65rem 0.75rem;
    box-sizing: border-box;
    display: flex; flex-direction: column; justify-content: center;
    min-width: 0;
}
.metric-card-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.6px;
    color: #707d8e; text-transform: uppercase; margin-bottom: 0.35rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.metric-card-val {
    font-size: 1.3rem; font-weight: 600; color: #ffffff;
    line-height: 1.1; margin-bottom: 0.25rem;
}
.metric-card-val.active-green { color: var(--accent); }
.metric-card-sub {
    font-size: 0.68rem; color: var(--txt-faint);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* Execution trace card: fixed shell, scrolling body */
.trace-card-box {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    background-color: var(--bg-card);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.95rem 1rem;
    box-sizing: border-box;
    overflow: hidden;
}
.trace-top-bar {
    flex: 0 0 auto;
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.6px;
    color: #6d7a8c; text-transform: uppercase;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}
.trace-scroll {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    overflow-x: hidden;
    padding-right: 0.35rem;
}
.trace-empty-view {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; height: 100%; padding: 1rem;
}
.trace-empty-icon {
    width: 40px; height: 40px; border-radius: 9px;
    background-color: #141a24;
    border: 1px solid rgba(255, 255, 255, 0.08);
    display: flex; align-items: center; justify-content: center;
    color: var(--txt-faint); font-size: 1.25rem; margin-bottom: 0.75rem;
}
.trace-empty-title {
    font-size: 0.9rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.35rem;
}
.trace-empty-desc {
    font-size: 0.73rem; color: #6a7788;
    max-width: 320px; line-height: 1.45; margin-bottom: 0.9rem;
}
.trace-empty-footer {
    font-size: 0.66rem; color: #525e6f; display: flex; gap: 0.85rem;
}
.trace-item-row {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 0.78rem; line-height: 1.4;
}
.trace-item-left { display: flex; gap: 0.5rem; color: #cbd5e1; }
.trace-latency {
    color: #6a7686; font-size: 0.7rem;
    font-family: ui-monospace, monospace; margin-left: 0.5rem;
}

/* Bottom stack: working memory directly on top of the confidence score */
.memory-card-box {
    background-color: var(--bg-card);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
}
.memory-top-bar {
    flex: 0 0 auto;
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.6px;
    color: #6d7a8c; text-transform: uppercase; margin-bottom: 0.65rem;
}
/* The bottom block is flex-none, so the chunk list is capped and scrolls
   internally rather than pushing the execution trace off screen. */
.memory-scroll {
    max-height: 20vh;
    overflow-y: auto;
    overflow-x: hidden;
    padding-right: 0.3rem;
}
.memory-empty-row {
    display: flex; align-items: center; justify-content: space-between;
    background-color: var(--bg-input);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 6px; padding: 0.7rem 0.9rem;
}
.memory-empty-left { display: flex; align-items: center; gap: 0.7rem; }
.memory-empty-title { font-size: 0.84rem; font-weight: 600; color: #e2e8f0; }
.memory-empty-sub { font-size: 0.72rem; color: #637082; }

.chunk-card {
    background-color: var(--bg-card-alt);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 0.6rem 0.75rem;
    margin-bottom: 0.5rem;
}
.chunk-card:last-child { margin-bottom: 0; }
.chunk-header-row {
    display: flex; align-items: center; gap: 0.55rem;
    font-size: 0.78rem; font-weight: 600; color: #e2e8f0;
}
.sim-badge {
    background-color: rgba(0, 229, 153, 0.12);
    color: var(--accent);
    font-size: 0.68rem; font-weight: 700;
    padding: 1px 6px; border-radius: 4px;
    font-family: ui-monospace, monospace;
}
.chunk-code-box {
    background-color: #0d1118;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 4px; padding: 0.5rem 0.65rem;
    font-family: ui-monospace, monospace;
    font-size: 0.72rem; color: #a5b4c7;
    margin-top: 0.45rem; line-height: 1.4;
    overflow-x: auto;
}
.chunk-meta-footer {
    font-size: 0.66rem; color: var(--txt-faint);
    margin-top: 0.4rem; font-family: ui-monospace, monospace;
}

.confidence-card {
    display: flex; justify-content: space-between; align-items: center;
    background-color: var(--bg-card);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    box-sizing: border-box;
}
.confidence-title { font-size: 0.78rem; font-weight: 600; color: #cbd5e1; }
.confidence-sub { font-size: 0.68rem; color: #637082; }
.equalizer-bars { display: flex; align-items: flex-end; gap: 3px; height: 16px; }
.eq-bar { width: 3px; background-color: var(--accent); border-radius: 1px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Boot script: keeps the sidebar expanded and pins the chat
# scroll surface to its newest message after every rerun.
# Rendered inside a keyed container so CSS can collapse it to zero size
# without display:none (a hidden iframe must still execute).
# ---------------------------------------------------------
with st.container(key="boot_script"):
    st.iframe(
        """
<script>
(function () {
    try {
        const pWin = window.parent;
        const doc = pWin.document;

        for (let i = pWin.localStorage.length - 1; i >= 0; i--) {
            const key = pWin.localStorage.key(i);
            if (key && key.includes('stSidebarCollapsed')) {
                pWin.localStorage.removeItem(key);
            }
        }

        doc.body.style.overflow = 'hidden';
        doc.documentElement.style.overflow = 'hidden';

        const sidebar = doc.querySelector('[data-testid="stSidebar"]');
        if (sidebar && sidebar.getAttribute('aria-expanded') === 'false') {
            const btn = doc.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
                        doc.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
                        doc.querySelector('button[kind="header"]');
            if (btn) { btn.click(); }
        }

        [60, 220, 600].forEach(function (delay) {
            pWin.setTimeout(function () {
                const pane = doc.querySelector('.st-key-chat_scroll');
                if (pane) { pane.scrollTop = pane.scrollHeight; }
            }, delay);
        });
    } catch (e) {}
})();
</script>
""",
        height=1,
    )

# ---------------------------------------------------------
# State Management
# ---------------------------------------------------------
if "sessions" not in st.session_state:
    st.session_state.sessions = []

# Per-turn measurements, keyed by thread id. One entry is appended each time
# the agent actually runs, so a rendered message shows *its own* latency and
# context size rather than the current totals.
if "turn_stats" not in st.session_state:
    st.session_state.turn_stats = {}

if "active_session_id" not in st.session_state:
    initial_id = "chat-" + str(uuid.uuid4())[:8]
    st.session_state.active_session_id = initial_id
    st.session_state.sessions.append(
        {"id": initial_id, "title": "New chat", "created_at": datetime.datetime.now()}
    )


# ---------------------------------------------------------
# Backend Graph Compilation & Active State
# ---------------------------------------------------------
@st.cache_resource
def get_agent_app():
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)


agent_app = get_agent_app()

current_thread_id = st.session_state.active_session_id
config = {"configurable": {"thread_id": current_thread_id}}

current_state = agent_app.get_state(config).values
messages = current_state.get("messages", [])
has_messages = bool(messages)


@st.cache_data
def full_corpus_words() -> int:
    """Words the prompt would carry if the whole knowledge base were ingested.

    Measured from mock_data rather than hardcoded, so the baseline stays true
    if the corpus grows. Numerator and denominator are both word counts, which
    makes the ratio valid without pretending to count real tokens.
    """
    from mock_data import mock_database

    total = len(" ".join(mock_database.get("topics_index", [])).split())
    for topic in mock_database.get("content", {}).values():
        total += len(topic.get("summary", "").split())
        for body in topic.get("sections", {}).values():
            total += len(body.split())
    return total


def esc(text: str) -> str:
    """Escape user/model text before it goes into a raw-HTML block.

    Unescaped angle brackets were able to close the surrounding card div and
    break the column's flex chain, which is one source of the overlap.
    """
    return html.escape(str(text)).replace("\n", "<br>")


# =========================================================
# COLUMN 1 — SIDEBAR  (flex-none top / flex-1 list / flex-none profile)
# =========================================================
with st.sidebar:
    # --- Top: fixed ------------------------------------------------------
    with st.container(key="sb_top", gap=None):
        st.markdown(
            '<div class="sidebar-brand-title">'
            '<span class="sparkle-icon">✦</span><span>SauravGPT</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        if st.button("✏️  New chat", key="btn_new_chat"):
            new_id = "chat-" + str(uuid.uuid4())[:8]
            st.session_state.sessions.insert(
                0,
                {"id": new_id, "title": "New chat", "created_at": datetime.datetime.now()},
            )
            st.session_state.active_session_id = new_id
            st.rerun()

        search_query = st.text_input(
            "Search chats",
            placeholder="🔍  Search chats",
            label_visibility="collapsed",
        )

        st.markdown('<div class="sidebar-section-label">Recent</div>', unsafe_allow_html=True)

    # --- Middle: the only scrollable surface in this column --------------
    filtered_sessions = st.session_state.sessions
    if search_query:
        filtered_sessions = [
            s for s in filtered_sessions if search_query.lower() in s["title"].lower()
        ]

    with st.container(key="sb_list", gap=None):
        if not filtered_sessions:
            st.markdown('<div class="sidebar-empty-note">No chats yet</div>', unsafe_allow_html=True)
        else:
            for s in filtered_sessions:
                s_id = s["id"]
                is_active = s_id == st.session_state.active_session_id
                # Button `type` carries the active state instead of a wrapper
                # div — one less nested block in the scroll container.
                if st.button(
                    s["title"],
                    key=f"session_{s_id}",
                    type="primary" if is_active else "tertiary",
                    width="stretch",
                ):
                    st.session_state.active_session_id = s_id
                    st.rerun()

    # --- Bottom: fixed, 5rem above the viewport floor --------------------
    with st.container(key="sb_bottom", gap=None):
        st.markdown(
            """<div class="sidebar-user-card">
    <div class="user-info-wrap">
        <div class="user-avatar-circle">SS</div>
        <div class="user-text-col">
            <div class="user-name">Saurav Sarkar</div>
        </div>
    </div>
    <div class="user-settings-icon">⚙️</div>
</div>""",
            unsafe_allow_html=True,
        )

# =========================================================
# MAIN GRID — two locked full-height columns
# =========================================================
col_chat, col_telemetry = st.columns([5.8, 4.2], gap="medium", wrap=False)

# ---------------------------------------------------------
# COLUMN 2 — MAIN CHAT
# ---------------------------------------------------------
with col_chat:
    # --- Scrollable message history (flex-1) -----------------------------
    with st.container(key="chat_scroll", gap=None):
        if not has_messages:
            st.markdown(
                '<div class="hero-container">'
                '<div class="hero-icon-badge">✦</div>'
                "<div class=\"hero-title\">Hi Saurav, what's the plan?</div>"
                '<div class="hero-subtitle">Autonomous multi-agent orchestration, context '
                "retrieval, and verifiable chain-of-thought ready.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            turn_stats = st.session_state.turn_stats.get(current_thread_id, [])
            live_chunks = len(current_state.get("active_context", []))
            ai_turn = 0
            for msg in messages:
                if msg.type == "human":
                    time_str = datetime.datetime.now().strftime("%H:%M:%S UTC")
                    st.markdown(
                        f'<div class="msg-user-box">'
                        f'<div class="msg-meta-row">'
                        f'<div class="msg-user-avatar">US</div>'
                        f'<div class="msg-author-title">Principal Architect</div>'
                        f'<div class="msg-time">{time_str}</div>'
                        f"</div>"
                        f'<div class="msg-user-content">{esc(msg.content)}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                elif msg.type == "ai" and msg.content:
                    # Stats recorded when this specific turn ran; fall back to the
                    # live figures for history with no measurement on file.
                    stat = turn_stats[ai_turn] if ai_turn < len(turn_stats) else None
                    ai_turn += 1
                    n_chunks = stat["chunks"] if stat else live_chunks
                    timing = f" in {stat['ms']}ms" if stat else ""

                    if n_chunks:
                        # Same .verified-pill geometry, recoloured inline so the
                        # CSS block is untouched.
                        pill = '<span class="verified-pill">● Verified Grounding</span>'
                    else:
                        pill = (
                            '<span class="verified-pill" style="background-color:'
                            "rgba(148,163,184,0.10); border-color:rgba(148,163,184,0.28); "
                            'color:#94a3b8;">○ Unassisted</span>'
                        )

                    if n_chunks:
                        reasoning = (
                            f"🧠 Reasoned across {n_chunks} context "
                            f'chunk{"" if n_chunks == 1 else "s"}{timing} ▾'
                        )
                    else:
                        reasoning = f"🧠 Answered without loading context{timing} ▾"

                    st.markdown(
                        f'<div class="msg-agent-box">'
                        f'<div class="msg-agent-header">'
                        f'<span style="color:#00e599; font-size:1.1rem;">✦</span>'
                        f"<span>Autonomous Reasoning Agent</span>"
                        f"{pill}"
                        f"</div>"
                        f'<div class="reasoning-pill">{reasoning}</div>'
                        f'<div class="msg-agent-text">{esc(msg.content)}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # --- Fixed dock (flex-none), 5rem above the viewport floor -----------
    with st.container(key="chat_dock", gap=None):
        placeholder = (
            "Ask agent to simulate remediation rollout or isolate..."
            if has_messages
            else "Ask Agent..."
        )
        user_input = st.chat_input(placeholder, key="agent_chat_input")

# ---------------------------------------------------------
# COLUMN 3 — CONTEXT OBSERVABILITY
# ---------------------------------------------------------
with col_telemetry:
    # Everything this column renders comes from these four state values.
    has_idx = current_state.get("has_index", False)
    f_summaries = current_state.get("fetched_summaries", [])
    f_details = current_state.get("fetched_details", [])
    audit_log = current_state.get("audit_log", [])
    active_context = current_state.get("active_context", [])

    # --- Top: fixed telemetry header + metric cards ----------------------
    with st.container(key="tel_top", gap=None):
        if has_messages:
            badge = '<span class="telemetry-badge-live">REALTIME</span>'
            status = f"{len(audit_log)} events • {len(active_context)} chunks loaded"
        else:
            badge = '<span class="telemetry-badge-idle">IDLE</span>'
            status = "Awaiting first query"

        v_index_val = "Active" if has_idx else "Standby"
        v_index_sub = "Topic index loaded" if has_idx else "Not fetched yet"
        # Sub-lines list what was actually fetched; .metric-card-sub already
        # ellipsises overflow, so a long list cannot change the card height.
        v_summaries_sub = ", ".join(f_summaries) if f_summaries else "None fetched"
        v_context_sub = ", ".join(f_details) if f_details else "None fetched"

        st.markdown(
            f'<div class="telemetry-header-bar">'
            f'<div class="telemetry-title-group">'
            f'<span class="telemetry-title">Agent Telemetry</span>{badge}</div>'
            f'<div class="telemetry-cluster-status">'
            f'<span class="pulse-dot"></span> {status}</div>'
            f"</div>"
            f'<div class="telemetry-metrics-grid">'
            f'<div class="telemetry-metric-card">'
            f'<div class="metric-card-label">Vector Index</div>'
            f'<div class="metric-card-val active-green">{v_index_val}</div>'
            f'<div class="metric-card-sub">{esc(v_index_sub)}</div></div>'
            f'<div class="telemetry-metric-card">'
            f'<div class="metric-card-label">Topic Summaries</div>'
            f'<div class="metric-card-val">{len(f_summaries)}</div>'
            f'<div class="metric-card-sub">{esc(v_summaries_sub)}</div></div>'
            f'<div class="telemetry-metric-card">'
            f'<div class="metric-card-label">Deep Context</div>'
            f'<div class="metric-card-val">{len(f_details)}</div>'
            f'<div class="metric-card-sub">{esc(v_context_sub)}</div></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # --- Middle: fixed container, scrolling contents ---------------------
    with st.container(key="tel_trace", gap=None):
        if not audit_log:
            trace_head = '<span>Execution Trace</span><span>0 events</span>'
            trace_body = (
                '<div class="trace-empty-view">'
                '<div class="trace-empty-icon">◎</div>'
                '<div class="trace-empty-title">Awaiting user query</div>'
                '<div class="trace-empty-desc">No tool calls or reasoning steps recorded '
                "yet. Each step the agent takes will be logged here as it runs.</div>"
                "</div>"
            )
        else:
            trace_head = (
                f"<span>Live Execution Trace</span>"
                f'<span style="color:#798799;">Buffered ({len(audit_log)} events)</span>'
            )
            rows = []
            for log in reversed(audit_log):
                act = esc(log.get("action", ""))
                det = esc(log.get("details", ""))
                # Real figure from the audit entry; the graph records no timings,
                # so there is nothing truthful to put in a latency column.
                tokens = log.get("tokens", 0) or 0
                icon = "🛠" if "Tool" in act else ("🧠" if "Reasoned" in act else "✅")
                rows.append(
                    f'<div class="trace-item-row">'
                    f'<div class="trace-item-left"><span>{icon}</span>'
                    f"<div><strong>{act}:</strong> {det}</div></div>"
                    f'<div class="trace-latency">{tokens} tok</div></div>'
                )
            trace_body = "".join(rows)

        st.markdown(
            f'<div class="trace-card-box">'
            f'<div class="trace-top-bar">{trace_head}</div>'
            f'<div class="trace-scroll">{trace_body}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # --- Bottom: fixed memory + confidence stack, 5rem above the floor ---
    with st.container(key="tel_bottom", gap=None):
        if not active_context:
            mem_head = '<span>Working Memory</span><span style="color:#637082;">Buffer: Empty</span>'
            mem_body = (
                '<div class="memory-empty-row">'
                '<div class="memory-empty-left">'
                '<span style="font-size:1.1rem; color:#556274;">▤</span>'
                '<div><div class="memory-empty-title">0 chunks mounted</div>'
                '<div class="memory-empty-sub">Ready for live context retrieval and synthesis'
                "</div></div></div>"
                '<div class="telemetry-badge-idle">● Standby</div>'
                "</div>"
            )
        else:
            n_chunks = len(active_context)
            mem_head = (
                f"<span>Working Memory</span>"
                f'<span style="color:#798799;">{n_chunks} '
                f'chunk{"" if n_chunks == 1 else "s"}</span>'
            )
            cards = []
            for i, chunk in enumerate(active_context):
                cards.append(
                    f'<div class="chunk-card">'
                    f'<div class="chunk-header-row"><span>Chunk {i + 1}</span></div>'
                    f'<div class="chunk-code-box">{esc(chunk)}</div>'
                    f'<div class="chunk-meta-footer">{len(chunk.split())} words • '
                    f"{len(chunk)} chars</div></div>"
                )
            mem_body = "".join(cards)

        # Context Reduction Efficiency: how much of the corpus we did NOT load.
        corpus_words = full_corpus_words()
        loaded_words = sum(len(c.split()) for c in active_context)
        if loaded_words and corpus_words:
            savings = max(0.0, round((1 - (loaded_words / corpus_words)) * 100, 1))
            eff_sub = (
                f"{savings}% saved vs full ingestion • "
                f"{loaded_words}/{corpus_words} words"
            )
            filled = max(1, min(7, round(savings / 100 * 7)))
        else:
            # Nothing loaded means nothing answered, so there is no saving to claim.
            eff_sub = "No context loaded yet"
            filled = 0

        # 7-segment meter: lit bars track the ratio. Heights stay inside the
        # 16px .equalizer-bars box, so the card geometry is unchanged.
        eq_bars = "".join(
            f'<div class="eq-bar" style="height: {16 if i < filled else 4}px;'
            f' opacity: {1 if i < filled else 0.22};"></div>'
            for i in range(7)
        )

        st.markdown(
            f'<div class="memory-card-box">'
            f'<div class="memory-top-bar">{mem_head}</div>'
            f'<div class="memory-scroll">{mem_body}</div>'
            f"</div>"
            f'<div class="confidence-card">'
            f'<div><div class="confidence-title">Context Reduction Efficiency</div>'
            f'<div class="confidence-sub">{eff_sub}</div>'
            f"</div>"
            f'<div class="equalizer-bars">{eq_bars}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# Agent invocation (after the layout is painted)
# ---------------------------------------------------------
if user_input:
    for s in st.session_state.sessions:
        if s["id"] == current_thread_id and s["title"] in ("New chat", "New conversation"):
            s["title"] = user_input[:28] + ("..." if len(user_input) > 28 else "")
            break

    with st.spinner("Analyzing intent and synthesizing telemetry..."):
        try:
            started = time.perf_counter()
            agent_app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                # Backstop below the graph's own tool budget. Without a handler a
                # GraphRecursionError propagates and Streamlit replaces the locked
                # layout with a red traceback.
                config={**config, "recursion_limit": 40},
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Agent run failed: {type(exc).__name__}: {exc}")
            st.stop()

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        settled = agent_app.get_state(config).values
        st.session_state.turn_stats.setdefault(current_thread_id, []).append(
            {"ms": elapsed_ms, "chunks": len(settled.get("active_context", []))}
        )
    st.rerun()
