import html
import math
import os
import re
import tempfile
from pathlib import Path

import streamlit as st

from shopping_agent import (
    agent,
    bootstrap_data,
    checkout_product,
    fetch_products,
    get_product,
    visual_similarity_search,
)

bootstrap_data()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "store.db"
FALLBACK_IMAGE = BASE_DIR / "resources" / "elephant.png"

st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛍️", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

            :root {
                --bg: #0a0a0a;
                --surface: #111111;
                --surface-raised: #1a1a1a;
                --surface-hover: #222222;
                --ink: #f5f5f0;
                --ink-secondary: #999990;
                --ink-muted: #555550;
                --line: #2a2a2a;
                --line-strong: #3a3a3a;
                --accent: #f5d94e;
                --accent-deep: #e5c93e;
                --accent-dim: rgba(245, 217, 78, 0.08);
                --accent-border: rgba(245, 217, 78, 0.2);
                --white: #f5f5f0;
                --radius-xl: 20px;
                --radius-lg: 14px;
                --radius-md: 10px;
                --radius-sm: 6px;
                --shadow: 0 24px 48px rgba(0, 0, 0, 0.6);
                --shadow-soft: 0 4px 16px rgba(0, 0, 0, 0.4);
                --panel-width: 60%;
            }

            html, body, [class*="css"], .stApp, p, span, div, h1, h2, h3, h4, h5, h6, input, select, textarea, button {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            }

            .stApp {
                background: var(--bg) !important;
            }

            .block-container {
                padding: 1.5rem 2rem 3rem 2rem !important;
                max-width: 1360px !important;
            }

            [data-testid="stSidebar"] { display: none !important; }
            header[role="banner"] { background: transparent !important; }

            /* ── RESET stacked border wrappers ── */
            [data-testid="stVerticalBlockBorderWrapper"] {
                border: 1px solid var(--line) !important;
                border-radius: var(--radius-lg) !important;
                background: var(--surface) !important;
                box-shadow: none !important;
                padding: 1rem !important;
                transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
            }

            [data-testid="stColumn"] > div > [data-testid="stVerticalBlockBorderWrapper"] {
                border: none !important;
                background: transparent !important;
                padding: 0 !important;
            }

            [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"]:hover {
                border-color: var(--line-strong) !important;
                box-shadow: var(--shadow-soft) !important;
            }

            [data-testid="stForm"] {
                border: 1px solid var(--line) !important;
                border-radius: var(--radius-lg) !important;
                background: var(--surface) !important;
                padding: 1.25rem !important;
                box-shadow: none !important;
            }
            [data-testid="stForm"]:hover { transform: none !important; }

            /* ── HEADER BAR ── */
            .topbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 0 1.5rem 0;
                border-bottom: 1px solid var(--line);
                margin-bottom: 1.75rem;
            }

            .topbar-brand {
                display: flex;
                align-items: center;
                gap: 0.6rem;
            }

            .brand-mark {
                width: 32px;
                height: 32px;
                background: var(--accent);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 16px;
                font-weight: 900;
                color: #000;
                line-height: 1;
            }

            .brand-name {
                font-family: 'Space Grotesk', sans-serif !important;
                font-size: 1.15rem;
                font-weight: 700;
                color: var(--white);
                letter-spacing: -0.02em;
            }

            .topbar-meta {
                color: var(--ink-muted);
                font-size: 0.8rem;
                font-weight: 500;
            }

            /* ── METRICS ROW ── */
            .metric-strip {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 1rem;
                margin-bottom: 2rem;
            }

            .metric-tile {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: var(--radius-lg);
                padding: 1.1rem 1.25rem;
                transition: border-color 0.2s ease;
            }

            .metric-tile:hover { border-color: var(--line-strong); }

            .metric-tile-label {
                font-size: 0.7rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: var(--ink-muted);
                margin-bottom: 0.4rem;
            }

            .metric-tile-value {
                font-family: 'Space Grotesk', sans-serif !important;
                font-size: 1.8rem;
                font-weight: 700;
                color: var(--white);
                line-height: 1;
                margin-bottom: 0.3rem;
            }

            .metric-tile-note {
                font-size: 0.75rem;
                color: var(--ink-muted);
                line-height: 1.4;
            }

            /* ── SECTION HEADER ── */
            .section-head {
                margin-bottom: 1.25rem;
            }

            .section-eyebrow {
                font-size: 0.68rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: var(--accent);
                margin-bottom: 0.35rem;
            }

            .section-title {
                font-family: 'Space Grotesk', sans-serif !important;
                font-size: 1.4rem;
                font-weight: 700;
                color: var(--white);
                letter-spacing: -0.02em;
                line-height: 1.25;
            }

            /* ── CATALOG FILTER FORM ── */
            .stTextInput input {
                background: var(--surface-raised) !important;
                border: 1px solid var(--line) !important;
                border-radius: var(--radius-md) !important;
                color: var(--white) !important;
                font-size: 0.875rem !important;
                padding: 0.55rem 0.85rem !important;
            }

            .stTextInput input:focus {
                border-color: var(--accent) !important;
                box-shadow: 0 0 0 3px rgba(245, 217, 78, 0.12) !important;
                outline: none !important;
            }

            .stSelectbox [data-baseweb="select"] > div {
                background: var(--surface-raised) !important;
                border: 1px solid var(--line) !important;
                border-radius: var(--radius-md) !important;
                color: var(--white) !important;
            }

            .stSelectbox [data-baseweb="select"] > div:focus-within {
                border-color: var(--accent) !important;
                box-shadow: 0 0 0 3px rgba(245, 217, 78, 0.12) !important;
            }

            /* ── BUTTONS ── */
            .stButton > button,
            .stFormSubmitButton > button,
            [data-testid="stBaseButton-secondary"] {
                border-radius: var(--radius-md) !important;
                border: 1px solid var(--line-strong) !important;
                background: var(--surface-raised) !important;
                color: var(--white) !important;
                font-weight: 600 !important;
                font-size: 0.825rem !important;
                min-height: 2.4rem !important;
                transition: all 0.18s ease !important;
                letter-spacing: 0.01em !important;
            }

            .stButton > button:hover,
            [data-testid="stBaseButton-secondary"]:hover {
                border-color: var(--ink-secondary) !important;
                background: var(--surface-hover) !important;
                color: var(--white) !important;
            }

            .stButton > button[kind="primary"],
            .stFormSubmitButton > button,
            [data-testid="stBaseButton-primary"] {
                background: var(--accent) !important;
                border-color: var(--accent-deep) !important;
                color: #000000 !important;
                font-weight: 700 !important;
            }

            .stButton > button[kind="primary"]:hover,
            .stFormSubmitButton > button:hover,
            [data-testid="stBaseButton-primary"]:hover {
                background: var(--accent-deep) !important;
                border-color: #c5a92e !important;
                color: #000000 !important;
                box-shadow: 0 4px 20px rgba(245, 217, 78, 0.25) !important;
            }

            .stButton > button:active { transform: scale(0.97) !important; }

            /* ── TABS ── */
            .stTabs [data-baseweb="tab-list"] {
                gap: 0;
                background: var(--surface);
                padding: 0.3rem;
                border-radius: var(--radius-md);
                border: 1px solid var(--line);
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: var(--radius-sm) !important;
                color: var(--ink-secondary) !important;
                font-weight: 600 !important;
                font-size: 0.825rem !important;
                min-height: 34px !important;
                padding: 0 0.9rem !important;
                transition: all 0.2s ease !important;
            }

            .stTabs [aria-selected="true"] {
                background: var(--surface-hover) !important;
                color: var(--white) !important;
            }

            /* ── PRODUCT CARD ── */
            .product-card {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: var(--radius-lg);
                overflow: hidden;
                transition: border-color 0.2s ease, box-shadow 0.2s ease;
            }

            .product-card:hover {
                border-color: var(--line-strong);
                box-shadow: var(--shadow-soft);
            }

            .product-badge {
                display: inline-block;
                padding: 0.22rem 0.55rem;
                border-radius: 99px;
                font-size: 0.65rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 0.6rem;
            }

            .badge-organic {
                background: rgba(74, 222, 128, 0.1);
                color: #4ade80;
                border: 1px solid rgba(74, 222, 128, 0.2);
            }

            .badge-pick {
                background: var(--accent-dim);
                color: var(--accent);
                border: 1px solid var(--accent-border);
            }

            .product-name {
                font-family: 'Space Grotesk', sans-serif !important;
                font-size: 1rem;
                font-weight: 700;
                color: var(--white);
                line-height: 1.3;
                margin: 0 0 0.3rem 0;
                letter-spacing: -0.01em;
            }

            .product-meta {
                font-size: 0.75rem;
                color: var(--ink-muted);
                margin-bottom: 0.5rem;
                line-height: 1.4;
            }

            .product-price {
                font-family: 'Space Grotesk', sans-serif !important;
                font-size: 1.3rem;
                font-weight: 700;
                color: var(--white);
                margin-bottom: 0.4rem;
                letter-spacing: -0.02em;
            }

            .product-price span {
                font-size: 0.8rem;
                font-weight: 400;
                color: var(--ink-muted);
                margin-left: 0.3rem;
            }

            .product-desc {
                font-size: 0.8rem;
                color: var(--ink-secondary);
                line-height: 1.5;
                margin-bottom: 0.9rem;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }

            .product-source {
                font-size: 0.7rem;
                color: var(--ink-muted);
                margin-bottom: 0.35rem;
                font-style: italic;
            }

            /* product image */
            .catalog-image img,
            .visual-image img,
            .cart-thumb img {
                border-radius: var(--radius-md) !important;
                object-fit: cover !important;
                width: 100% !important;
            }

            .catalog-image img { height: 150px !important; }
            .visual-image img { height: 130px !important; }
            .cart-thumb img { height: 70px !important; width: 70px !important; }

            /* ── PAGINATION ── */
            .page-bar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 1rem;
            }

            .page-info {
                font-size: 0.8rem;
                color: var(--ink-muted);
            }

            /* ── FLOATING PANEL BUTTON ── */
            .panel-fab {
                position: fixed;
                bottom: 2rem;
                right: 2rem;
                z-index: 9998;
            }

            /* ── SLIDING PANEL ── */
            .panel-overlay {
                position: fixed;
                inset: 0;
                background: rgba(0, 0, 0, 0.7);
                backdrop-filter: blur(4px);
                z-index: 9999;
                display: flex;
                justify-content: flex-end;
            }

            .panel-drawer {
                width: var(--panel-width);
                height: 100%;
                background: var(--surface);
                border-left: 1px solid var(--line-strong);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                animation: slideIn 0.28s cubic-bezier(0.32, 0.72, 0, 1);
                box-shadow: -24px 0 60px rgba(0, 0, 0, 0.5);
            }

            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to   { transform: translateX(0);    opacity: 1; }
            }

            .panel-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 1.25rem 1.5rem;
                border-bottom: 1px solid var(--line);
                background: var(--surface);
                flex-shrink: 0;
            }

            .panel-title-block {}

            .panel-eyebrow {
                font-size: 0.65rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: var(--accent);
                margin-bottom: 0.2rem;
            }

            .panel-title {
                font-family: 'Space Grotesk', sans-serif !important;
                font-size: 1.1rem;
                font-weight: 700;
                color: var(--white);
                letter-spacing: -0.01em;
            }

            .panel-close {
                width: 34px;
                height: 34px;
                border-radius: var(--radius-sm);
                background: var(--surface-raised);
                border: 1px solid var(--line);
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                color: var(--ink-secondary);
                font-size: 1.1rem;
                font-weight: 300;
                transition: all 0.2s ease;
                line-height: 1;
            }

            .panel-close:hover {
                background: var(--surface-hover);
                color: var(--white);
                border-color: var(--line-strong);
            }

            .panel-body {
                flex: 1;
                overflow-y: auto;
                padding: 1.25rem 1.5rem;
            }

            .panel-body::-webkit-scrollbar { width: 5px; }
            .panel-body::-webkit-scrollbar-track { background: transparent; }
            .panel-body::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 999px; }

            /* ── CHAT MESSAGES ── */
            .msg {
                border-radius: var(--radius-md);
                padding: 0.85rem 1rem;
                margin-bottom: 0.75rem;
                font-size: 0.875rem;
                line-height: 1.55;
                border: 1px solid transparent;
            }

            .msg-user {
                background: var(--accent-dim);
                border-color: var(--accent-border);
                color: var(--white);
            }

            .msg-assistant {
                background: var(--surface-raised);
                border-color: var(--line);
                color: var(--white);
            }

            .msg-role {
                font-size: 0.65rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: var(--ink-muted);
                margin-bottom: 0.35rem;
            }

            .msg-user .msg-role { color: var(--accent); }

            .msg-body p {
                margin: 0 0 0.4rem 0;
                color: var(--white);
            }

            .msg-body p:last-child { margin-bottom: 0; }

            .msg-body ul {
                margin: 0.3rem 0 0 0;
                padding-left: 1.2rem;
            }

            .msg-body li { margin-bottom: 0.3rem; color: var(--ink-secondary); }
            .msg-body strong { color: var(--white); }

            .chat-empty {
                border: 1px dashed var(--line-strong);
                border-radius: var(--radius-md);
                padding: 2.5rem 1.5rem;
                text-align: center;
                color: var(--ink-muted);
                font-size: 0.85rem;
                line-height: 1.6;
            }

            .chat-empty-icon {
                font-size: 1.75rem;
                margin-bottom: 0.75rem;
                display: block;
            }

            /* ── QUICK ACTION CHIPS ── */
            .quick-chips {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-bottom: 1.25rem;
            }

            /* ── PANEL FAB TRIGGER ── */
            .fab-trigger {
                position: fixed;
                right: 1.75rem;
                bottom: 1.75rem;
                z-index: 9000;
            }

            /* ── TOOLBAR ── */
            .toolbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin-bottom: 0.75rem;
            }

            .toolbar-count {
                font-size: 0.8rem;
                color: var(--ink-muted);
                font-weight: 500;
            }

            .toolbar-count strong {
                color: var(--white);
            }

            /* ── MINI STAT ── */
            .mini-stat {
                background: var(--surface-raised);
                border: 1px solid var(--line);
                border-radius: var(--radius-md);
                padding: 0.75rem 1rem;
                font-size: 0.75rem;
                color: var(--ink-muted);
                line-height: 1.5;
            }

            /* ── CART ITEM ── */
            .cart-row {
                background: var(--surface-raised);
                border: 1px solid var(--line);
                border-radius: var(--radius-md);
                padding: 0.85rem;
                margin-bottom: 0.6rem;
                transition: border-color 0.2s ease;
            }

            .cart-row:hover { border-color: var(--line-strong); }

            .cart-name {
                font-size: 0.875rem;
                font-weight: 600;
                color: var(--white);
                margin-bottom: 0.25rem;
            }

            .cart-meta {
                font-size: 0.75rem;
                color: var(--ink-muted);
                margin-bottom: 0.6rem;
            }

            /* ── FILE UPLOADER ── */
            .stFileUploader section {
                border: 1px dashed var(--line-strong) !important;
                background: var(--surface-raised) !important;
                border-radius: var(--radius-md) !important;
                padding: 1.25rem !important;
                transition: border-color 0.2s ease !important;
            }

            .stFileUploader section:hover {
                border-color: var(--accent) !important;
                background: var(--accent-dim) !important;
            }

            /* ── TOGGLE ── */
            [data-testid="stToggle"] {
                gap: 0.5rem;
            }

            /* ── SLIDER ── */
            [data-testid="stSlider"] [class*="stSlider"] div[role="slider"] {
                background: var(--accent) !important;
                border-color: var(--accent) !important;
            }

            /* ── CAPTION TEXT ── */
            .stCaption {
                color: var(--ink-muted) !important;
                font-size: 0.75rem !important;
            }

            /* ── WARNING / INFO ── */
            [data-testid="stAlert"] {
                border-radius: var(--radius-md) !important;
                border: 1px solid var(--line-strong) !important;
                background: var(--surface-raised) !important;
            }

            /* ── DIVIDER ── */
            hr {
                border-color: var(--line) !important;
                margin: 1.25rem 0 !important;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "messages": [],
        "cart": {},
        "catalog_query": "",
        "catalog_category": "All",
        "catalog_sort": "Top rated",
        "catalog_max_price": 30.0,
        "catalog_organic_only": False,
        "catalog_page": 1,
        "catalog_page_size": 6,
        "last_uploaded_image": None,
        "last_visual_matches": [],
        "panel_open": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def db_connection():
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_catalog_summary() -> dict:
    conn = db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS total FROM products")
    total = cur.fetchone()["total"]
    cur.execute("SELECT COUNT(*) AS total FROM products WHERE is_organic = 1")
    organic = cur.fetchone()["total"]
    cur.execute("SELECT COUNT(*) AS total FROM products WHERE image_path IS NOT NULL AND image_path != ''")
    visual = cur.fetchone()["total"]
    cur.execute("SELECT ROUND(AVG(rating), 2) AS avg_rating FROM reviews")
    avg_rating = cur.fetchone()["avg_rating"] or 0
    conn.close()
    return {"total": total, "organic": organic, "visual": visual, "avg_rating": avg_rating}


def get_categories() -> list[str]:
    conn = db_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT category FROM products WHERE category IS NOT NULL ORDER BY category")
    categories = [row["category"] for row in cur.fetchall() if row["category"]]
    conn.close()
    return ["All"] + categories


def cart_count() -> int:
    return sum(st.session_state.cart.values())


def get_cart_items() -> list[dict]:
    items = []
    for product_id, qty in st.session_state.cart.items():
        product = get_product(product_id)
        if product:
            product = dict(product)
            product["qty"] = qty
            items.append(product)
    return sorted(items, key=lambda item: item["name"])


def cart_subtotal() -> float:
    return sum(float(item["price"]) * int(item["qty"]) for item in get_cart_items())


def format_rating(rating: float, review_count: int) -> str:
    stars = "★" * round(rating) + "☆" * (5 - round(rating))
    return f"{stars} {rating:.1f} ({review_count})"


def product_source_label(product: dict) -> str | None:
    return (
        product.get("photo_source_brand")
        or product.get("photo_source_name")
        or product.get("photo_source_dataset")
    )


def get_product_image(product: dict) -> str:
    image_path = product.get("image_path") or ""
    if image_path and os.path.exists(image_path):
        return image_path
    return str(FALLBACK_IMAGE)


def add_to_cart(product_id: int) -> None:
    st.session_state.cart[product_id] = st.session_state.cart.get(product_id, 0) + 1
    product = get_product(product_id)
    if product:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Added {product['name']} to your cart."}
        )


def remove_from_cart(product_id: int) -> None:
    if product_id not in st.session_state.cart:
        return
    if st.session_state.cart[product_id] <= 1:
        del st.session_state.cart[product_id]
    else:
        st.session_state.cart[product_id] -= 1


def clear_cart() -> None:
    st.session_state.cart = {}


def extract_product_ids(text: str) -> list[int]:
    ids, seen = [], set()
    for match in re.findall(r"\(ID:(\d+)\)", text or ""):
        value = int(match)
        if value not in seen:
            seen.add(value)
            ids.append(value)
    return ids


def display_message_text(message: dict) -> str:
    content = (message.get("content") or "").strip()
    if message.get("role") == "user" and "I uploaded a product image." in content:
        filename = content.split("Image path:")[-1].strip()
        return f"Visual search: {os.path.basename(filename)}"
    return re.sub(r"\s*\(ID:\d+\)", "", content)


def format_message_html(text: str) -> str:
    paragraphs: list[str] = []
    bullet_buffer: list[str] = []

    def flush_bullets() -> None:
        nonlocal bullet_buffer
        if bullet_buffer:
            items = "".join(f"<li>{item}</li>" for item in bullet_buffer)
            paragraphs.append(f"<ul>{items}</ul>")
            bullet_buffer = []

    normalized = (text or "").replace("Â·", "|")
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            flush_bullets()
            continue
        bullet_match = re.match(r"^(?:\d+\.\s+|-\s+)(.*)$", line)
        if bullet_match:
            bullet_buffer.append(html.escape(bullet_match.group(1).strip()))
            continue
        flush_bullets()
        if ":" in line and len(line) < 90:
            key, value = line.split(":", 1)
            paragraphs.append(f"<p><strong>{html.escape(key.strip())}:</strong>{html.escape(value)}</p>")
        else:
            paragraphs.append(f"<p>{html.escape(line)}</p>")

    flush_bullets()
    return "".join(paragraphs) if paragraphs else "<p>No details returned.</p>"


def run_assistant(prompt: str) -> None:
    if not prompt.strip():
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        result = agent.invoke({"messages": st.session_state.messages})
        response = result["messages"][-1].content.replace("`", "")
    st.session_state.messages.append({"role": "assistant", "content": response})


def run_image_search(image_path: str, uploaded_name: str) -> None:
    matches = visual_similarity_search(image_path=image_path, top_k=5)
    st.session_state.last_visual_matches = matches
    if matches:
        lines = [f"Visual matches for {uploaded_name}:"]
        for idx, product in enumerate(matches, start=1):
            lines.append(
                f"{idx}. {product['name']} (ID:{product['id']}) — "
                f"similarity {float(product.get('similarity_score', 0)):.3f} — "
                f"${float(product['price']):.2f} — {float(product['average_rating']):.2f}/5"
            )
        response = "\n".join(lines)
    else:
        response = f"No visual matches found for {uploaded_name}. Try a clearer product photo."
    st.session_state.messages.append({"role": "assistant", "content": response})


# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────

def render_page_header(summary: dict) -> None:
    cart_label = f"Cart ({cart_count()})" if cart_count() > 0 else "Cart"

    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-brand">
                <div class="brand-mark">S</div>
                <div class="brand-name">Storefront</div>
            </div>
            <div class="topbar-meta">{summary['total']} products · AI-powered search</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    tiles = [
        ("Products", str(summary["total"]), "Items in catalog"),
        ("Organic", str(summary["organic"]), "Certified organic picks"),
        ("With photos", str(summary["visual"]), "Real product images"),
        ("Avg rating", f"{summary['avg_rating']:.1f}★", "Across all reviews"),
    ]
    for col, (label, value, note) in zip([col1, col2, col3, col4], tiles):
        with col:
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="metric-tile-label">{label}</div>
                    <div class="metric-tile-value">{value}</div>
                    <div class="metric-tile-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# CATALOG CARD
# ─────────────────────────────────────────────

def render_catalog_card(product: dict, key_prefix: str) -> None:
    badge_label = "Organic" if product.get("is_organic") else "Top pick"
    badge_class = "badge-organic" if product.get("is_organic") else "badge-pick"
    source_label = product_source_label(product)

    with st.container(border=True):
        st.markdown('<div class="catalog-image">', unsafe_allow_html=True)
        st.image(get_product_image(product), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f'<span class="product-badge {badge_class}">{badge_label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="product-name">{product["name"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="product-meta">'
            f'{product["category"] or "General"} &nbsp;·&nbsp; '
            f'{format_rating(float(product["average_rating"]), int(product["review_count"]))}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="product-price">${float(product["price"]):.2f}'
            f'<span>/ unit</span></div>',
            unsafe_allow_html=True,
        )
        if source_label:
            st.markdown(
                f'<div class="product-source">Photo: {source_label}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="product-desc">{product["description"]}</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Add to cart", key=f"add_{key_prefix}_{product['id']}", use_container_width=True):
                add_to_cart(int(product["id"]))
                st.rerun()
        with c2:
            if st.button("Buy now", key=f"buy_{key_prefix}_{product['id']}", use_container_width=True, type="primary"):
                confirmation = checkout_product(int(product["id"]))
                st.session_state.messages.append({"role": "assistant", "content": confirmation})
                st.rerun()


# ─────────────────────────────────────────────
# CATALOG SECTION
# ─────────────────────────────────────────────

def render_catalog_section() -> None:
    st.markdown(
        """
        <div class="section-head">
            <div class="section-eyebrow">Catalog</div>
            <div class="section-title">Browse products</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    categories = get_categories()

    with st.container(border=True):
        with st.form("catalog_filters", clear_on_submit=False):
            r1c1, r1c2, r1c3 = st.columns([1.4, 0.9, 0.9])
            with r1c1:
                query = st.text_input("Search", value=st.session_state.catalog_query, placeholder="honey, oats, coffee…", label_visibility="collapsed")
            with r1c2:
                category = st.selectbox(
                    "Category", categories,
                    index=categories.index(st.session_state.catalog_category) if st.session_state.catalog_category in categories else 0,
                    label_visibility="collapsed",
                )
            with r1c3:
                sort_options = ["Top rated", "Price low to high", "Price high to low", "Most reviewed"]
                sort_by = st.selectbox(
                    "Sort", sort_options,
                    index=sort_options.index(st.session_state.catalog_sort) if st.session_state.catalog_sort in sort_options else 0,
                    label_visibility="collapsed",
                )

            r2c1, r2c2, r2c3, r2c4 = st.columns([1.2, 0.6, 0.6, 0.6])
            with r2c1:
                max_price = st.slider("Max price", min_value=5.0, max_value=35.0, value=float(st.session_state.catalog_max_price), step=1.0, format="$%.0f")
            with r2c2:
                organic_only = st.toggle("Organic only", value=st.session_state.catalog_organic_only)
            with r2c3:
                page_size = st.selectbox("Per page", options=[6, 9, 12], index=[6, 9, 12].index(int(st.session_state.catalog_page_size)) if int(st.session_state.catalog_page_size) in [6, 9, 12] else 0, label_visibility="collapsed")
            with r2c4:
                submitted = st.form_submit_button("Filter", use_container_width=True)

    if submitted:
        st.session_state.catalog_query = query.strip()
        st.session_state.catalog_category = category
        st.session_state.catalog_sort = sort_by
        st.session_state.catalog_max_price = max_price
        st.session_state.catalog_organic_only = organic_only
        st.session_state.catalog_page_size = page_size
        st.session_state.catalog_page = 1

    products = fetch_products(
        query=st.session_state.catalog_query,
        max_price=float(st.session_state.catalog_max_price),
        is_organic=True if st.session_state.catalog_organic_only else None,
        category=None if st.session_state.catalog_category == "All" else st.session_state.catalog_category,
        limit=60,
    )

    sort_key = st.session_state.catalog_sort
    if sort_key == "Price low to high":
        products = sorted(products, key=lambda p: (float(p["price"]), -float(p["average_rating"])))
    elif sort_key == "Price high to low":
        products = sorted(products, key=lambda p: (-float(p["price"]), -float(p["average_rating"])))
    elif sort_key == "Most reviewed":
        products = sorted(products, key=lambda p: (-int(p["review_count"]), -float(p["average_rating"])))
    else:
        products = sorted(products, key=lambda p: (-float(p["average_rating"]), -int(p["review_count"]), float(p["price"])))

    if not products:
        st.warning("No products match your filters.")
        return

    total = len(products)
    page_size = int(st.session_state.catalog_page_size)
    total_pages = max(1, math.ceil(total / page_size))
    st.session_state.catalog_page = min(max(1, st.session_state.catalog_page), total_pages)
    start = (st.session_state.catalog_page - 1) * page_size
    current_page = products[start : start + page_size]

    pc1, pc2, pc3 = st.columns([2, 0.8, 0.8])
    with pc1:
        st.markdown(
            f'<div class="toolbar-count"><strong>{total}</strong> products · page {st.session_state.catalog_page} of {total_pages}</div>',
            unsafe_allow_html=True,
        )
    with pc2:
        if st.button("← Prev", use_container_width=True, disabled=st.session_state.catalog_page <= 1):
            st.session_state.catalog_page -= 1
            st.rerun()
    with pc3:
        if st.button("Next →", use_container_width=True, disabled=st.session_state.catalog_page >= total_pages):
            st.session_state.catalog_page += 1
            st.rerun()

    grid = st.columns(2)
    for idx, product in enumerate(current_page):
        with grid[idx % 2]:
            render_catalog_card(product, key_prefix=str(start + idx))


# ─────────────────────────────────────────────
# PANEL: CHAT TAB
# ─────────────────────────────────────────────

def render_panel_chat() -> None:
    qc1, qc2 = st.columns(2)
    with qc1:
        if st.button("Best rated", use_container_width=True):
            run_assistant("Show me the best rated products in the store.")
            st.rerun()
        if st.button("Budget finds", use_container_width=True):
            run_assistant("Show me the best products under $10.")
            st.rerun()
    with qc2:
        if st.button("Organic under $20", use_container_width=True):
            run_assistant("Show me organic products under $20 with strong ratings.")
            st.rerun()
        if st.button("Compare honey", use_container_width=True):
            run_assistant("Compare the best honey products with ratings and prices.")
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="chat-empty">
                <span class="chat-empty-icon">🛍️</span>
                Ask for recommendations, compare products, or search by ingredient.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for idx, message in enumerate(st.session_state.messages):
            render_message_block(message, idx)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("panel_chat_form", clear_on_submit=True):
        prompt = st.text_area(
            "Message",
            placeholder="Ask for organic honey under $20 with 4+ rating…",
            height=88,
            label_visibility="collapsed",
        )
        s1, s2 = st.columns([3, 1])
        with s1:
            submitted = st.form_submit_button("Send", use_container_width=True)
        with s2:
            if st.form_submit_button("Clear", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    if submitted:
        run_assistant(prompt)
        st.rerun()


# ─────────────────────────────────────────────
# PANEL: VISUAL SEARCH TAB
# ─────────────────────────────────────────────

def render_panel_visual() -> None:
    st.markdown("**Upload a product image** to find similar items in the catalog.")
    uploaded = st.file_uploader("Product image", type=["jpg", "jpeg", "png", "webp"], key="panel_image")

    if uploaded:
        st.markdown('<div class="visual-image">', unsafe_allow_html=True)
        st.image(uploaded, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Find similar products", use_container_width=True, type="primary"):
            suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                image_path = tmp.name
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": f"I uploaded a product image. Please analyze it and find similar products. Image path: {image_path}",
                }
            )
            st.session_state.last_uploaded_image = image_path
            with st.spinner("Scanning catalog…"):
                run_image_search(image_path=image_path, uploaded_name=uploaded.name)
            st.rerun()

    matches = st.session_state.get("last_visual_matches", [])
    if matches:
        st.markdown("**Top matches**")
        mc1, mc2 = st.columns(2)
        for idx, product in enumerate(matches[:4]):
            with [mc1, mc2][idx % 2]:
                with st.container(border=True):
                    st.markdown('<div class="visual-image">', unsafe_allow_html=True)
                    st.image(get_product_image(product), use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown(f"**{product['name']}**")
                    st.caption(f"Sim {float(product.get('similarity_score', 0)):.3f} · ${float(product['price']):.2f}")
                    if st.button("Add to cart", key=f"vis_add_{product['id']}", use_container_width=True):
                        add_to_cart(int(product["id"]))
                        st.rerun()
    elif st.session_state.get("last_uploaded_image"):
        st.info("No similar products found. Try a clearer packaging photo.")


# ─────────────────────────────────────────────
# PANEL: CART TAB
# ─────────────────────────────────────────────

def render_panel_cart() -> None:
    items = get_cart_items()
    if not items:
        st.markdown(
            """
            <div class="chat-empty">
                <span class="chat-empty-icon">🛒</span>
                Your cart is empty. Add items from the catalog or ask the assistant.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    subtotal = cart_subtotal()
    st.markdown(
        f"""
        <div class="mini-stat">
            <strong style="color:var(--white)">{cart_count()} item{"s" if cart_count() != 1 else ""}</strong>
            &nbsp;·&nbsp; Subtotal: <strong style="color:var(--accent)">${subtotal:.2f}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    for item in items:
        cc1, cc2 = st.columns([0.28, 0.72])
        with cc1:
            st.markdown('<div class="cart-thumb">', unsafe_allow_html=True)
            st.image(get_product_image(item), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with cc2:
            st.markdown(f'<div class="cart-name">{item["name"]}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="cart-meta">${float(item["price"]):.2f} · qty {item["qty"]}</div>',
                unsafe_allow_html=True,
            )
            a1, a2 = st.columns(2)
            with a1:
                if st.button("Remove", key=f"cart_rm_{item['id']}", use_container_width=True):
                    remove_from_cart(int(item["id"]))
                    st.rerun()
            with a2:
                if st.button("Buy", key=f"cart_buy_{item['id']}", use_container_width=True, type="primary"):
                    confirmation = checkout_product(int(item["id"]))
                    remove_from_cart(int(item["id"]))
                    st.session_state.messages.append({"role": "assistant", "content": confirmation})
                    st.rerun()
        st.markdown('<hr style="margin:0.5rem 0;">', unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    with fc1:
        if st.button("Clear cart", use_container_width=True):
            clear_cart()
            st.rerun()
    with fc2:
        if st.button("Checkout all", use_container_width=True, type="primary"):
            receipts = []
            for item in items:
                for _ in range(int(item["qty"])):
                    receipts.append(checkout_product(int(item["id"])))
            clear_cart()
            st.session_state.messages.append(
                {"role": "assistant", "content": "Checkout complete!\n\n" + "\n".join(receipts)}
            )
            st.rerun()


# ─────────────────────────────────────────────
# MESSAGE RENDERING
# ─────────────────────────────────────────────

def render_message_block(message: dict, index: int) -> None:
    role = message.get("role", "assistant")
    display_text = display_message_text(message)
    wrapper_class = "msg-user" if role == "user" else "msg-assistant"
    label = "You" if role == "user" else "Shopping Agent"

    safe_text = (
        f"<p>{html.escape(display_text).replace(chr(10), '<br>')}</p>"
        if role == "user"
        else format_message_html(display_text)
    )

    st.markdown(
        f"""
        <div class="msg {wrapper_class}">
            <div class="msg-role">{label}</div>
            <div class="msg-body">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if role == "assistant":
        render_message_products(display_text, key_prefix=f"msg_{index}")


def render_message_products(text: str, key_prefix: str) -> None:
    product_ids = extract_product_ids(text)
    if not product_ids:
        return
    products = [dict(p) for pid in product_ids[:4] if (p := get_product(pid))]
    if not products:
        return

    st.markdown('<div style="font-size:0.75rem;font-weight:700;color:var(--ink-muted);text-transform:uppercase;letter-spacing:0.08em;margin:0.5rem 0 0.5rem 0;">Suggested</div>', unsafe_allow_html=True)
    cols = st.columns(min(2, len(products)))
    for idx, product in enumerate(products):
        with cols[idx % len(cols)]:
            with st.container(border=True):
                st.markdown('<div class="visual-image">', unsafe_allow_html=True)
                st.image(get_product_image(product), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown(f"**{product['name']}**")
                st.caption(f"${float(product['price']):.2f} · {float(product['average_rating']):.1f}★")
                if st.button("Add", key=f"{key_prefix}_add_{product['id']}", use_container_width=True):
                    add_to_cart(int(product["id"]))
                    st.rerun()


# ─────────────────────────────────────────────
# FLOATING PANEL BUTTON + PANEL
# ─────────────────────────────────────────────

def render_panel_toggle() -> None:
    """Renders the fixed floating button that opens/closes the panel."""
    cart_n = cart_count()
    cart_badge = f" ({cart_n})" if cart_n > 0 else ""
    msg_n = len([m for m in st.session_state.messages if m["role"] == "user"])
    label = f"🛍️ AI Assistant{cart_badge}"

    # The FAB is injected as a Streamlit button rendered in a fixed div via CSS trick:
    # We wrap it in custom HTML to position it.
    if not st.session_state.panel_open:
        fab_col = st.columns([1])[0]
        st.markdown(
            f"""
            <style>
            div[data-testid="stVerticalBlock"]:has( > div > div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div > div > button[title="fab_open"]) {{
                position: fixed;
                right: 1.75rem;
                bottom: 2rem;
                z-index: 9000;
                width: auto !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )


def render_assistant_panel() -> None:
    """Full-width overlay panel that slides in from the right."""
    chat_tab, vis_tab, cart_tab = st.tabs([
        f"💬 Ask AI",
        "🔍 Visual Search",
        f"🛒 Cart ({cart_count()})",
    ])
    with chat_tab:
        render_panel_chat()
    with vis_tab:
        render_panel_visual()
    with cart_tab:
        render_panel_cart()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

inject_css()
init_state()
summary = get_catalog_summary()

render_page_header(summary)

# ── Floating "Open panel" button ──────────────
cart_n = cart_count()
cart_suffix = f" · {cart_n} in cart" if cart_n > 0 else ""
btn_label = f"🛍️ AI Assistant{cart_suffix}"

open_col = st.columns([5, 1])[1]
with open_col:
    if st.button(btn_label, use_container_width=True, type="primary"):
        st.session_state.panel_open = not st.session_state.panel_open
        st.rerun()

# ── Catalog (full width when panel closed, partial when open) ─────────
if st.session_state.panel_open:
    main_col, panel_col = st.columns([0.38, 0.62], gap="large")
    with main_col:
        render_catalog_section()
    with panel_col:
        # Panel chrome
        st.markdown(
            """
            <div style="
                background: var(--surface);
                border: 1px solid var(--line-strong);
                border-radius: var(--radius-xl);
                overflow: hidden;
                box-shadow: var(--shadow);
                position: sticky;
                top: 1.5rem;
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 1rem 1.25rem;
                    border-bottom: 1px solid var(--line);
                    background: var(--surface);
                ">
                    <div>
                        <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:var(--accent);margin-bottom:0.2rem;">AI-Powered</div>
                        <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:700;color:var(--white);">Shopping Assistant</div>
                    </div>
                </div>
                <div style="padding: 1.25rem;">
            """,
            unsafe_allow_html=True,
        )
        render_assistant_panel()
        st.markdown("</div></div>", unsafe_allow_html=True)

        if st.button("✕ Close panel", use_container_width=True):
            st.session_state.panel_open = False
            st.rerun()
else:
    render_catalog_section()