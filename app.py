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
            :root {
                --bg: #f4f1ea;
                --surface: #fbf8f2;
                --surface-strong: #ffffff;
                --ink: #1f2937;
                --muted: #6b7280;
                --line: #e7dfd1;
                --accent: #d97706;
                --accent-deep: #9a3412;
                --forest: #365314;
                --shadow: 0 18px 45px rgba(31, 41, 55, 0.08);
                --shadow-soft: 0 8px 24px rgba(31, 41, 55, 0.06);
                --radius-xl: 28px;
                --radius-lg: 22px;
                --radius-md: 16px;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(217, 119, 6, 0.13), transparent 24%),
                    radial-gradient(circle at top right, rgba(54, 83, 20, 0.08), transparent 22%),
                    linear-gradient(180deg, #f8f4ed 0%, #f4f1ea 100%);
            }

            .block-container {
                padding-top: 2.2rem;
                padding-bottom: 2rem;
                max-width: 1500px;
            }

            [data-testid="stSidebar"] {
                display: none !important;
            }

            header[role="banner"] {
                background: transparent;
            }

            [data-testid="stVerticalBlockBorderWrapper"] {
                border-radius: var(--radius-lg);
            }

            .hero-shell {
                background: linear-gradient(145deg, #1f2937 0%, #111827 62%, #7c2d12 120%);
                border-radius: var(--radius-xl);
                padding: 1.65rem 1.8rem;
                border: 1px solid rgba(255, 255, 255, 0.08);
                color: #fffdf9;
                box-shadow: 0 24px 55px rgba(17, 24, 39, 0.2);
            }

            .eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.18em;
                font-size: 0.76rem;
                color: rgba(255, 251, 235, 0.72);
                margin-bottom: 0.6rem;
            }

            .hero-title {
                font-size: 2.6rem;
                line-height: 0.98;
                font-weight: 800;
                margin-bottom: 0.55rem;
            }

            .hero-copy {
                font-size: 1rem;
                color: rgba(255, 250, 240, 0.88);
                max-width: 48rem;
                line-height: 1.55;
            }

            .pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 1rem;
            }

            .pill {
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 999px;
                padding: 0.42rem 0.72rem;
                background: rgba(255, 255, 255, 0.08);
                color: #fff8ed;
                font-size: 0.82rem;
            }

            .metric-card {
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid var(--line);
                border-radius: var(--radius-md);
                padding: 0.95rem 1rem;
                box-shadow: var(--shadow-soft);
                min-height: 110px;
            }

            .metric-label {
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: var(--muted);
                font-size: 0.74rem;
                margin-bottom: 0.35rem;
            }

            .metric-value {
                color: var(--ink);
                font-size: 1.55rem;
                font-weight: 800;
                line-height: 1;
                margin-bottom: 0.25rem;
            }

            .metric-note {
                color: #4b5563;
                font-size: 0.82rem;
                line-height: 1.4;
            }

            .section-intro {
                margin: 0.35rem 0 0.8rem 0;
            }

            .section-title {
                color: var(--ink);
                font-size: 1.35rem;
                font-weight: 800;
                line-height: 1.1;
                margin-bottom: 0.25rem;
            }

            .section-copy {
                color: var(--muted);
                font-size: 0.94rem;
                line-height: 1.45;
                max-width: 44rem;
            }

            .toolbar-note {
                color: var(--muted);
                font-size: 0.84rem;
                margin-top: 0.35rem;
            }

            .assistant-frame {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid var(--line);
                border-radius: var(--radius-xl);
                padding: 1rem;
                box-shadow: var(--shadow);
            }

            .side-rail {
                position: sticky;
                top: 1.1rem;
            }

            .assistant-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 1rem;
                margin-bottom: 0.6rem;
            }

            .assistant-title {
                color: var(--ink);
                font-size: 1.2rem;
                font-weight: 800;
                margin-bottom: 0.15rem;
            }

            .assistant-copy {
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.45;
            }

            .assistant-badge {
                display: inline-block;
                padding: 0.35rem 0.6rem;
                border-radius: 999px;
                background: #fff7ed;
                border: 1px solid #fed7aa;
                color: var(--accent-deep);
                font-size: 0.78rem;
                font-weight: 700;
                white-space: nowrap;
            }

            .msg {
                border-radius: 14px;
                padding: 0.6rem 0.75rem;
                margin-bottom: 0.7rem;
                border: 1px solid #f0e8dc;
                font-size: 0.95rem;
            }

            .msg-user {
                background: #e0f2fe; /* light blue for user messages */
                border-color: #bae6fd;
            }

            .msg-assistant {
                background: #f8fafc; /* subtle contrast for assistant */
                border-color: #e6eef6;
            }

            .msg-label {
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-size: 0.72rem;
                font-weight: 700;
                color: #1f2937;
                margin-bottom: 0.35rem;
            }

            .msg-title {
                color: var(--ink);
                font-size: 0.95rem;
                font-weight: 700;
                margin-bottom: 0.3rem;
            }

            .msg-body {
                color: #000000;
                font-size: 0.93rem;
                line-height: 1.55;
            }

            .msg-body p {
                margin: 0 0 0.5rem 0;
            }

            .msg-body p:last-child {
                margin-bottom: 0;
            }

            .msg-body ul {
                margin: 0.15rem 0 0 0;
                padding-left: 1.1rem;
            }

            .msg-body li {
                margin-bottom: 0.3rem;
            }

            .msg-body strong {
                color: #000000;
            }

            .chat-empty {
                border: 1px dashed #d8cdbb;
                border-radius: 18px;
                padding: 1rem;
                background: #fffdf9;
                color: var(--muted);
                font-size: 0.92rem;
            }

            .support-chip {
                display: inline-block;
                border-radius: 999px;
                padding: 0.24rem 0.55rem;
                background: #f8f4ee;
                border: 1px solid var(--line);
                color: #7c2d12;
                font-size: 0.76rem;
                margin-right: 0.35rem;
            }

            .catalog-card {
                background: rgba(255, 255, 255, 0.97);
                border: 1px solid var(--line);
                border-radius: var(--radius-lg);
                padding: 0.8rem;
                box-shadow: var(--shadow-soft);
                min-height: 100%;
            }

            .product-kicker {
                display: inline-block;
                border-radius: 999px;
                padding: 0.22rem 0.52rem;
                background: #f7f6f1;
                border: 1px solid #ece4d7;
                color: #5b4636;
                font-size: 0.74rem;
                font-weight: 700;
                margin-top: 0.25rem;
            }

            .product-name {
                color: var(--ink);
                font-size: 1.02rem;
                font-weight: 800;
                line-height: 1.25;
                margin: 0.55rem 0 0.15rem 0;
            }

            .product-meta {
                color: #374151; /* darker for better contrast */
                font-size: 0.82rem;
                line-height: 1.35;
            }

            .product-price {
                color: var(--accent-deep);
                font-size: 1.05rem;
                font-weight: 800;
                margin-top: 0.35rem;
            }

            .product-source {
                color: #6b7280;
                font-size: 0.78rem;
                margin-top: 0.15rem;
            }

            .product-description {
                color: #374151;
                font-size: 0.86rem;
                line-height: 1.35;
                margin-top: 0.35rem;
                min-height: 40px;
                display: -webkit-box;
                -webkit-line-clamp: 3;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }

            .catalog-image img,
            .visual-image img {
                border-radius: 12px;
                object-fit: cover;
                width: 100%;
                height: 120px; /* smaller product and visual thumbnails */
            }

            .cart-thumb img {
                border-radius: 10px;
                object-fit: cover;
                height: 72px; /* compact cart thumbnail */
                width: auto;
            }

            .quick-card {
                background: #fffdf9;
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.7rem;
            }

            .chat-shell {
                background: #ffffff;
                border: 1px solid #e7dfd1;
                border-radius: 14px;
                padding: 0.6rem;
                max-height: 420px;
                overflow-y: auto;
            }

            .mini-stat {
                border-radius: 16px;
                border: 1px solid #efe3d2;
                background: #fffaf2;
                padding: 0.7rem 0.8rem;
            }

            .mini-stat-label {
                color: var(--muted);
                font-size: 0.73rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.2rem;
            }

            .mini-stat-value {
                color: var(--ink);
                font-size: 1rem;
                font-weight: 800;
            }

            .mini-note {
                color: var(--muted);
                font-size: 0.8rem;
            }

            .cart-line {
                border-bottom: 1px solid #efe7db;
                padding-bottom: 0.8rem;
                margin-bottom: 0.8rem;
            }

            .cart-line:last-child {
                border-bottom: none;
                margin-bottom: 0;
                padding-bottom: 0;
            }

            .panel-gap {
                height: 0.5rem;
            }

            .stButton > button,
            .stFormSubmitButton > button {
                border-radius: 14px;
                border: 1px solid #d6c7b3;
                background: linear-gradient(180deg, #fffefb 0%, #f6efe4 100%);
                color: #1f2937;
                font-weight: 700;
                min-height: 2.8rem;
                box-shadow: none;
            }

            .stButton > button:hover,
            .stFormSubmitButton > button:hover {
                border-color: #d97706;
                color: #9a3412;
            }

            .stButton > button[kind="primary"],
            .stFormSubmitButton > button[kind="primary"] {
                background: linear-gradient(180deg, #f59e0b 0%, #d97706 100%);
                border-color: #d97706;
                color: white;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.35rem;
                background: #f7f1e6;
                padding: 0.35rem;
                border-radius: 16px;
                border: 1px solid #eadfce;
            }

            .stTabs [data-baseweb="tab"] {
                border-radius: 12px;
                color: #6b7280;
                font-weight: 700;
                min-height: 42px;
                padding: 0 0.9rem;
            }

            .stTabs [aria-selected="true"] {
                background: #ffffff;
                color: #1f2937;
                box-shadow: 0 4px 10px rgba(31, 41, 55, 0.07);
            }

            .stTextInput input,
            .stTextArea textarea,
            .stSelectbox [data-baseweb="select"] > div,
            .stSlider,
            .stFileUploader section {
                border-radius: 14px !important;
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
    return {
        "total": total,
        "organic": organic,
        "visual": visual,
        "avg_rating": avg_rating,
    }


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


def render_metric(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_rating(rating: float, review_count: int) -> str:
    return f"{rating:.2f}/5 rating from {review_count} reviews"


def product_source_label(product: dict) -> str | None:
    return (
        product.get("photo_source_brand")
        or product.get("photo_source_name")
        or product.get("photo_source_dataset")
    )


def get_product_image(product: dict) -> str:
    """Return product image path if it exists, otherwise return fallback elephant image."""
    image_path = product.get("image_path") or ""
    if image_path and os.path.exists(image_path):
        return image_path
    return str(FALLBACK_IMAGE)


def add_to_cart(product_id: int) -> None:
    st.session_state.cart[product_id] = st.session_state.cart.get(product_id, 0) + 1
    product = get_product(product_id)
    if product:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Added {product['name']} to your cart.",
            }
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
    ids = []
    seen = set()
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
        return f"Visual search request for {os.path.basename(filename)}"
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
            item = html.escape(bullet_match.group(1).strip())
            bullet_buffer.append(item)
            continue

        flush_bullets()
        if ":" in line and len(line) < 90:
            key, value = line.split(":", 1)
            paragraphs.append(f"<p><strong>{html.escape(key.strip())}:</strong>{html.escape(value)}</p>")
        else:
            paragraphs.append(f"<p>{html.escape(line)}</p>")

    flush_bullets()
    return "".join(paragraphs) if paragraphs else "<p>No details returned.</p>"


def render_page_header(summary: dict) -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="eyebrow">Storefront redesign</div>
            <div class="hero-title">A cleaner AI shopping experience</div>
            <div class="hero-copy">
                Browse the catalog like a real storefront, keep filters compact, and use the assistant as a focused side panel for retrieval, visual search, and cart actions.
            </div>
            <div class="pill-row">
                <span class="pill">Curated product photos</span>
                <span class="pill">Visual similarity search</span>
                <span class="pill">Assistant-guided cart building</span>
                <span class="pill">Review-aware ranking</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric("Products", str(summary["total"]), "Every item is available from the main catalog.")
    with metric_cols[1]:
        render_metric("Organic picks", str(summary["organic"]), "Healthy options stay easy to spot.")
    with metric_cols[2]:
        render_metric("Real photos", str(summary["visual"]), "Product cards now use curated packaging images.")
    with metric_cols[3]:
        render_metric("Average rating", f"{summary['avg_rating']:.2f}", "Search and assistant recommendations stay rating aware.")


def render_catalog_card(product: dict, key_prefix: str) -> None:
    badge = "Organic" if product.get("is_organic") else "Top pick"
    source_label = product_source_label(product)

    with st.container(border=True):
        st.markdown('<div class="catalog-image">', unsafe_allow_html=True)
        product_image = get_product_image(product)
        st.image(product_image, width='content')
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="product-kicker">{badge}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="product-name">{product["name"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="product-meta">{product["category"] or "General"} · {format_rating(float(product["average_rating"]), int(product["review_count"]))}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="product-price">${float(product["price"]):.2f}</div>',
            unsafe_allow_html=True,
        )
        if source_label:
            st.markdown(f'<div class="product-source">Photo source: {source_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="product-description">{product["description"]}</div>', unsafe_allow_html=True)

        button_cols = st.columns(2)
        with button_cols[0]:
            if st.button("Add to cart", key=f"catalog_add_{key_prefix}_{product['id']}", width='stretch'):
                add_to_cart(int(product["id"]))
                st.rerun()
        with button_cols[1]:
            if st.button("Buy now", key=f"catalog_buy_{key_prefix}_{product['id']}", width='stretch'):
                confirmation = checkout_product(int(product["id"]))
                st.session_state.messages.append({"role": "assistant", "content": confirmation})
                st.rerun()


def run_assistant(prompt: str) -> None:
    if not prompt.strip():
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Finding the best matches..."):
        result = agent.invoke({"messages": st.session_state.messages})
        response = result["messages"][-1].content.replace("`", "")
    st.session_state.messages.append({"role": "assistant", "content": response})


def run_image_search(image_path: str, uploaded_name: str) -> None:
    matches = visual_similarity_search(image_path=image_path, top_k=5)
    st.session_state.last_visual_matches = matches
    if matches:
        lines = [f"Visual similarity matches for {uploaded_name}:"]
        for idx, product in enumerate(matches, start=1):
            lines.append(
                f"{idx}. {product['name']} (ID:{product['id']}) - similarity {float(product.get('similarity_score', 0)):.3f} - "
                f"${float(product['price']):.2f} - {float(product['average_rating']):.2f}/5"
            )
        response = "\n".join(lines)
    else:
        response = f"I could not find visual matches for {uploaded_name}, so I recommend browsing the catalog filters instead."
    st.session_state.messages.append({"role": "assistant", "content": response})


def render_catalog_section() -> None:
    st.markdown(
        """
        <div class="section-intro">
            <div class="section-title">Browse the catalog</div>
            <div class="section-copy">
                Keep the storefront simple: filter once, compare visually, and let the assistant handle deeper questions on the side.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    categories = get_categories()
    with st.container(border=True):
        with st.form("catalog_filters", clear_on_submit=False):
            row_one = st.columns([1.4, 0.9, 0.9])
            with row_one[0]:
                query = st.text_input(
                    "Search",
                    value=st.session_state.catalog_query,
                    placeholder="Search honey, oats, coffee, organic...",
                )
            with row_one[1]:
                category = st.selectbox(
                    "Category",
                    categories,
                    index=categories.index(st.session_state.catalog_category)
                    if st.session_state.catalog_category in categories
                    else 0,
                )
            with row_one[2]:
                sort_by = st.selectbox(
                    "Sort",
                    ["Top rated", "Price low to high", "Price high to low", "Most reviewed"],
                    index=["Top rated", "Price low to high", "Price high to low", "Most reviewed"].index(
                        st.session_state.catalog_sort
                    )
                    if st.session_state.catalog_sort in ["Top rated", "Price low to high", "Price high to low", "Most reviewed"]
                    else 0,
                )

            row_two = st.columns([1.0, 0.8, 0.8, 0.8])
            with row_two[0]:
                max_price = st.slider(
                    "Max price",
                    min_value=5.0,
                    max_value=35.0,
                    value=float(st.session_state.catalog_max_price),
                    step=1.0,
                )
            with row_two[1]:
                organic_only = st.toggle("Organic only", value=st.session_state.catalog_organic_only)
            with row_two[2]:
                page_size = st.selectbox(
                    "Cards per page",
                    options=[6, 9, 12],
                    index=[6, 9, 12].index(int(st.session_state.catalog_page_size))
                    if int(st.session_state.catalog_page_size) in [6, 9, 12]
                    else 1,
                )
            with row_two[3]:
                submitted = st.form_submit_button("Apply filters", width='stretch')

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

    if st.session_state.catalog_sort == "Price low to high":
        products = sorted(products, key=lambda item: (float(item["price"]), -float(item["average_rating"])))
    elif st.session_state.catalog_sort == "Price high to low":
        products = sorted(products, key=lambda item: (-float(item["price"]), -float(item["average_rating"])))
    elif st.session_state.catalog_sort == "Most reviewed":
        products = sorted(products, key=lambda item: (-int(item["review_count"]), -float(item["average_rating"])))
    else:
        products = sorted(products, key=lambda item: (-float(item["average_rating"]), -int(item["review_count"]), float(item["price"])))

    if not products:
        st.warning("No products matched the current filters.")
        return

    total_products = len(products)
    page_size = int(st.session_state.catalog_page_size)
    total_pages = max(1, math.ceil(total_products / page_size))
    st.session_state.catalog_page = min(max(1, st.session_state.catalog_page), total_pages)
    start = (st.session_state.catalog_page - 1) * page_size
    end = start + page_size
    current_page = products[start:end]

    summary_cols = st.columns([1.5, 1.0, 1.0])
    with summary_cols[0]:
        st.markdown(f"**{total_products} products** match your current view")
        st.markdown('<div class="toolbar-note">Use the assistant panel for conversational follow-up instead of crowding the catalog view.</div>', unsafe_allow_html=True)
    with summary_cols[1]:
        if st.button("Previous page", width='stretch', disabled=st.session_state.catalog_page <= 1):
            st.session_state.catalog_page -= 1
            st.rerun()
    with summary_cols[2]:
        if st.button("Next page", width='stretch', disabled=st.session_state.catalog_page >= total_pages):
            st.session_state.catalog_page += 1
            st.rerun()

    st.caption(f"Page {st.session_state.catalog_page} of {total_pages}")

    grid = st.columns(2)
    for idx, product in enumerate(current_page):
        with grid[idx % 2]:
            render_catalog_card(product, key_prefix=str(start + idx))


def render_message_block(message: dict, index: int) -> None:
    role = message.get("role", "assistant")
    display_text = display_message_text(message)
    wrapper_class = "msg-user" if role == "user" else "msg-assistant"
    label = "You" if role == "user" else "Shopping AI"
    title = "Visual search request" if "Visual search request for" in display_text else ""

    safe_text = html.escape(display_text).replace("\n", "<br>")

    st.markdown(f'<div class="msg {wrapper_class}">', unsafe_allow_html=True)
    st.markdown(f'<div class="msg-label">{label}</div>', unsafe_allow_html=True)
    if title and role == "user":
        st.markdown(f'<div class="msg-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="msg-body">{safe_text}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if role == "assistant":
        render_message_products(display_text, key_prefix=f"msg_{index}")


def render_message_products(text: str, key_prefix: str) -> None:
    product_ids = extract_product_ids(text)
    if not product_ids:
        return

    products = []
    for product_id in product_ids[:4]:
        product = get_product(product_id)
        if product:
            products.append(dict(product))

    if not products:
        return

    st.markdown("**Suggested products**")
    cols = st.columns(min(2, len(products)))
    for idx, product in enumerate(products):
        with cols[idx % len(cols)]:
            with st.container(border=True):
                st.markdown('<div class="visual-image">', unsafe_allow_html=True)
                product_image = get_product_image(product)
                st.image(product_image, width='content')
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown(f"**{product['name']}**")
                st.caption(f"${float(product['price']):.2f} · {format_rating(float(product['average_rating']), int(product['review_count']))}")
                if st.button("Add to cart", key=f"{key_prefix}_add_{product['id']}", width='stretch'):
                    add_to_cart(int(product["id"]))
                    st.rerun()


def render_assistant_chat_tab() -> None:
    quick_actions = st.columns(2)
    with quick_actions[0]:
        if st.button("Best rated", width='stretch'):
            run_assistant("Show me the best rated products in the store.")
            st.rerun()
    with quick_actions[1]:
        if st.button("Organic under $20", width='stretch'):
            run_assistant("Show me organic products under $20 with strong ratings.")
            st.rerun()

    quick_actions_two = st.columns(2)
    with quick_actions_two[0]:
        if st.button("Budget finds", width='stretch'):
            run_assistant("Show me the best budget-friendly products under $10.")
            st.rerun()
    with quick_actions_two[1]:
        if st.button("Compare honey", width='stretch'):
            run_assistant("Compare the best honey products with ratings and prices.")
            st.rerun()

    st.markdown('<div class="panel-gap"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        if not st.session_state.messages:
            st.markdown(
                """
                <div class="chat-empty">
                    Ask for recommendations, compare products, or use the visual search tab to find similar items from a package photo.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for idx, message in enumerate(st.session_state.messages):
                render_message_block(message, idx)

    with st.form("assistant_prompt_form", clear_on_submit=True):
        prompt = st.text_area(
            "Message the assistant",
            placeholder="Ask for something like organic honey under $20 with 4+ rating",
            height=100,
            label_visibility="collapsed",
        )
        send_col, note_col = st.columns([0.72, 0.28])
        with send_col:
            submitted = st.form_submit_button("Send request", width='stretch')
        with note_col:
            st.markdown('<div class="mini-note">The assistant can also add products to your cart directly from recommendations.</div>', unsafe_allow_html=True)

    if submitted:
        run_assistant(prompt)
        st.rerun()


def render_visual_search_tab() -> None:
    st.markdown("**Upload a product photo**")
    st.caption("Use this for a packaging photo or product front image. The assistant will rank visually similar catalog items.")

    uploaded = st.file_uploader("Product image", type=["jpg", "jpeg", "png", "webp"], key="assistant_image")
    if uploaded:
        st.markdown('<div class="visual-image">', unsafe_allow_html=True)
        st.image(uploaded, width='content')
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Search visual matches", width='stretch'):
            suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                image_path = tmp.name
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": f"I uploaded a product image. Please analyze it and find similar products in the store. Image path: {image_path}",
                }
            )
            st.session_state.last_uploaded_image = image_path
            with st.spinner("Searching by visual similarity..."):
                run_image_search(image_path=image_path, uploaded_name=uploaded.name)
            st.rerun()

    matches = st.session_state.get("last_visual_matches", [])
    if matches:
        st.markdown("**Top visual matches**")
        cols = st.columns(2)
        for idx, product in enumerate(matches[:4]):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.markdown('<div class="visual-image">', unsafe_allow_html=True)
                    product_image = get_product_image(product)
                    st.image(product_image, width='content')
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown(f"**{product['name']}**")
                    st.caption(
                        f"Similarity {float(product.get('similarity_score', 0)):.3f} · ${float(product['price']):.2f}"
                    )
                    if st.button("Add match to cart", key=f"visual_add_{product['id']}", width='stretch'):
                        add_to_cart(int(product["id"]))
                        st.rerun()
    elif st.session_state.get("last_uploaded_image"):
        st.info("No similar products found in our catalog. Try uploading a different product image.")


def render_cart_tab() -> None:
    items = get_cart_items()
    if not items:
        st.info("Your cart is empty. Add products from the catalog or from an assistant suggestion.")
        return

    st.markdown(f"**{cart_count()} item(s)** selected")
    st.caption(f"Subtotal: ${cart_subtotal():.2f}")
    st.markdown('<div class="panel-gap"></div>', unsafe_allow_html=True)

    for item in items:
        with st.container(border=True):
            row = st.columns([0.34, 0.66])
            with row[0]:
                st.markdown('<div class="cart-thumb">', unsafe_allow_html=True)
                product_image = get_product_image(item)
                st.image(product_image, width='content')
                st.markdown("</div>", unsafe_allow_html=True)
            with row[1]:
                st.markdown(f"**{item['name']}**")
                st.caption(f"${float(item['price']):.2f} each · quantity {item['qty']}")
                action_cols = st.columns(2)
                with action_cols[0]:
                    if st.button("Remove one", key=f"cart_remove_{item['id']}", width='stretch'):
                        remove_from_cart(int(item["id"]))
                        st.rerun()
                with action_cols[1]:
                    if st.button("Buy now", key=f"cart_buy_{item['id']}", width='stretch'):
                        confirmation = checkout_product(int(item["id"]))
                        remove_from_cart(int(item["id"]))
                        st.session_state.messages.append({"role": "assistant", "content": confirmation})
                        st.rerun()

    footer_cols = st.columns(2)
    with footer_cols[0]:
        if st.button("Clear cart", width='stretch'):
            clear_cart()
            st.rerun()
    with footer_cols[1]:
        if st.button("Checkout cart", width='stretch'):
            receipts = []
            for item in items:
                for _ in range(int(item["qty"])):
                    receipts.append(checkout_product(int(item["id"])))
            clear_cart()
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Cart checkout complete.\n\n" + "\n".join(receipts),
                }
            )
            st.rerun()


def render_assistant_panel() -> None:
    st.markdown(
        """
        <div class="assistant-frame">
            <div class="assistant-header">
                <div>
                    <div class="assistant-title">Shopping assistant</div>
                    <div class="assistant-copy">
                        A cleaner support-style panel for retrieval, visual search, and cart building. Use the tabs to keep each task focused.
                    </div>
                </div>
                <div class="assistant-badge">RAG + visual search</div>
            </div>
            <div>
                <span class="support-chip">Ask AI</span>
                <span class="support-chip">Visual search</span>
                <span class="support-chip">Cart actions</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chat_tab, visual_tab, cart_tab = st.tabs(["Assistant", "Visual Search", "Cart"])
    with chat_tab:
        render_assistant_chat_tab()
    with visual_tab:
        render_visual_search_tab()
    with cart_tab:
        render_cart_tab()


inject_css()
init_state()

summary = get_catalog_summary()
render_page_header(summary)

main_col, side_col = st.columns([1.9, 1.1], gap="large")
with main_col:
    render_catalog_section()
with side_col:
    render_assistant_panel()
