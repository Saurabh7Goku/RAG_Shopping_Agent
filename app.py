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

st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛒", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(245, 158, 11, 0.16), transparent 28%),
                    radial-gradient(circle at top right, rgba(14, 165, 233, 0.13), transparent 26%),
                    linear-gradient(180deg, #0f172a 0%, #0f172a 24%, #f8fafc 24%, #f8fafc 100%);
            }
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }
            [data-testid="stSidebar"] { display: none !important; }
            .hero {
                background: linear-gradient(135deg, #0f172a 0%, #111827 55%, #7c2d12 120%);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 28px;
                padding: 1.6rem 1.8rem;
                color: white;
                box-shadow: 0 18px 55px rgba(15, 23, 42, 0.28);
            }
            .hero-title {
                font-size: 2.3rem;
                line-height: 1.05;
                font-weight: 800;
                margin-bottom: 0.45rem;
            }
            .hero-subtitle {
                font-size: 1rem;
                opacity: 0.88;
                max-width: 52rem;
                margin-bottom: 1rem;
            }
            .pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 0.8rem;
            }
            .pill {
                background: rgba(255, 255, 255, 0.10);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 999px;
                padding: 0.4rem 0.75rem;
                font-size: 0.82rem;
            }
            .panel-card {
                background: rgba(255, 255, 255, 0.88);
                border: 1px solid rgba(148, 163, 184, 0.20);
                border-radius: 24px;
                padding: 1rem 1rem 0.8rem 1rem;
                box-shadow: 0 20px 50px rgba(15, 23, 42, 0.10);
                backdrop-filter: blur(8px);
            }
            .section-title {
                font-size: 1.05rem;
                font-weight: 750;
                margin: 0.35rem 0 0.2rem 0;
                color: #0f172a;
            }
            .section-subtitle {
                color: #475569;
                font-size: 0.86rem;
                margin-bottom: 0.7rem;
            }
            .metric-box {
                background: rgba(255,255,255,0.72);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 20px;
                padding: 0.85rem 1rem;
                min-height: 88px;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
            }
            .metric-label {
                color: #64748b;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.25rem;
            }
            .metric-value {
                color: #0f172a;
                font-size: 1.55rem;
                font-weight: 800;
                line-height: 1.0;
            }
            .metric-note {
                color: #475569;
                font-size: 0.82rem;
                margin-top: 0.3rem;
            }
            .product-card {
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 22px;
                padding: 0.75rem;
                box-shadow: 0 14px 28px rgba(15, 23, 42, 0.07);
                min-height: 100%;
            }
            .product-name {
                font-size: 1rem;
                font-weight: 750;
                color: #0f172a;
                margin-top: 0.55rem;
            }
            .product-meta {
                color: #475569;
                font-size: 0.84rem;
                margin-top: 0.15rem;
            }
            .product-price {
                color: #0f766e;
                font-weight: 800;
                margin-top: 0.3rem;
                font-size: 0.95rem;
            }
            .badge {
                display: inline-block;
                padding: 0.24rem 0.55rem;
                border-radius: 999px;
                background: #ecfeff;
                color: #0f766e;
                border: 1px solid #99f6e4;
                font-size: 0.76rem;
                font-weight: 700;
                margin-top: 0.35rem;
            }
            .chat-shell {
                background: rgba(255,255,255,0.92);
                border: 1px solid rgba(148, 163, 184, 0.20);
                border-radius: 24px;
                box-shadow: 0 20px 50px rgba(15, 23, 42, 0.10);
                padding: 1rem;
            }
            .chat-msg {
                border-radius: 18px;
                padding: 0.65rem 0.8rem;
            }
            .chat-msg.user {
                background: linear-gradient(135deg, #dbeafe, #eff6ff);
            }
            .chat-msg.assistant {
                background: linear-gradient(135deg, #fff7ed, #fffbeb);
            }
            .tiny {
                color: #64748b;
                font-size: 0.78rem;
            }
            .support-pill {
                display: inline-block;
                padding: 0.25rem 0.6rem;
                border-radius: 999px;
                background: #f8fafc;
                color: #334155;
                border: 1px solid #e2e8f0;
                font-size: 0.76rem;
                margin: 0 0.25rem 0.25rem 0;
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
        "assistant_input": "",
        "last_uploaded_image": None,
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
    cur.execute(
        """
        SELECT ROUND(AVG(rating), 2) AS avg_rating
        FROM reviews
        """
    )
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


def get_cart_items() -> list[dict]:
    items = []
    for product_id, qty in st.session_state.cart.items():
        product = get_product(product_id)
        if product:
            product = dict(product)
            product["qty"] = qty
            items.append(product)
    items.sort(key=lambda item: item["name"])
    return items


def cart_count() -> int:
    return sum(st.session_state.cart.values())


def cart_subtotal() -> float:
    subtotal = 0.0
    for item in get_cart_items():
        subtotal += float(item["price"]) * int(item["qty"])
    return subtotal


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


def render_metric(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rating(rating: float, review_count: int) -> str:
    stars = "★" * int(round(rating))
    return f"{rating:.2f} {stars} ({review_count} reviews)"


def render_product_card(product: dict, key_prefix: str) -> None:
    image_path = product.get("image_path") or ""
    badge = "Organic" if product.get("is_organic") else "Best seller"
    with st.container(border=True):
        if image_path and os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.image(str(FALLBACK_IMAGE), use_container_width=True)

        st.markdown(f'<div class="badge">{badge}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="product-name">{product["name"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="product-meta">{product["category"] or "General"} • {render_rating(float(product["average_rating"]), int(product["review_count"]))}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="product-price">${float(product["price"]):.2f}</div>',
            unsafe_allow_html=True,
        )
        st.caption(product["description"])

        left, right = st.columns(2)
        with left:
            if st.button("Add to cart", key=f"add_{key_prefix}_{product['id']}", use_container_width=True):
                add_to_cart(int(product["id"]))
                st.rerun()
        with right:
            if st.button("Buy now", key=f"buy_{key_prefix}_{product['id']}", use_container_width=True):
                confirmation = checkout_product(int(product["id"]))
                st.session_state.messages.append({"role": "assistant", "content": confirmation})
                st.rerun()


def render_cart_panel() -> None:
    items = get_cart_items()
    st.markdown("### Cart")
    st.caption(f"{cart_count()} item(s) selected")

    if not items:
        st.info("Your cart is empty. Add a product from the catalog or from the assistant suggestions.")
        return

    for item in items:
        cols = st.columns([0.85, 1.3, 0.6])
        with cols[0]:
            if item.get("image_path") and os.path.exists(item["image_path"]):
                st.image(item["image_path"], use_container_width=True)
        with cols[1]:
            st.markdown(f"**{item['name']}**")
            st.caption(f"${float(item['price']):.2f} x {item['qty']}")
        with cols[2]:
            if st.button("Remove", key=f"remove_{item['id']}", use_container_width=True):
                remove_from_cart(int(item["id"]))
                st.rerun()

    st.markdown(f"**Subtotal:** ${cart_subtotal():.2f}")
    left, right = st.columns(2)
    with left:
        if st.button("Clear cart", use_container_width=True):
            clear_cart()
            st.rerun()
    with right:
        if st.button("Checkout cart", use_container_width=True):
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


def extract_product_ids(text: str) -> list[int]:
    ids = []
    seen = set()
    for match in re.findall(r"\(ID:(\d+)\)", text or ""):
        value = int(match)
        if value not in seen:
            seen.add(value)
            ids.append(value)
    return ids


def render_suggested_products(message: str) -> None:
    ids = extract_product_ids(message)
    if not ids:
        return

    products = []
    for product_id in ids:
        product = get_product(product_id)
        if product:
            products.append(dict(product))

    if not products:
        return

    st.markdown("**Quick add**")
    cols = st.columns(min(2, len(products)))
    for index, product in enumerate(products):
        with cols[index % len(cols)]:
            with st.container(border=True):
                if product.get("image_path") and os.path.exists(product["image_path"]):
                    st.image(product["image_path"], use_container_width=True)
                st.markdown(f"**{product['name']}**")
                st.caption(f"${float(product['price']):.2f} • {render_rating(float(product['average_rating']), int(product['review_count']))}")
                if st.button(f"Add {product['id']} to cart", key=f"quick_add_{product['id']}"):
                    add_to_cart(int(product["id"]))
                    st.rerun()


def run_assistant(prompt: str) -> None:
    if not prompt.strip():
        return
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking about the best picks..."):
        result = agent.invoke({"messages": st.session_state.messages})
        response = result["messages"][-1].content.replace("`", "")
    st.session_state.messages.append({"role": "assistant", "content": response})


def run_image_search(image_path: str, uploaded_name: str) -> None:
    matches = visual_similarity_search(image_path=image_path, top_k=5)
    if matches:
        lines = [
            f"Visually similar matches for **{uploaded_name}** using embedding search:"
        ]
        for idx, product in enumerate(matches, start=1):
            lines.append(
                f"{idx}. {product['name']} (ID:{product['id']}) - similarity {float(product.get('similarity_score', 0)):.3f} - "
                f"${float(product['price']):.2f} - {float(product['average_rating']):.2f} stars"
            )
        response = "\n".join(lines)
    else:
        response = (
            f"I could not compute embedding matches for **{uploaded_name}**, so I fell back to regular catalog search."
        )
    st.session_state.messages.append({"role": "assistant", "content": response})


def catalog_controls() -> None:
    st.markdown("### Discover the catalog")
    st.caption("Browse everything available, then refine the selection with filters. The assistant panel can add items to cart directly.")

    categories = get_categories()
    with st.form("catalog_filters", clear_on_submit=False):
        left, middle, right = st.columns([1.2, 0.9, 0.9])
        with left:
            query = st.text_input("Search products", value=st.session_state.catalog_query, placeholder="Try honey, coffee, organic, oats...")
        with middle:
            category = st.selectbox("Category", categories, index=categories.index(st.session_state.catalog_category) if st.session_state.catalog_category in categories else 0)
        with right:
            sort_by = st.selectbox("Sort by", ["Top rated", "Price low to high", "Price high to low", "Most reviewed"])

        price_col, organic_col, submit_col = st.columns([0.8, 0.8, 0.8])
        with price_col:
            max_price = st.slider("Max price", min_value=0.0, max_value=35.0, value=float(st.session_state.catalog_max_price), step=1.0)
        with organic_col:
            organic_only = st.toggle("Organic only", value=st.session_state.catalog_organic_only)
        with submit_col:
            apply_filters = st.form_submit_button("Refresh results", use_container_width=True)

    if apply_filters:
        st.session_state.catalog_query = query.strip()
        st.session_state.catalog_category = category
        st.session_state.catalog_sort = sort_by
        st.session_state.catalog_max_price = max_price
        st.session_state.catalog_organic_only = organic_only

    products = fetch_products(
        query=st.session_state.catalog_query,
        max_price=st.session_state.catalog_max_price if st.session_state.catalog_max_price > 0 else None,
        is_organic=True if st.session_state.catalog_organic_only else None,
        category=None if st.session_state.catalog_category == "All" else st.session_state.catalog_category,
        limit=24,
    )

    if st.session_state.catalog_sort == "Price low to high":
        products = sorted(products, key=lambda item: (float(item["price"]), -float(item["average_rating"])))
    elif st.session_state.catalog_sort == "Price high to low":
        products = sorted(products, key=lambda item: (-float(item["price"]), -float(item["average_rating"])))
    elif st.session_state.catalog_sort == "Most reviewed":
        products = sorted(products, key=lambda item: (-int(item["review_count"]), -float(item["average_rating"])))
    else:
        products = sorted(products, key=lambda item: (-float(item["average_rating"]), -int(item["review_count"]), float(item["price"])))

    st.markdown(f"Showing {len(products)} product(s)")

    if not products:
        st.warning("No products matched the current filters.")
        return

    grid = st.columns(3)
    for idx, product in enumerate(products):
        with grid[idx % 3]:
            render_product_card(product, key_prefix="catalog")


def render_hero(summary: dict) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">AI Shopping Assistant</div>
            <div class="hero-subtitle">
                A storefront-style shopping experience with visual catalog browsing, image search, and a Rufus-style assistant panel for conversational shopping and one-click cart actions.
            </div>
            <div class="pill-row">
                <span class="pill">Hybrid search</span>
                <span class="pill">Visual catalog</span>
                <span class="pill">AI recommendations</span>
                <span class="pill">One-click cart</span>
                <span class="pill">Image search</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    metrics = st.columns(4)
    with metrics[0]:
        render_metric("Products", str(summary["total"]), "All items are visible in the catalog")
    with metrics[1]:
        render_metric("Organic picks", str(summary["organic"]), "Health-focused options are highlighted")
    with metrics[2]:
        render_metric("Visual assets", str(summary["visual"]), "Every product has a store image")
    with metrics[3]:
        render_metric("Avg rating", f"{summary['avg_rating']:.2f}", "Review-aware ranking is enabled")


def render_assistant_panel() -> None:
    with st.container(border=True):
        st.markdown("### Shopping Assistant")
        st.caption("Ask for products, compare prices, search by image, or add items directly from recommendations.")

        uploaded = st.file_uploader("Shop by image", type=["jpg", "jpeg", "png", "webp"], key="assistant_image")
        if uploaded:
            st.image(uploaded, use_container_width=True)
            if st.button("Find similar products", use_container_width=True):
                suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    image_path = tmp.name
                prompt = (
                    "I uploaded a product image. Please analyze it and find similar products in the store. "
                    f"Image path: {image_path}"
                )
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.last_uploaded_image = image_path
                with st.spinner("Searching by visual embeddings..."):
                    run_image_search(image_path=image_path, uploaded_name=uploaded.name)
                st.rerun()

        quick_row = st.columns(2)
        with quick_row[0]:
            if st.button("Best rated", use_container_width=True):
                run_assistant("Show me the best rated products in the store.")
                st.rerun()
        with quick_row[1]:
            if st.button("Organic picks", use_container_width=True):
                run_assistant("Show me organic products under $20 with strong ratings.")
                st.rerun()

        quick_row_2 = st.columns(2)
        with quick_row_2[0]:
            if st.button("Budget finds", use_container_width=True):
                run_assistant("Show me the best budget-friendly products under $10.")
                st.rerun()
        with quick_row_2[1]:
            if st.button("Honey ideas", use_container_width=True):
                run_assistant("Show me the best honey products with ratings and prices.")
                st.rerun()

    with st.container(border=True):
        st.markdown("### Conversation")
        if st.session_state.messages:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant":
                        render_suggested_products(msg["content"])
        else:
            st.info("Ask me for product suggestions, image matches, or a cart-friendly comparison.")

    with st.form("assistant_composer", clear_on_submit=True):
        prompt = st.text_area(
            "Message the assistant",
            value="",
            placeholder="Ask for something like: organic honey under $20 with 4+ rating",
            height=90,
            label_visibility="collapsed",
        )
        send_col, tip_col = st.columns([0.75, 0.25])
        with send_col:
            submitted = st.form_submit_button("Send", use_container_width=True)
        with tip_col:
            st.markdown('<span class="tiny">Use the assistant to refine the catalog or build your cart.</span>', unsafe_allow_html=True)

    if submitted:
        run_assistant(prompt)
        st.rerun()

    st.markdown("<div style='height: 0.45rem;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        render_cart_panel()


inject_css()
init_state()

st.markdown(
    """
    <div class="support-pill">Storefront</div>
    <div class="support-pill">AI Assistant</div>
    <div class="support-pill">Cart</div>
    <div class="support-pill">Image Search</div>
    """,
    unsafe_allow_html=True,
)

summary = get_catalog_summary()
render_hero(summary)

left_col, right_col = st.columns([1.75, 1.0], gap="large")
with left_col:
    catalog_controls()

with right_col:
    render_assistant_panel()
