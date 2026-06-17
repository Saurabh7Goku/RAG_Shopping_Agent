import base64
import json
import os
import sqlite3
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from PIL import Image, ImageDraw, ImageFont

from reviews_api import get_product_rating, get_ratings_for_products

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except Exception:  # pragma: no cover - graceful fallback when deps are absent
    torch = None
    CLIPModel = None
    CLIPProcessor = None

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "store.db"
RESOURCE_DIR = BASE_DIR / "resources"
GENERATED_DIR = RESOURCE_DIR / "generated"
EMBEDDING_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_CACHE_PATH = RESOURCE_DIR / "embedding_cache.npz"
EMBEDDING_CACHE_META_PATH = RESOURCE_DIR / "embedding_cache_meta.json"

_EMBEDDING_MODEL = None
_EMBEDDING_PROCESSOR = None
_EMBEDDING_DEVICE = "cpu"
_PRODUCT_EMBEDDING_CACHE: dict[int, list[float]] = {}
_EMBEDDING_CACHE_READY = False
_EMBEDDING_CACHE_LAST_REFRESH: dict[str, int] = {
    "loaded": 0,
    "reused": 0,
    "updated": 0,
    "removed": 0,
}

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def _ensure_column(cursor: sqlite3.Cursor, table: str, column: str, ddl: str) -> None:
    if not _column_exists(cursor, table, column):
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _ensure_table(cursor: sqlite3.Cursor, ddl: str) -> None:
    cursor.execute(ddl)


def _embedding_components():
    global _EMBEDDING_MODEL, _EMBEDDING_PROCESSOR

    if torch is None or CLIPModel is None or CLIPProcessor is None:
        return None, None

    if _EMBEDDING_MODEL is None or _EMBEDDING_PROCESSOR is None:
        _EMBEDDING_PROCESSOR = CLIPProcessor.from_pretrained(EMBEDDING_MODEL_NAME)
        _EMBEDDING_MODEL = CLIPModel.from_pretrained(EMBEDDING_MODEL_NAME)
        _EMBEDDING_MODEL.to(_EMBEDDING_DEVICE)
        _EMBEDDING_MODEL.eval()

    return _EMBEDDING_MODEL, _EMBEDDING_PROCESSOR


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _encode_image(image_path: str) -> list[float] | None:
    try:
        model, processor = _embedding_components()
        if model is None or processor is None:
            return None

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(_EMBEDDING_DEVICE) for key, value in inputs.items()}

        with torch.no_grad():
            output = model.get_image_features(pixel_values=inputs["pixel_values"])
            if isinstance(output, torch.Tensor):
                features = output
            else:
                features = getattr(output, "pooler_output", None)
            if features is None or not isinstance(features, torch.Tensor):
                return None
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy().astype(np.float32).tolist()
    except Exception:
        return None


def _ensure_embeddings_table(cursor: sqlite3.Cursor) -> None:
    _ensure_table(
        cursor,
        """
        CREATE TABLE IF NOT EXISTS product_embeddings (
            product_id INTEGER PRIMARY KEY,
            model_name TEXT NOT NULL,
            image_embedding TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY(product_id) REFERENCES products(id)
        )
        """,
    )


def _store_product_embedding(cursor: sqlite3.Cursor, product_id: int, embedding: list[float]) -> None:
    cursor.execute(
        """
        INSERT INTO product_embeddings (product_id, model_name, image_embedding, updated_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(product_id) DO UPDATE SET
            model_name = excluded.model_name,
            image_embedding = excluded.image_embedding,
            updated_at = datetime('now')
        """,
        (product_id, EMBEDDING_MODEL_NAME, json.dumps(embedding)),
    )


def _get_embedding_signatures() -> list[dict[str, int | str]]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_path FROM products ORDER BY id")
    rows = cursor.fetchall()
    conn.close()

    signatures: list[dict[str, int | str]] = []
    for row in rows:
        image_path = row["image_path"]
        if not image_path or not os.path.exists(image_path):
            continue
        stat = os.stat(image_path)
        signatures.append(
            {
                "product_id": int(row["id"]),
                "image_path": image_path,
                "mtime_ns": int(stat.st_mtime_ns),
                "size": int(stat.st_size),
            }
        )
    return signatures


def _signature_map(signatures: list[dict[str, int | str]]) -> dict[int, dict[str, int | str]]:
    return {int(item["product_id"]): item for item in signatures}


def _signature_changed(left: dict[str, int | str] | None, right: dict[str, int | str] | None) -> bool:
    if left is None or right is None:
        return True
    return (
        str(left.get("image_path", "")) != str(right.get("image_path", ""))
        or int(left.get("mtime_ns", -1)) != int(right.get("mtime_ns", -1))
        or int(left.get("size", -1)) != int(right.get("size", -1))
    )


def _load_disk_cache_payload() -> tuple[dict[int, list[float]], list[dict[str, int | str]]]:
    payload = np.load(EMBEDDING_CACHE_PATH, allow_pickle=False)
    product_ids = payload["product_ids"].astype(np.int64)
    embeddings = payload["embeddings"].astype(np.float32)
    meta = json.loads(EMBEDDING_CACHE_META_PATH.read_text(encoding="utf-8"))
    cached_vectors = {
        int(product_id): embeddings[idx].astype(np.float32).tolist()
        for idx, product_id in enumerate(product_ids)
    }
    cached_signatures = meta.get("signatures", [])
    return cached_vectors, cached_signatures


def _sync_embedding_table_from_cache(cache_items: list[tuple[int, list[float]]]) -> None:
    conn = _connect()
    cursor = conn.cursor()
    for product_id, embedding in cache_items:
        _store_product_embedding(cursor, product_id, embedding)
    conn.commit()
    conn.close()


def _write_embedding_cache(cache_items: list[tuple[int, list[float]]]) -> None:
    signatures = _get_embedding_signatures()
    EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    product_ids = np.array([item[0] for item in cache_items], dtype=np.int64)
    embeddings = np.array([item[1] for item in cache_items], dtype=np.float32)
    np.savez_compressed(
        EMBEDDING_CACHE_PATH,
        product_ids=product_ids,
        embeddings=embeddings,
    )
    EMBEDDING_CACHE_META_PATH.write_text(
        json.dumps(
            {
                "model_name": EMBEDDING_MODEL_NAME,
                "signatures": signatures,
                "product_count": len(cache_items),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _rebuild_embedding_cache() -> bool:
    global _PRODUCT_EMBEDDING_CACHE, _EMBEDDING_CACHE_READY, _EMBEDDING_CACHE_LAST_REFRESH

    if torch is None or CLIPModel is None or CLIPProcessor is None:
        return False

    try:
        conn = _connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.id, p.image_path
            FROM products p
            ORDER BY p.id
            """
        )
        rows = cursor.fetchall()
        conn.close()

        cache_items: list[tuple[int, list[float]]] = []
        for row in rows:
            image_path = row["image_path"]
            if not image_path or not os.path.exists(image_path):
                continue
            embedding = _encode_image(image_path)
            if embedding is None:
                continue
            cache_items.append((int(row["id"]), embedding))

        _PRODUCT_EMBEDDING_CACHE = {product_id: embedding for product_id, embedding in cache_items}
        _write_embedding_cache(cache_items)
        _sync_embedding_table_from_cache(cache_items)
        _EMBEDDING_CACHE_READY = True
        _EMBEDDING_CACHE_LAST_REFRESH = {
            "loaded": len(cache_items),
            "reused": 0,
            "updated": len(cache_items),
            "removed": 0,
        }
        return True
    except Exception:
        return False


def _refresh_embedding_cache_incrementally() -> bool:
    global _PRODUCT_EMBEDDING_CACHE, _EMBEDDING_CACHE_READY, _EMBEDDING_CACHE_LAST_REFRESH

    if torch is None or CLIPModel is None or CLIPProcessor is None:
        return False

    try:
        current_signatures = _signature_map(_get_embedding_signatures())

        if not EMBEDDING_CACHE_PATH.exists() or not EMBEDDING_CACHE_META_PATH.exists():
            return _rebuild_embedding_cache()

        meta = json.loads(EMBEDDING_CACHE_META_PATH.read_text(encoding="utf-8"))
        if meta.get("model_name") != EMBEDDING_MODEL_NAME:
            return _rebuild_embedding_cache()

        cached_vectors, cached_signatures_list = _load_disk_cache_payload()
        _PRODUCT_EMBEDDING_CACHE = cached_vectors
        cached_signatures = _signature_map(cached_signatures_list)

        existing_ids = set(_PRODUCT_EMBEDDING_CACHE.keys())
        current_ids = set(current_signatures.keys())
        stale_ids = existing_ids - current_ids

        changed_or_new_ids = [
            product_id
            for product_id, signature in current_signatures.items()
            if _signature_changed(cached_signatures.get(product_id), signature)
        ]

        if not stale_ids and not changed_or_new_ids:
            _EMBEDDING_CACHE_READY = True
            _EMBEDDING_CACHE_LAST_REFRESH = {
                "loaded": len(_PRODUCT_EMBEDDING_CACHE),
                "reused": len(existing_ids),
                "updated": 0,
                "removed": 0,
            }
            return True

        conn = _connect()
        cursor = conn.cursor()

        for product_id in stale_ids:
            _PRODUCT_EMBEDDING_CACHE.pop(product_id, None)
            cursor.execute("DELETE FROM product_embeddings WHERE product_id = ?", (product_id,))

        for product_id in changed_or_new_ids:
            image_path = str(current_signatures[product_id]["image_path"])
            if not os.path.exists(image_path):
                continue
            embedding = _encode_image(image_path)
            if embedding is None:
                continue
            _PRODUCT_EMBEDDING_CACHE[product_id] = embedding
            _store_product_embedding(cursor, product_id, embedding)

        conn.commit()
        conn.close()

        ordered_cache_items: list[tuple[int, list[float]]] = []
        for product_id in sorted(_PRODUCT_EMBEDDING_CACHE.keys()):
            ordered_cache_items.append((product_id, _PRODUCT_EMBEDDING_CACHE[product_id]))

        _write_embedding_cache(ordered_cache_items)
        _EMBEDDING_CACHE_READY = True
        _EMBEDDING_CACHE_LAST_REFRESH = {
            "loaded": len(_PRODUCT_EMBEDDING_CACHE),
            "reused": len(current_ids) - len(changed_or_new_ids),
            "updated": len(changed_or_new_ids),
            "removed": len(stale_ids),
        }
        return True
    except Exception:
        return False


def _ensure_embedding_cache() -> None:
    global _EMBEDDING_CACHE_READY
    if _EMBEDDING_CACHE_READY and _PRODUCT_EMBEDDING_CACHE:
        return
    if _refresh_embedding_cache_incrementally():
        return
    _rebuild_embedding_cache()


def _load_product_embeddings() -> list[dict]:
    _ensure_embedding_cache()
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, name, category, price, description, is_organic, image_path
        FROM products
        ORDER BY id
        """
    )
    rows = cursor.fetchall()
    conn.close()

    products = []
    for row in rows:
        embedding = _PRODUCT_EMBEDDING_CACHE.get(int(row["id"]))
        products.append(
            {
                "id": row["id"],
                "name": row["name"],
                "category": row["category"],
                "price": row["price"],
                "description": row["description"],
                "is_organic": bool(row["is_organic"]),
                "image_path": row["image_path"],
                "embedding": embedding,
                "model_name": EMBEDDING_MODEL_NAME if embedding is not None else None,
            }
        )
    return products


def _category_theme(category: str, product_name: str) -> tuple[str, str]:
    category = (category or "").lower()
    name = (product_name or "").lower()

    if "honey" in category or "honey" in name:
        return ("#F7B500", "#D97706")
    if "oil" in category:
        return ("#84CC16", "#3F6212")
    if "grain" in category or "oat" in name or "rice" in name:
        return ("#D6B38F", "#A16207")
    if "nut" in category or "seed" in category:
        return ("#B45309", "#78350F")
    if "tea" in category:
        return ("#34D399", "#0F766E")
    if "coffee" in category:
        return ("#8B5E34", "#4C1D95")
    if "snack" in category:
        return ("#FB7185", "#BE185D")
    if "dairy" in category:
        return ("#38BDF8", "#0369A1")
    return ("#94A3B8", "#334155")


def _wrap_label(text: str, width: int = 18) -> list[str]:
    return textwrap.wrap(text, width=width) or [text]


def _generate_product_image(product_id: int, name: str, category: str, is_organic: bool) -> str:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GENERATED_DIR / f"product_{product_id}.png"
    if out_path.exists():
        return str(out_path)

    start_hex, end_hex = _category_theme(category, name)
    start_rgb = tuple(int(start_hex[i : i + 2], 16) for i in (1, 3, 5))
    end_rgb = tuple(int(end_hex[i : i + 2], 16) for i in (1, 3, 5))

    width, height = 900, 900
    image = Image.new("RGB", (width, height), start_rgb)
    draw = ImageDraw.Draw(image)

    for y in range(height):
        ratio = y / max(height - 1, 1)
        row_color = tuple(
            int(start_rgb[i] * (1 - ratio) + end_rgb[i] * ratio) for i in range(3)
        )
        draw.line((0, y, width, y), fill=row_color)

    # Decorative orbs for a more premium storefront feel.
    draw.ellipse((620, 110, 980, 470), fill=(255, 255, 255))
    draw.ellipse((540, 530, 900, 890), fill=(245, 245, 245))
    draw.rounded_rectangle((70, 70, 830, 830), radius=54, outline=(255, 255, 255), width=6)

    try:
        font_big = ImageFont.truetype("arial.ttf", 54)
        font_mid = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font_big = ImageFont.load_default()
        font_mid = ImageFont.load_default()
        font_small = ImageFont.load_default()

    badge = "ORGANIC" if is_organic else "CURATED PICK"
    draw.rounded_rectangle((110, 120, 320, 180), radius=22, fill=(255, 255, 255, 215))
    draw.text((145, 138), badge, fill="#0F172A", font=font_small)

    label_lines = _wrap_label(name, 16)
    y_cursor = 260
    for line in label_lines:
        draw.text((110, y_cursor), line, fill="white", font=font_big)
        y_cursor += 62

    draw.text((110, 650), category.upper(), fill=(255, 255, 255, 235), font=font_mid)
    draw.text((110, 702), f"Product ID #{product_id:02d}", fill=(255, 255, 255, 210), font=font_small)
    draw.text((110, 748), "AI curated storefront asset", fill=(255, 255, 255, 210), font=font_small)

    image.save(out_path)
    return str(out_path)


def ensure_database() -> None:
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            description TEXT,
            is_organic INTEGER DEFAULT 0
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            rating REAL,
            reviewer_name TEXT,
            review_text TEXT,
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            price REAL NOT NULL,
            ordered_at TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
        """
    )

    _ensure_column(cursor, "products", "image_path", "TEXT")
    _ensure_embeddings_table(cursor)

    conn.commit()
    conn.close()


def seed_product_images() -> None:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, category, is_organic FROM products ORDER BY id")
    rows = cursor.fetchall()

    for row in rows:
        image_path = _generate_product_image(row["id"], row["name"], row["category"] or "", bool(row["is_organic"]))
        cursor.execute("UPDATE products SET image_path = ? WHERE id = ?", (image_path, row["id"]))

    conn.commit()
    conn.close()


def seed_product_embeddings() -> None:
    _ensure_embedding_cache()


def bootstrap_data() -> None:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM products")
    product_count = cursor.fetchone()[0]
    conn.close()

    if product_count == 0:
        from setup_db import create_database

        create_database()

    ensure_database()
    seed_product_images()
    seed_product_embeddings()


def get_product(product_id: int) -> Optional[dict]:
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, name, category, price, description, is_organic, image_path
        FROM products
        WHERE id = ?
        """,
        (product_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None

    rating_info = get_product_rating(product_id)
    return {
        "id": row["id"],
        "name": row["name"],
        "category": row["category"],
        "price": row["price"],
        "description": row["description"],
        "is_organic": bool(row["is_organic"]),
        "image_path": row["image_path"],
        "average_rating": rating_info["average_rating"],
        "review_count": rating_info["review_count"],
    }


def fetch_products(
    query: str = "",
    max_price: Optional[float] = None,
    is_organic: Optional[bool] = None,
    category: Optional[str] = None,
    limit: Optional[int] = 12,
) -> list[dict]:
    bootstrap_data()
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, name, category, price, description, is_organic, image_path
        FROM products
        ORDER BY id
        """
    )
    rows = cursor.fetchall()
    conn.close()

    ratings = get_ratings_for_products([row["id"] for row in rows])
    ratings_map = {r["product_id"]: r for r in ratings}

    tokens = [token for token in (query or "").lower().replace("-", " ").split() if token]
    category = category.lower().strip() if category else None

    results: list[dict] = []
    for row in rows:
        product_category = (row["category"] or "").lower()
        product_name = (row["name"] or "").lower()
        product_description = (row["description"] or "").lower()
        product_is_organic = bool(row["is_organic"])
        rating = ratings_map.get(row["id"], {"average_rating": 0.0, "review_count": 0})

        if max_price is not None and float(row["price"]) > float(max_price):
            continue
        if is_organic is True and not product_is_organic:
            continue
        if is_organic is False and product_is_organic:
            continue
        if category and category not in product_category:
            continue

        relevance = 0.0
        if tokens:
            haystack = f"{product_name} {product_category} {product_description}"
            for token in tokens:
                if token in product_name:
                    relevance += 3.0
                if token in product_category:
                    relevance += 2.2
                if token in product_description:
                    relevance += 1.0
                if token in haystack:
                    relevance += 0.3

        if query and query.lower() in product_name:
            relevance += 3.5

        if max_price is not None:
            relevance += max(0.0, (float(max_price) - float(row["price"])) / max(float(max_price), 1.0))

        if is_organic is True and product_is_organic:
            relevance += 1.5

        relevance += float(rating["average_rating"]) * 1.2
        relevance += float(rating["review_count"]) * 0.08

        results.append(
            {
                "id": row["id"],
                "name": row["name"],
                "category": row["category"],
                "price": row["price"],
                "description": row["description"],
                "is_organic": product_is_organic,
                "image_path": row["image_path"],
                "average_rating": rating["average_rating"],
                "review_count": rating["review_count"],
                "relevance_score": round(relevance, 3),
            }
        )

    results.sort(key=lambda item: (-item["relevance_score"], -item["average_rating"], item["price"]))
    if limit is not None:
        results = results[:limit]
    return results


def visual_similarity_search(
    image_path: str,
    top_k: int = 5,
    max_price: Optional[float] = None,
    is_organic: Optional[bool] = None,
    category: Optional[str] = None,
) -> list[dict]:
    bootstrap_data()

    query_embedding = _encode_image(image_path)
    if query_embedding is None:
        # Graceful fallback if the embedding stack is unavailable.
        return fetch_products(
            query="",
            max_price=max_price,
            is_organic=is_organic,
            category=category,
            limit=top_k,
        )

    products = _load_product_embeddings()
    query_vector = np.array(query_embedding, dtype=np.float32)
    category = category.lower().strip() if category else None
    ratings = get_ratings_for_products([product["id"] for product in products])
    ratings_map = {item["product_id"]: item for item in ratings}

    ranked: list[dict] = []
    for product in products:
        if not product.get("embedding"):
            continue

        product_category = (product["category"] or "").lower()
        if max_price is not None and float(product["price"]) > float(max_price):
            continue
        if is_organic is True and not product["is_organic"]:
            continue
        if is_organic is False and product["is_organic"]:
            continue
        if category and category not in product_category:
            continue

        product_vector = np.array(product["embedding"], dtype=np.float32)
        similarity = _cosine_similarity(query_vector, product_vector)

        ranked.append(
            {
                "id": product["id"],
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "description": product["description"],
                "is_organic": product["is_organic"],
                "image_path": product["image_path"],
                "average_rating": ratings_map.get(product["id"], {}).get("average_rating", 0.0),
                "review_count": ratings_map.get(product["id"], {}).get("review_count", 0),
                "similarity_score": round(similarity, 4),
                "match_type": "visual_embedding",
            }
        )

    ranked.sort(
        key=lambda item: (
            -float(item["similarity_score"]),
            -float(item["average_rating"]),
            float(item["price"]),
        )
    )
    return ranked[:top_k]


def checkout_product(product_id: int) -> str:
    bootstrap_data()
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row["name"], float(row["price"])
    cursor.execute(
        "INSERT INTO orders (product_id, product_name, price) VALUES (?, ?, ?)",
        (product_id, name, price),
    )
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return (
        f"Order #{order_id} confirmed! '{name}' has been added to your orders for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days."
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_products(
    query: str,
    max_price: Optional[float] = None,
    is_organic: Optional[bool] = None,
    category: Optional[str] = None,
    limit: Optional[int] = 8,
) -> str:
    """
    Search the product database by keyword and optional filters.
    Returns a JSON array of matching products with ratings and image paths.
    """
    products = fetch_products(
        query=query,
        max_price=max_price,
        is_organic=is_organic,
        category=category,
        limit=limit,
    )
    return json.dumps(products)


@tool
def get_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for a product by its ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    result = get_product_rating(product_id)
    return json.dumps(result)


@tool
def checkout(product_id: int) -> str:
    """
    Place an order for the given product ID and return a confirmation message.
    """
    return checkout_product(product_id)


@tool
def search_similar_products(image_path: str, top_k: int = 5, max_price: Optional[float] = None, is_organic: Optional[bool] = None, category: Optional[str] = None) -> str:
    """
    Search for products visually similar to the uploaded image using CLIP embeddings.
    Returns a JSON array with similarity scores and product metadata.
    """
    matches = visual_similarity_search(
        image_path=image_path,
        top_k=top_k,
        max_price=max_price,
        is_organic=is_organic,
        category=category,
    )
    return json.dumps(matches)


@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product image and return its key attributes as a JSON object.
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

    message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{image_data}"},
            },
            {
                "type": "text",
                "text": (
                    "Look at this product image and extract its key attributes. "
                    "Return ONLY a JSON object with these fields:\n"
                    "- product_type: what kind of product it is\n"
                    "- search_query: a short keyword to search for it\n"
                    "- is_organic: true if the label says organic, false if not, null if unclear\n"
                    "- description: one sentence describing the product\n"
                    "- confidence: a number between 0 and 1 describing how certain you are"
                ),
            },
        ]
    )

    response = vision_llm.invoke([message])
    return response.content


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

bootstrap_data()

agent = create_agent(
    tools=[search_products, get_rating, checkout, describe_product_image, search_similar_products],
    model=llm,
    system_prompt=(
        "You are a premium shopping assistant for a modern ecommerce storefront.\n\n"
        "Goals:\n"
        "- Help users browse products visually and conversationally.\n"
        "- Always surface product IDs so the UI can add items to cart.\n"
        "- Use the tool outputs, especially rating and image data, to make recommendations.\n\n"
        "IMAGE SEARCH:\n"
        "1. First call search_similar_products with the uploaded image path to get true embedding-based matches.\n"
        "2. Then call describe_product_image only if you need a textual explanation of the product in the image.\n"
        "3. Use search_products if you need to broaden or refine results with price, organic, or category filters.\n"
        "4. Recommend the top matches as a concise numbered list and mention similarity or match quality.\n\n"
        "BROWSING:\n"
        "1. Call search_products to find matching items.\n"
        "2. Prefer the returned fields average_rating, review_count, image_path, and relevance_score.\n"
        "3. Present qualifying products as a numbered list in this format:\n\n"
        "   #<number>. <name> (ID:<product_id>) - $<price> - <rating> stars - <organic or non-organic>\n"
        "   <short reason>\n\n"
        "4. Add a short one-line recommendation summary above the list.\n"
        "5. If the assistant is responding to an image search, include the most similar products first and mention that the match is based on visual embeddings.\n"
        "6. If only one product qualifies, still show it and ask whether to add it to cart or order it.\n"
        "7. Do not call checkout unless the user explicitly confirms purchase.\n\n"
        "ORDERING:\n"
        "1. Read the product ID from your prior message.\n"
        "2. Call checkout with that product_id.\n"
        "3. Confirm the order in plain text.\n"
    ),
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "I want organic honey with 4.5+ rating and less than $20 price.",
                }
            ]
        }
    )
    print(result["messages"][-1].content)
