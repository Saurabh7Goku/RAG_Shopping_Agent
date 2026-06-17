# Shopping Agent Project Report

## 1. Current State

This project is a solid prototype for an AI shopping assistant, but it is still more of a chat-driven demo than a full shopping experience.

What already works:
- Text-based product search
- Price and organic filters
- Product rating lookup from review data
- Image upload flow that identifies a product and searches the catalog
- Simple order creation in SQLite

Project snapshot:
- 32 products in the local catalog
- 102 review rows
- 2 sample orders
- 9 product categories
- 16 organic products

## 2. Main Gaps

### A. Users cannot browse the catalog visually
The app does not currently show a product listing page or product cards. Product discovery depends on the chat assistant returning text.

Relevant code:
- `C:\Saurabh\Projects\shopping_agent\app.py`
- `C:\Saurabh\Projects\shopping_agent\shopping_agent.py`

### B. Image search returns text only
The current image flow:
- uploads the image
- runs a vision model to describe it
- converts the description into a text query
- returns matching products as text

There is no product image associated with each catalog item, so the assistant cannot render image-based search results or similar product visuals.

Relevant code:
- `C:\Saurabh\Projects\shopping_agent\shopping_agent.py`
- `C:\Saurabh\Projects\shopping_agent\setup_db.py`

### C. Data model is too small for an online shopping experience
The `products` table only stores:
- name
- category
- price
- description
- organic flag

Missing fields that would make the app feel much more real:
- product image URL or file path
- stock / availability
- brand
- discount / offer price
- product tags
- dimensions / variants / color / size
- shipping ETA
- popularity / sales rank

### D. UI is minimal and not storefront-like
The Streamlit app only has:
- a chat box
- an image uploader in the sidebar

There is no:
- homepage hero section
- category navigation
- product grid
- filters panel
- cart preview
- recommendations section
- recently viewed items

## 3. Why It Feels Limited Today

The biggest reason is that the product experience is search-first and text-first.

The assistant can answer a query like:
- "I want organic honey under $15 with 4+ rating"

But the user cannot easily:
- see all available products
- compare items visually
- explore categories
- click a product card
- view product images alongside ratings and price

That makes the app feel like an AI chatbot with a product database, not yet like a shopping platform.

## 4. Recommended Product Plan

### Phase 1: Make the catalog visible
Goal: users should immediately see what is available.

Build:
- Product grid in the main page
- Category filters
- Sort by price, rating, and popularity
- Search bar above the catalog
- Product cards with image, name, price, rating, and organic badge

Data changes:
- add `image_path` or `image_url` to `products`
- add `in_stock` or `availability`

### Phase 2: Upgrade image search
Goal: uploaded image should return visually matched products, not only text.

Build:
- show the uploaded image preview
- return top 3 matched products as cards
- include the matched product image
- show "why this matched" explanation
- optionally support visual similarity scoring

Data/model changes:
- link every product to an image
- add embeddings for product images
- store similarity metadata or vector index

### Phase 3: Add recommendation intelligence
Goal: make the app feel like an AI shopping assistant, not a basic search tool.

Build:
- personalized recommendations
- similar items and frequently bought together
- "best value", "top rated", and "budget pick" labels
- conversational refinement such as "show me cheaper ones"

AI/ML upgrades:
- ranking model for search results
- hybrid retrieval using text + image similarity
- product embedding search
- session-based recommendation logic

### Phase 4: Add shopping flow depth
Goal: make it behave more like a real online store.

Build:
- cart
- wishlist
- product detail pages
- checkout history
- order tracking placeholder
- review summaries and sentiment highlights

### Phase 5: Polish for resume impact
Goal: make the project clearly stand out in interviews.

Build:
- evaluation dashboard for search quality
- latency monitoring
- model comparison page
- prompt/version tracking
- fallback handling and confidence scores
- analytics for clicks, adds-to-cart, and orders

## 5. Feature Ideas That Stand Out

These features would make the project stronger and more memorable:

1. Visual product search
- Upload an image and get visually similar product cards
- Show the reference image side-by-side with matched products

2. Hybrid retrieval
- Combine keyword search, embeddings, price filters, and rating signals
- This is a strong AI/ML story for a resume

3. Smart ranking
- Rank by relevance, rating, organic preference, budget, and conversion likelihood

4. Explainable recommendations
- Example: "Recommended because it is organic, under budget, and highly rated"

5. Product comparison mode
- Compare 2-4 products by price, rating, reviews, category, and attributes

6. Conversational filters
- Users can say "show cheaper options" or "only 5-star items"

7. Personalization
- Remember preferences such as organic, budget range, favorite categories

8. Inventory-aware shopping
- Hide out-of-stock products and highlight low stock

9. Realistic checkout flow
- Cart, shipping estimate, order confirmation, and order history

10. Search analytics
- Track queries, click-through rate, and order conversions

## 6. AI/ML Resume Value

To make this project impressive on a resume, emphasize:

- Multimodal retrieval: text + image search
- Ranking pipeline: filtering, scoring, and re-ranking
- Vision-language workflow: product image understanding
- Recommendation logic: similar products and top picks
- Structured tool use: search, rating, checkout orchestration
- Product analytics: conversion and query monitoring

Good resume framing:
- "Built a multimodal AI shopping assistant with image-based product discovery, structured product retrieval, and review-aware ranking."
- "Designed a conversational shopping workflow that combines vision models, retrieval, and recommendation logic."
- "Extended a lightweight e-commerce prototype into a catalog-driven assistant with ranking, filtering, and order simulation."

## 7. Suggested Technical Roadmap

### Short term
- add a visible product catalog
- add product images to the database
- render product cards in Streamlit
- show image-search results as cards

### Mid term
- add similarity search over product embeddings
- add cart and wishlist
- add comparison and recommendation widgets

### Long term
- add personalization
- add analytics dashboards
- add evaluation harness for ranking quality
- package it as a polished portfolio project

## 8. Bottom Line

The project has a good foundation, but right now it feels like a chatbot attached to a small database.

The fastest path to a much stronger product is:
- expose the catalog visually
- attach images to products
- return image search results as cards
- add ranking and recommendation intelligence

That combination will make the app feel much more like a real shopping platform and much more valuable as an AI/ML portfolio project.
