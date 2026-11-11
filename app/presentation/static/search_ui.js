// SearchUI (boundary, Figure 4.3): keyword search bar, suggestions and result list.
export const MSG_INVALID_KEYWORD = "Keyword must be 2-50 characters";

export async function api(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || "Something went wrong, please try again");
  return body;
}

const fmtPrice = (n) => `$${Number(n).toFixed(2)}`;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export class SearchUI {
  constructor() {
    this.form = document.getElementById("search-form");
    this.input = document.getElementById("search-input");
    this.error = document.getElementById("search-error");
    this.suggestions = document.getElementById("suggestions");
    this.status = document.getElementById("status");
    this.results = document.getElementById("results");
    this.productDialog = document.getElementById("product-dialog");
    this.searchSeq = 0;

    this.input.addEventListener("focus", () => this.selectSearchBar());
    this.input.addEventListener("input", () => this.#hideError());
    document.addEventListener("click", (e) => {
      if (!e.target.closest("#search")) this.suggestions.hidden = true;
    });
    this.form.addEventListener("submit", (e) => { e.preventDefault(); this.submitKeyword(this.input.value); });
    this.results.addEventListener("click", (e) => {
      const card = e.target.closest(".card");
      if (card) this.showProduct(card.dataset.id);
    });
    document.getElementById("product-close").addEventListener("click", () => this.productDialog.close());
  }

  async selectSearchBar() {
    const seq = this.searchSeq;
    try {
      const { recent, trending } = await api("/api/suggestions");
      // a search started while loading: do not reopen the dropdown over its results
      if (seq !== this.searchSeq || document.activeElement !== this.input) return;
      this.showSuggestedKeywords(recent, trending);
    } catch { /* suggestions are optional */ }
  }

  showSuggestedKeywords(recent, trending) {
    const group = (title, items) => items.length
      ? `<h3>${title}</h3>${items.map((k) => `<button type="button" data-k="${esc(k)}">${esc(k)}</button>`).join("")}` : "";
    const html = group("Recent searches", recent) + group("Trending", trending);
    this.suggestions.innerHTML = html;
    this.suggestions.hidden = !html;
    this.suggestions.querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", () => { this.input.value = b.dataset.k; this.submitKeyword(b.dataset.k); }));
  }

  validateFormat(keyword) {
    const k = keyword.trim();
    return k.length >= 2 && k.length <= 50;
  }

  async submitKeyword(keyword) {
    this.suggestions.hidden = true;
    if (!this.validateFormat(keyword)) return this.showInvalidKeywordMessage(MSG_INVALID_KEYWORD);
    await this.run(`Searching for “${keyword.trim()}”…`,
      () => api(`/api/search/keyword?q=${encodeURIComponent(keyword.trim())}`),
      `Results for “${keyword.trim()}”`, "No products found");
  }

  // Shared by the three search modes: show progress, then products or the empty message.
  // showSimilarity: image results carry a real cosine similarity worth showing.
  async run(progress, request, title, emptyMessage, showSimilarity = false) {
    const seq = ++this.searchSeq;
    this.onSearchStart?.(); // lets the catalogue hide its browsing controls
    this.suggestions.hidden = true;
    this.status.textContent = progress;
    this.results.innerHTML = "";
    try {
      const { products } = await request();
      if (seq !== this.searchSeq) return; // a newer search has started: drop this answer
      products.length ? this.displayProducts(products, title, showSimilarity) : this.showNoProductsFound(emptyMessage);
    } catch (err) {
      if (seq === this.searchSeq) this.status.textContent = err.message;
    }
  }

  showInvalidKeywordMessage(msg) {
    this.error.textContent = msg;
    this.error.hidden = false;
    this.input.focus();
  }

  #hideError() { this.error.hidden = true; }

  displayProducts(products, title, showSimilarity = false) {
    this.status.textContent = `${title} — ${products.length} product${products.length > 1 ? "s" : ""}`;
    this.results.innerHTML = this.cardsHtml(products, showSimilarity ? "similarity" : "search");
  }

  // badges: "similarity" = real image cosine, "search" = mark semantic matches, "none" = browsing
  cardsHtml(products, badges = "none") {
    const badge = (p) => badges === "similarity"
      ? ` · <span class="badge">${Math.round(p.score * 100)}% similar</span>`
      : badges === "search" && p.score < 1 ? ` · <span class="badge">related</span>` : "";
    return products.map((p) => `
      <button class="card" data-id="${p.id}">
        <img src="${esc(p.image_url)}" alt="${esc(p.name)}" loading="lazy">
        <div class="body">
          <span class="name">${esc(p.name)}</span>
          <span class="meta">${esc((p.category || "").replace(/-/g, " "))}</span>
          <span class="price">${fmtPrice(p.price)}</span>
          <span class="meta">${p.stock > 0 ? `${p.stock} in stock` : "Out of stock"}${badge(p)}</span>
        </div>
      </button>`).join("");
  }

  showNoProductsFound(message = "No products found") {
    this.status.textContent = message;
    this.results.innerHTML = "";
  }

  async showProduct(id) {
    let p;
    try {
      p = await api(`/api/products/${id}`);
    } catch (err) {
      this.status.textContent = err.message;
      return;
    }
    document.getElementById("product-detail").innerHTML = `
      <div class="detail">
        <img src="${esc(p.image_url)}" alt="${esc(p.name)}">
        <div>
          <h2>${esc(p.name)}</h2>
          <p class="meta">${esc((p.category || "").replace(/-/g, " "))}</p>
          <p class="price">${fmtPrice(p.price)} · ${p.stock > 0 ? `${p.stock} in stock` : "Out of stock"}</p>
          <p>${esc(p.description)}</p>
        </div>
      </div>`;
    this.productDialog.showModal();
  }
}
