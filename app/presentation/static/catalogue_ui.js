// Home page: browse the catalogue by category before searching.
import { api } from "./search_ui.js";

const PAGE = 24;
const label = (name) => name.replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

export class CatalogueUI {
  constructor(searchUI) {
    this.searchUI = searchUI;
    this.chips = document.getElementById("categories");
    this.more = document.getElementById("load-more");
    this.category = null;
    this.loaded = 0;
    this.seq = 0;

    searchUI.onSearchStart = () => this.#setBrowsing(false);
    this.more.addEventListener("click", () => this.#load(true));
    this.chips.addEventListener("click", (e) => {
      const chip = e.target.closest("button");
      if (chip) this.show(chip.dataset.category || null);
    });
    // clearing the search box (its x button or Esc) returns to the catalogue
    searchUI.input.addEventListener("search", () => { if (!searchUI.input.value) this.show(this.category); });
  }

  async init() {
    try {
      const cats = await api("/api/categories");
      const total = cats.reduce((n, c) => n + c.count, 0);
      this.chips.innerHTML = `<button type="button" data-category="">All (${total})</button>`
        + cats.map((c) => `<button type="button" data-category="${c.name}">${label(c.name)} (${c.count})</button>`).join("");
    } catch { /* categories are optional; the grid still loads */ }
    await this.show(null);
  }

  async show(category) {
    this.category = category;
    this.chips.querySelectorAll("button").forEach((b) =>
      b.classList.toggle("selected", (b.dataset.category || null) === category));
    await this.#load(false);
  }

  async #load(append) {
    const seq = ++this.seq;
    this.searchUI.searchSeq++; // any search still loading must not overwrite the catalogue
    this.#setBrowsing(true);
    const offset = append ? this.loaded : 0;
    const query = new URLSearchParams({ offset, limit: PAGE });
    if (this.category) query.set("category", this.category);
    try {
      const { products, total } = await api(`/api/products?${query}`);
      if (seq !== this.seq) return;
      this.loaded = offset + products.length;
      const html = this.searchUI.cardsHtml(products, "none");
      if (append) this.searchUI.results.insertAdjacentHTML("beforeend", html);
      else this.searchUI.results.innerHTML = html;
      this.searchUI.status.textContent =
        `${this.category ? label(this.category) : "All products"} — showing ${this.loaded} of ${total}`;
      this.more.hidden = this.loaded >= total;
    } catch (err) {
      this.searchUI.status.textContent = err.message;
    }
  }

  #setBrowsing(on) {
    this.chips.hidden = !on;
    if (!on) this.more.hidden = true;
  }
}
