// ImageSearchUI (boundary, Figure 4.3): upload, preview, mark the target object, search.
import { PERMISSION_DENIED } from "./permission_manager.js";
import { api } from "./search_ui.js";

const MSG_INVALID_FILE = "Only JPG/PNG, max 10MB";
const MAX_BYTES = 10 * 1024 * 1024;

export class ImageSearchUI {
  constructor(searchUI, permissions) {
    this.searchUI = searchUI;
    this.permissions = permissions;
    this.el = (id) => document.getElementById(id);
    this.dialog = this.el("image-dialog");
    this.canvas = this.el("image-canvas");
    this.ctx = this.canvas.getContext("2d");

    this.el("camera-btn").addEventListener("click", () => this.selectCameraIcon());
    this.el("image-file").addEventListener("change", (e) => e.target.files[0] && this.submitImage(e.target.files[0]));
    this.el("image-cancel").addEventListener("click", () => this.dialog.close());
    this.el("image-confirm").addEventListener("click", () => this.confirmSelection());
    this.el("image-camera").addEventListener("click", () => this.#openCamera());
    this.el("camera-capture").addEventListener("click", () => this.#capture());
    this.dialog.addEventListener("close", () => this.#stopCamera());
    const zone = this.dialog.querySelector(".dropzone");
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("over"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault(); zone.classList.remove("over");
      if (e.dataTransfer.files[0]) this.submitImage(e.dataTransfer.files[0]);
    });
    this.#bindDrawing();
  }

  selectCameraIcon() { this.showUploadPopup(); }

  showUploadPopup() {
    this.el("image-file").value = "";
    this.#view("upload");
    this.dialog.showModal();
  }

  async #openCamera() {
    if ((await this.permissions.checkCameraPermission()) === "denied") return this.showFileError(PERMISSION_DENIED);
    this.stream = await this.permissions.requestPermission("camera");
    if (!this.stream) return this.showFileError(PERMISSION_DENIED);
    const video = this.el("camera-video");
    video.srcObject = this.stream;
    video.hidden = false;
    this.el("camera-capture").hidden = false;
  }

  #capture() {
    const video = this.el("camera-video");
    const c = document.createElement("canvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext("2d").drawImage(video, 0, 0);
    c.toBlob((blob) => this.submitImage(new File([blob], "camera.jpg", { type: "image/jpeg" })), "image/jpeg", 0.92);
    this.#stopCamera();
  }

  #stopCamera() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.el("camera-video").hidden = true;
    this.el("camera-capture").hidden = true;
  }

  validateFileFormat(file) {
    return ["image/jpeg", "image/png"].includes(file.type) && file.size <= MAX_BYTES;
  }

  async submitImage(file) {
    if (!this.validateFileFormat(file)) return this.showFileError(MSG_INVALID_FILE);
    this.#view("upload", "Analyzing image…");
    const form = new FormData();
    form.append("image", file);
    try {
      const res = await api("/api/image/analyze", { method: "POST", body: form });
      this.imageId = res.image_id;
      this.objects = res.objects;
      this.box = null;
      await this.previewImage(file, res.width, res.height);
    } catch (err) {
      this.showFileError(err.message);
    }
  }

  // Shows the image in the selection area with the detected objects outlined.
  async previewImage(file, width, height) {
    this.img = await createImageBitmap(file);
    this.canvas.width = width;
    this.canvas.height = height;
    this.#view("select");
    this.el("object-chips").innerHTML = this.objects.map((o, i) =>
      `<button type="button" data-i="${i}">${o.label} ${(o.confidence * 100).toFixed(0)}%</button>`).join("")
      + `<button type="button" data-i="-1" class="selected">Whole image</button>`;
    this.el("object-chips").querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", () => this.markTargetObject(Number(b.dataset.i))));
    this.markTargetObject(this.objects.length ? 0 : -1);
  }

  markTargetObject(index, customBox = null) {
    this.box = customBox ?? (index >= 0 ? this.objects[index].box : null);
    this.el("object-chips").querySelectorAll("button").forEach((b) =>
      b.classList.toggle("selected", !customBox && Number(b.dataset.i) === index));
    this.#draw();
  }

  #draw(preview = null) {
    const { ctx, canvas } = this;
    ctx.drawImage(this.img, 0, 0, canvas.width, canvas.height);
    const lw = Math.max(2, canvas.width / 300);
    this.objects.forEach((o) => this.#rect(o.box, "rgba(255,255,255,.9)", lw));
    if (this.box) {
      ctx.fillStyle = "rgba(0,0,0,.35)";
      const [x1, y1, x2, y2] = this.box;
      ctx.fillRect(0, 0, canvas.width, y1); ctx.fillRect(0, y2, canvas.width, canvas.height - y2);
      ctx.fillRect(0, y1, x1, y2 - y1); ctx.fillRect(x2, y1, canvas.width - x2, y2 - y1);
      this.#rect(this.box, "#2563eb", lw * 1.5);
    }
    if (preview) this.#rect(preview, "#2563eb", lw);
  }

  #rect([x1, y1, x2, y2], color, width) {
    this.ctx.strokeStyle = color; this.ctx.lineWidth = width;
    this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }

  #bindDrawing() {
    const point = (e) => {
      const r = this.canvas.getBoundingClientRect();
      return [(e.clientX - r.left) * this.canvas.width / r.width, (e.clientY - r.top) * this.canvas.height / r.height];
    };
    const norm = ([ax, ay], [bx, by]) => [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)].map(Math.round);
    this.canvas.addEventListener("pointerdown", (e) => { this.start = point(e); this.canvas.setPointerCapture(e.pointerId); });
    this.canvas.addEventListener("pointermove", (e) => { if (this.start) this.#draw(norm(this.start, point(e))); });
    this.canvas.addEventListener("pointerup", (e) => {
      if (!this.start) return;
      const box = norm(this.start, point(e));
      this.start = null;
      const small = box[2] - box[0] < 8 || box[3] - box[1] < 8;
      if (!small) return this.markTargetObject(-2, box);
      // a click: select the smallest detected object under the pointer
      const [x, y] = point(e);
      const hits = this.objects.map((o, i) => [o, i]).filter(([o]) => x >= o.box[0] && x <= o.box[2] && y >= o.box[1] && y <= o.box[3]);
      hits.sort(([a], [b]) => (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]) - (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]));
      this.markTargetObject(hits.length ? hits[0][1] : -1);
    });
  }

  async confirmSelection() {
    this.dialog.close();
    await this.searchSimilar();
  }

  searchSimilar() {
    return this.searchUI.run("Searching for similar products…",
      () => api("/api/search/image", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_id: this.imageId, box: this.box }) }),
      "Visually similar products", "No matching products", true);
  }

  showFileError(msg) {
    this.el("image-error").textContent = msg;
    this.el("image-error").hidden = false;
  }

  #view(step, info = "") {
    this.el("image-upload").hidden = step !== "upload";
    this.el("image-select").hidden = step !== "select";
    this.el("image-confirm").hidden = step !== "select";
    this.el("image-error").hidden = !info;
    this.el("image-error").textContent = info;
  }
}
