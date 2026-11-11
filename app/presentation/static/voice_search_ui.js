// VoiceSearchUI (boundary, Figure 4.3): record, transcribe, confirm, then search.
import { PERMISSION_DENIED } from "./permission_manager.js";
import { api } from "./search_ui.js";

export class VoiceSearchUI {
  constructor(searchUI, permissions) {
    this.searchUI = searchUI;
    this.permissions = permissions;
    this.dialog = document.getElementById("voice-dialog");
    this.el = (id) => document.getElementById(id);
    this.el("mic-btn").addEventListener("click", () => this.selectMicrophone());
    this.el("voice-stop").addEventListener("click", () => this.stopRecording());
    this.el("voice-again").addEventListener("click", () => this.speakAgain());
    this.el("voice-confirm").addEventListener("click", () => this.confirmText());
    this.el("voice-cancel").addEventListener("click", () => this.#close());
    this.el("voice-text").addEventListener("keydown", (e) => { if (e.key === "Enter") this.confirmText(); });
    this.dialog.addEventListener("close", () => this.#stopStream());
  }

  async selectMicrophone() {
    if ((await this.permissions.checkMicPermission()) !== "denied") {
      this.stream = await this.permissions.requestPermission("microphone");
    }
    if (!this.stream) return this.#showBlocked();
    this.dialog.showModal();
    this.speakSearchQuery();
  }

  // Microphone blocked: explain how to allow it instead of failing silently.
  #showBlocked() {
    this.#view({ error: `${PERMISSION_DENIED} To use voice search, click the lock icon left of the address bar, set Microphone to Allow, then reload the page.` });
    this.el("voice-again").hidden = true;
    this.dialog.showModal();
  }

  // Starts recording; the customer speaks the query (speakSearchQuery).
  speakSearchQuery() {
    this.showRecordingIndicator();
    this.chunks = [];
    this.recorder = new MediaRecorder(this.stream);
    this.recorder.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.recorder.onstop = () => this.#transcribe();
    this.recorder.start();
  }

  showRecordingIndicator() {
    this.#view({ recording: true });
  }

  stopRecording() {
    if (this.recorder?.state === "recording") {
      this.el("voice-stop").disabled = true;
      this.el("voice-recording").querySelector("span:last-child").textContent = "Recognizing speech…";
      this.recorder.stop();
    }
  }

  async #transcribe() {
    const blob = new Blob(this.chunks, { type: this.recorder.mimeType || "audio/webm" });
    const form = new FormData();
    form.append("audio", blob, "query.webm");
    try {
      const { audio_id, text, language } = await api("/api/voice/transcribe", { method: "POST", body: form });
      this.audioId = audio_id;
      this.displayRecognizedText(text, language);
    } catch (err) {
      this.showRecognitionError(err.message);
    }
  }

  displayRecognizedText(text, language) {
    this.#view({ review: true });
    this.el("voice-text").value = text;
    this.el("voice-lang").textContent = language ? `Detected language: ${language}` : "";
    this.el("voice-text").focus();
  }

  showRecognitionError(msg) {
    this.#view({ error: msg });
  }

  speakAgain() { this.speakSearchQuery(); }

  async confirmText() {
    const text = this.el("voice-text").value;
    if (!this.searchUI.validateFormat(text)) return this.#view({ review: true, error: "Keyword must be 2-50 characters" });
    this.#close();
    this.searchUI.input.value = text.trim();
    await this.submitQuery(text.trim());
  }

  submitQuery(text) {
    return this.searchUI.run(`Searching for “${text}”…`,
      () => api("/api/search/voice", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, audio_id: this.audioId }) }),
      `Voice results for “${text}”`, "No products found");
  }

  #view({ recording = false, review = false, error = "" }) {
    this.el("voice-recording").hidden = !recording;
    this.el("voice-recording").querySelector("span:last-child").textContent = "Listening… speak your search, then press Stop.";
    this.el("voice-review").hidden = !review;
    this.el("voice-error").hidden = !error;
    this.el("voice-error").textContent = error;
    this.el("voice-stop").hidden = !recording;
    this.el("voice-stop").disabled = false;
    this.el("voice-confirm").hidden = !review;
    this.el("voice-again").hidden = recording;
  }

  #stopStream() {
    if (this.recorder?.state === "recording") { this.recorder.onstop = null; this.recorder.stop(); }
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }

  #close() { this.dialog.close(); }
}
