// PermissionManager (control, Figure 4.3): microphone and camera permission.
export class PermissionManager {
  async #query(name) {
    try {
      return (await navigator.permissions.query({ name })).state; // granted | denied | prompt
    } catch {
      return "prompt"; // browser cannot query this permission: ask when used
    }
  }

  checkMicPermission() { return this.#query("microphone"); }

  checkCameraPermission() { return this.#query("camera"); }

  // requestPermission(): shows the browser's permission dialog (showPermissionDialog)
  // and returns a live stream, or null when the customer denies it.
  async requestPermission(kind) {
    if (!navigator.mediaDevices?.getUserMedia) return null;
    try {
      return await navigator.mediaDevices.getUserMedia(kind === "camera" ? { video: true } : { audio: true });
    } catch {
      return null;
    }
  }
}

export const PERMISSION_DENIED = "Permission denied. Please type a keyword instead.";
