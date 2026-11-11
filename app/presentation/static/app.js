import { CatalogueUI } from "./catalogue_ui.js";
import { ImageSearchUI } from "./image_search_ui.js";
import { PermissionManager } from "./permission_manager.js";
import { SearchUI } from "./search_ui.js";
import { VoiceSearchUI } from "./voice_search_ui.js";

const searchUI = new SearchUI();
const permissions = new PermissionManager();
new VoiceSearchUI(searchUI, permissions);
new ImageSearchUI(searchUI, permissions);
new CatalogueUI(searchUI).init();
