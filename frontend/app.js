// Millie frontend: Voice over WebSocket, Workspace & Knowledge over REST.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const S = {
  view: "files", // "files" | "research" | "notes" | "timers" | "knowledge" | "chat" | tab_id
  tabs: [],
  notes: [],
  reminders: [],
  stopwatches: [],
  workspace: { files: [] },
  knowledge: [],
  kbCategory: "",
  kbQuery: "",
  fsSubtab: "new", // "preview" | "new" | "convert" | "merge" | "upload"
  selectedFile: null,
  fileFilter: "",
  tabData: null,
  log: [],
  caps: null,
  micOn: false,
  state: "off",
  ws: null,
};

async function api(path, opts = {}) {
  const init = { method: opts.method || "GET", headers: {} };
  if (opts.json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.json);
  }
  if (opts.form) init.body = opts.form;
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* keep statusText */ }
    throw new Error(msg);
  }
  return r.json();
}

// ---------------------------------------------------------------- audio out
let actx = null;
const ensureCtx = () => { actx ||= new AudioContext(); if (actx.state === "suspended") actx.resume(); return actx; };
document.addEventListener("pointerdown", ensureCtx, { once: true });

const queue = [];
let playing = false, current = null;

function send(obj) { if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify(obj)); }

function enqueue(item) { queue.push(item); pump(); }

async function pump() {
  if (playing) return;
  const item = queue.shift();
  if (!item) return;
  playing = true;
  const done = () => { playing = false; current = null; send({ type: "audio_ended" }); pump(); };
  if (item.wav) {
    try {
      const buf = await ensureCtx().decodeAudioData(item.wav.slice(0));
      const src = actx.createBufferSource();
      src.buffer = buf; src.connect(actx.destination); src.onended = done; current = src; src.start();
    } catch { done(); }
  } else {
    const u = new SpeechSynthesisUtterance(item.text);
    u.onend = u.onerror = done; current = "tts"; speechSynthesis.speak(u);
  }
}

function stopAudio() {
  queue.length = 0;
  if (current && current !== "tts") { current.onended = null; try { current.stop(); } catch { /* already stopped */ } }
  speechSynthesis.cancel();
  playing = false; current = null;
}

function beep(times = 1, freq = 880) {
  const c = ensureCtx();
  for (let i = 0; i < times; i++) {
    const o = c.createOscillator(), g = c.createGain();
    o.frequency.value = freq; g.gain.value = 0.08;
    o.connect(g); g.connect(c.destination);
    const t = c.currentTime + i * 0.35;
    o.start(t); o.stop(t + 0.18);
  }
}

// ---------------------------------------------------------------- mic in
const WORKLET = `
class Cap extends AudioWorkletProcessor {
  constructor() { super(); this.ratio = sampleRate / 16000; this.pos = 0; this.acc = 0; this.cnt = 0; this.out = new Int16Array(1280); this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i]; this.cnt++; this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.cnt));
        this.acc = 0; this.cnt = 0;
        this.out[this.n++] = v * 32767;
        if (this.n === 1280) { this.port.postMessage(this.out.buffer.slice(0)); this.n = 0; }
      }
    }
    return true;
  }
}
registerProcessor("cap", Cap);`;

let stream = null, node = null, source = null, level = 0;

async function startMic() {
  stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const c = ensureCtx();
  await c.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" })));
  source = c.createMediaStreamSource(stream);
  node = new AudioWorkletNode(c, "cap");
  node.port.onmessage = (e) => {
    const pcm = new Int16Array(e.data);
    let sum = 0;
    for (let i = 0; i < pcm.length; i += 8) sum += pcm[i] * pcm[i];
    const rms = Math.sqrt(sum / (pcm.length / 8)) / 32768;
    level = Math.min(1, rms * 8);
    $("#orb").style.setProperty("--level", level.toFixed(2));
    if (S.ws && S.ws.readyState === 1) S.ws.send(e.data);
  };
  source.connect(node);
  S.micOn = true;
  setState("idle");
}

function stopMic() {
  if (node) { node.disconnect(); node.port.onmessage = null; }
  if (source) source.disconnect();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  node = source = stream = null;
  S.micOn = false;
  $("#orb").style.setProperty("--level", 0);
  setState("off");
}

// ---------------------------------------------------------------- voice UI
function setState(state) {
  S.state = (state === "idle" && !S.micOn) ? "off" : state;
  const wake = S.caps?.wake_name === "hey_jarvis" ? "Hey Jarvis" : "Hey Millie";
  const labels = {
    off: S.ws && S.ws.readyState === 1 ? "Mic is off" : "Connecting…",
    idle: S.caps?.wake ? `Say “${wake}”` : "Press Talk to speak",
    listening: "Listening…", thinking: "Thinking…", speaking: "Speaking",
  };
  $("#orb").dataset.state = S.state;
  $("#state-label").textContent = labels[S.state] || S.state;
  $("#btn-talk").disabled = !S.micOn;
  $("#btn-stop").disabled = S.state === "off" || S.state === "idle";
  $("#btn-mic").textContent = S.micOn ? "Mute mic" : "Start listening";
  $("#btn-mic").classList.toggle("primary", !S.micOn);
}

function say(role, text) {
  S.log.push({ role, text });
  if (S.log.length > 80) S.log.shift();
  $("#heard").textContent = (role === "user" ? "You: " : "Millie: ") + text;
  if (S.view === "chat") renderLog();
}

function renderLatency(m) {
  const el = $("#latency");
  if (!m) { el.innerHTML = '<span>Latency of your last request will appear here.</span>'; return; }
  const v = (x) => (x == null ? "–" : `${x} ms`);
  const parts = [m.stt_ms, m.llm_first_token_ms, m.tts_first_byte_ms].map((x) => Math.max(0, x || 0));
  const total = parts.reduce((a, b) => a + b, 0) || 1;
  el.innerHTML = `
    <span class="stat"><b>${v(m.to_first_audio_ms)}</b>to first audio</span>
    <span class="bar" role="img" aria-label="Latency breakdown">${parts.map((p) => `<i style="width:${(p / total) * 100}%"></i>`).join("")}</span>
    <span class="stat"><span class="key" style="background:var(--accent)"></span>STT <b>${v(m.stt_ms)}</b></span>
    <span class="stat"><span class="key" style="background:var(--amber)"></span>LLM 1st <b>${v(m.llm_first_token_ms)}</b></span>
    <span class="stat"><span class="key" style="background:var(--ok)"></span>TTS 1st <b>${v(m.tts_first_byte_ms)}</b></span>
    <span class="stat">wait <b>${v(m.endpoint_ms)}</b></span>`;
}

function toast(html, kind = "", sticky = false) {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.innerHTML = `<div class="grow">${html}</div><button class="ghost" aria-label="Dismiss">✕</button>`;
  t.querySelector("button").onclick = () => t.remove();
  $("#toasts").append(t);
  if (!sticky) setTimeout(() => t.remove(), 7000);
}

// ---------------------------------------------------------------- WebSocket
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  S.ws = ws;
  ws.onopen = () => setState(S.state === "off" ? "off" : "idle");
  ws.onclose = () => { setState("off"); setTimeout(connect, 2000); };
  ws.onmessage = (e) => {
    if (typeof e.data !== "string") { enqueue({ wav: e.data }); return; }
    const m = JSON.parse(e.data);
    switch (m.type) {
      case "capabilities": S.caps = m; showSystem(); setState(S.state); break;
      case "state": setState(m.state); break;
      case "wake": beep(1, 988); break;
      case "transcript": say(m.role, m.text); break;
      case "speak": enqueue({ text: m.text }); break;
      case "stop_audio": stopAudio(); break;
      case "latency": renderLatency(m); break;
      case "error": toast(`<b>Error</b><br>${esc(m.message)}`, "alarm", true); setState("idle"); break;
      case "reminder_fired": onReminder(m); break;
      case "research_done": toast(`Research finished in <b>${esc(m.name)}</b>`); break;
      case "changed": onChanged(m); break;
    }
  };
}

function onReminder(m) {
  const alarm = m.kind === "alarm";
  toast(`<b>${alarm ? "Alarm" : "Reminder"}</b><br>${esc(m.text)}`, alarm ? "alarm" : "", true);
  beep(alarm ? 6 : 2, alarm ? 740 : 880);
  if (!alarm) enqueue({ text: `Reminder: ${m.text}` });
}

async function onChanged(m) {
  if (["tabs", "notes", "reminders", "stopwatches", "workspace", "knowledge"].includes(m.what)) {
    await reloadData();
  }
  if (m.what === "tabs" && m.open && m.tab_id) {
    await go(m.tab_id);
    return;
  }
  if (typeof S.view === "number" && m.tab_id === S.view) {
    await loadTab(S.view);
  }
}

// ---------------------------------------------------------------- Data Loading
async function reloadData() {
  try {
    const d = await api("/api/dashboard");
    S.tabs = d.tabs || [];
    S.notes = d.notes || [];
    S.reminders = d.reminders || [];
    S.stopwatches = d.stopwatches || [];
    S.workspace = d.workspace || { files: [] };
  } catch (e) {
    console.error("Dashboard reload failed", e);
  }
  try {
    const kRes = await api(S.kbQuery ? `/api/knowledge?query=${encodeURIComponent(S.kbQuery)}` : `/api/knowledge${S.kbCategory ? `?category=${encodeURIComponent(S.kbCategory)}` : ""}`);
    S.knowledge = kRes.items || [];
  } catch {
    S.knowledge = [];
  }
  updateBadges();
  renderNav();
  renderCurrentView();
}

function updateBadges() {
  const filesCount = (S.workspace.files || []).filter((f) => !f.is_dir).length;
  $("#badge-files").textContent = filesCount;
  $("#rail-cnt-files").textContent = filesCount;

  $("#badge-tabs").textContent = S.tabs.length;
  $("#rail-cnt-tabs").textContent = S.tabs.length;

  $("#badge-notes").textContent = S.notes.length;
  $("#rail-cnt-notes").textContent = S.notes.length;

  const timersCount = S.reminders.filter((r) => !r.fired).length + S.stopwatches.filter((s) => !s.stopped_at).length;
  $("#badge-timers").textContent = timersCount;
  $("#rail-cnt-timers").textContent = timersCount;

  $("#badge-knowledge").textContent = S.knowledge.length;
  $("#rail-cnt-knowledge").textContent = S.knowledge.length;
}

function fmtDue(ts) {
  return new Date(ts * 1000).toLocaleString([], { weekday: "short", day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });
}

function fmtDur(sec) {
  sec = Math.max(0, Math.floor(sec));
  const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (d) return `${d}d ${h}h`;
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}:${String(s).padStart(2, "0")}`;
}

// ---------------------------------------------------------------- Navigation
async function go(view) {
  S.view = view;
  $$(".view-tab").forEach((t) => t.setAttribute("aria-selected", t.dataset.nav === String(view)));
  $$(".railnav .navitem").forEach((t) => t.setAttribute("aria-current", t.dataset.nav === String(view)));
  renderNav();
  if (typeof view === "number") {
    buildTab();
    await loadTab(view);
  } else {
    renderCurrentView();
  }
}

function renderNav() {
  $("#tablist").innerHTML = S.tabs.map((t) => `
    <button class="navitem" data-tab="${t.id}" aria-current="${S.view === t.id}">
      <span class="dot ${esc(t.status)}"></span><span>${esc(t.name)}</span>
      ${t.links.length ? `<small>${t.links.length} linked</small>` : ""}
    </button>`).join("") || '<div class="empty" style="color:var(--rail-muted);padding:4px 8px">No research tabs yet.</div>';
}

function renderCurrentView() {
  if (typeof S.view === "number") return;
  switch (S.view) {
    case "files": renderFilesView(); break;
    case "knowledge": renderKnowledgeView(); break;
    case "research": renderResearchOverview(); break;
    case "notes": renderNotesView(); break;
    case "timers": renderTimersView(); break;
    case "chat": renderChatView(); break;
    default: renderFilesView();
  }
}

// ---------------------------------------------------------------- 1. FILES & SUITE VIEW
const downloadUrl = (rel) => `/api/workspace/download/${rel.split("/").map(encodeURIComponent).join("/")}`;

function visibleFiles() {
  const allFiles = (S.workspace.files || []).filter((f) => !f.is_dir);
  const filter = S.fileFilter.toLowerCase().trim();
  const visible = filter ? allFiles.filter((f) => f.name.toLowerCase().includes(filter) || f.rel_path.toLowerCase().includes(filter)) : allFiles;
  return { allFiles, visible };
}

function fileRow(f) {
  let badgeClass = "code";
  if (f.ext === ".docx") badgeClass = "docx";
  else if (f.ext === ".pdf") badgeClass = "pdf";
  else if (f.ext === ".md" || f.ext === ".txt") badgeClass = "md";
  else if ([".png", ".jpg", ".jpeg", ".webp"].includes(f.ext)) badgeClass = "img";
  const isSel = S.selectedFile && S.selectedFile.rel_path === f.rel_path;
  return `
    <div class="file-row ${isSel ? "active" : ""}" data-file-select="${esc(f.rel_path)}">
      <span class="file-badge ${badgeClass}">${esc(f.ext ? f.ext.slice(1) : "FILE")}</span>
      <div class="name" title="${esc(f.rel_path)}">${esc(f.rel_path)}</div>
      <span class="meta">${esc(f.size_human)}</span>
      <div class="actions">
        <button class="ghost" data-file-dl="${esc(f.rel_path)}" title="Download file">↓</button>
        <button class="ghost danger" data-file-del="${esc(f.rel_path)}" title="Delete file">✕</button>
      </div>
    </div>`;
}

function renderFilesView() {
  const { allFiles, visible } = visibleFiles();

  const fileRows = visible.map(fileRow).join("") || `<div class="empty">No workspace files found. Create or upload one!</div>`;

  const sub = S.fsSubtab;

  $("#view").innerHTML = `
    <div class="fs-layout">
      <!-- Left: File Tree / List -->
      <section class="card">
        <header>
          <h3>Workspace Files (${allFiles.length})</h3>
          <button class="ghost" id="fs-refresh-btn" title="Refresh files">↻</button>
        </header>
        <input id="fs-search" placeholder="Search files…" value="${esc(S.fileFilter)}" style="margin-bottom:8px">
        <div class="fs-tree-list">${fileRows}</div>
      </section>

      <!-- Right: Action Center & Suite -->
      <section class="card file-preview-card">
        <div class="subtabs">
          <button class="subtab ${sub === "preview" ? "active" : ""}" data-fstab="preview">File Preview</button>
          <button class="subtab ${sub === "new" ? "active" : ""}" data-fstab="new">+ New File (.docx, .md, .txt)</button>
          <button class="subtab ${sub === "convert" ? "active" : ""}" data-fstab="convert">⇄ Convert</button>
          <button class="subtab ${sub === "merge" ? "active" : ""}" data-fstab="merge">☍ Merge</button>
          <button class="subtab ${sub === "upload" ? "active" : ""}" data-fstab="upload">↑ Upload</button>
        </div>
        <div id="fs-subpanel">
          ${renderFsSubpanel(allFiles)}
        </div>
      </section>
    </div>`;
}

function renderFsSubpanel(allFiles) {
  const sub = S.fsSubtab;
  if (sub === "preview") {
    if (!S.selectedFile) {
      return `<div class="empty" style="padding:40px 20px;text-align:center">Select a file on the left to preview its content and metadata, or create a new file.</div>`;
    }
    const f = S.selectedFile;
    return `
      <div class="file-preview-header">
        <h4 style="margin:0;font-size:16px;">${esc(f.name)}</h4>
        <span class="tag">${esc(f.kind || f.ext)}</span>
        <span class="muted">${(f.size_bytes / 1024).toFixed(1)} KB</span>
        <div style="margin-left:auto;display:flex;gap:8px;">
          <a class="button ghost" href="${downloadUrl(f.rel_path)}" download style="text-decoration:none;display:inline-block;padding:6px 12px;border:1px solid var(--line);border-radius:8px;">Download</a>
          <button class="ghost danger" data-file-del="${esc(f.rel_path)}">Delete</button>
        </div>
      </div>
      <pre class="preview-content">${esc(f.content || "[Binary or Empty File]")}</pre>`;
  }

  if (sub === "new") {
    return `
      <form id="fs-new-file-form" style="display:flex;flex-direction:column;gap:10px;">
        <div>
          <label style="display:block;font-size:13px;font-weight:600;margin-bottom:4px;">File Path / Name (in data/workspace):</label>
          <input id="fs-new-path" placeholder="e.g. summary.docx, report.md, analysis.txt, data.json" required>
        </div>
        <div>
          <label style="display:block;font-size:13px;font-weight:600;margin-bottom:4px;">File Content:</label>
          <textarea id="fs-new-content" style="min-height:220px;font-family:var(--mono);font-size:13px;" placeholder="Write or paste content here...\n\nTip for Word (.docx): Use # Heading, - Bullet points, and | Table | columns. They are styled directly into Word format!"></textarea>
        </div>
        <div style="display:flex;gap:10px;justify-content:flex-end;">
          <button type="button" class="ghost" id="fs-ask-millie-btn">Ask Millie to Fill</button>
          <button type="submit" class="primary">Create / Save File</button>
        </div>
      </form>`;
  }

  if (sub === "convert") {
    const fileOpts = allFiles.map((f) => `<option value="${esc(f.rel_path)}">${esc(f.rel_path)} (${f.size_human})</option>`).join("");
    return `
      <form id="fs-convert-form" style="display:flex;flex-direction:column;gap:12px;">
        <div>
          <label style="display:block;font-size:13px;font-weight:600;margin-bottom:4px;">Select Source File:</label>
          <select id="fs-convert-src" required>${fileOpts || '<option value="">No files available</option>'}</select>
        </div>
        <div>
          <label style="display:block;font-size:13px;font-weight:600;margin-bottom:4px;">Target Format:</label>
          <select id="fs-convert-ext" required>
            <option value=".docx">Word Document (.docx)</option>
            <option value=".md">Markdown (.md)</option>
            <option value=".txt">Plain Text (.txt)</option>
            <option value=".html">Styled HTML (.html)</option>
            <option value=".pdf">PDF Document (.pdf)</option>
          </select>
        </div>
        <div>
          <label style="display:block;font-size:13px;font-weight:600;margin-bottom:4px;">Custom Output Name (optional):</label>
          <input id="fs-convert-out" placeholder="e.g. converted_report.docx">
        </div>
        <button type="submit" class="primary" style="margin-top:6px;">Convert File</button>
      </form>`;
  }

  if (sub === "merge") {
    const checklist = allFiles.map((f) => `
      <label>
        <input type="checkbox" name="merge_file" value="${esc(f.rel_path)}">
        <span>${esc(f.rel_path)} <small class="muted">(${f.size_human})</small></span>
      </label>`).join("");

    return `
      <form id="fs-merge-form" style="display:flex;flex-direction:column;gap:12px;">
        <label style="display:block;font-size:13px;font-weight:600;margin-bottom:2px;">Select Files to Merge:</label>
        <div class="merge-checklist">${checklist || '<div class="empty">No files available to merge.</div>'}</div>
        <div>
          <label style="display:block;font-size:13px;font-weight:600;margin-bottom:4px;">Merged Output Filename:</label>
          <input id="fs-merge-out" placeholder="e.g. combined.pdf or master_report.docx" required>
        </div>
        <button type="submit" class="primary">Merge Documents</button>
      </form>`;
  }

  if (sub === "upload") {
    return `
      <div id="fs-dropzone" style="display:flex;flex-direction:column;gap:14px;padding:24px;text-align:center;border:2px dashed var(--line);border-radius:12px;cursor:pointer;">
        <div style="font-size:32px;">📁</div>
        <div>
          <h4 style="margin:0 0 6px;">Upload Any Files to Workspace</h4>
          <p class="muted" style="margin:0;">Drag & drop files here or click to choose from your Mac.</p>
        </div>
        <input type="file" id="fs-upload-input" multiple style="display:none">
        <div>
          <button type="button" class="primary" id="fs-choose-files">Choose Files from Mac</button>
        </div>
      </div>`;
  }
}

// ---------------------------------------------------------------- 2. KNOWLEDGE BASE VIEW
function renderKnowledgeView() {
  const cats = ["", "preference", "project", "personal", "work", "general"];
  const catPills = cats.map((c) => `
    <button class="kb-cat-pill ${S.kbCategory === c ? "active" : ""}" data-kb-cat="${esc(c)}">
      ${c ? c.toUpperCase() : "ALL"}
    </button>`).join("");

  const cards = S.knowledge.map((k) => `
    <div class="kb-card">
      <div class="kb-card-header">
        <h4>${esc(k.title)}</h4>
        <span class="tag">${esc(k.category || "general")}</span>
        <button class="ghost danger" data-del-kb="${k.id}" style="margin-left:auto;">✕</button>
      </div>
      <p>${esc(k.content)}</p>
      ${(k.tags && k.tags.length) ? `<div class="kb-tags">${k.tags.map((t) => `<span class="kb-tag">#${esc(t)}</span>`).join("")}</div>` : ""}
      <small class="muted" style="margin-top:auto;">Added ${new Date(k.created_at * 1000).toLocaleDateString()}</small>
    </div>`).join("") || `<div class="empty" style="grid-column:1/-1;">No knowledge items found. Speak to Millie or add one below!</div>`;

  $("#view").innerHTML = `
    <div class="kb-toolbar">
      <div style="flex:1;min-width:240px;">
        <input id="kb-search-input" placeholder="Search knowledge base (FTS full-text search)…" value="${esc(S.kbQuery)}">
      </div>
      <div class="kb-cats">${catPills}</div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 340px;gap:18px;align-items:start;">
      <div class="kb-grid">${cards}</div>
      <section class="card">
        <header><h3>+ Add Knowledge</h3></header>
        <p class="muted" style="font-size:12.5px;margin:0 0 10px;">Millie learns facts and preferences automatically from voice conversations, or you can record them here.</p>
        <form id="kb-add-form" style="display:flex;flex-direction:column;gap:10px;">
          <div><label style="font-size:12px;font-weight:600;">Title / Concept:</label><input id="kb-new-title" placeholder="e.g. Favorite Editor or Project MedGemma" required></div>
          <div>
            <label style="font-size:12px;font-weight:600;">Category:</label>
            <select id="kb-new-cat">
              <option value="preference">Preference</option>
              <option value="project">Project</option>
              <option value="personal">Personal</option>
              <option value="work">Work</option>
              <option value="general" selected>General</option>
            </select>
          </div>
          <div><label style="font-size:12px;font-weight:600;">Tags (comma separated):</label><input id="kb-new-tags" placeholder="python, vscode, mac"></div>
          <div><label style="font-size:12px;font-weight:600;">Details / Content:</label><textarea id="kb-new-content" style="min-height:90px;" placeholder="What should Millie remember?" required></textarea></div>
          <button type="submit" class="primary">Save Fact</button>
        </form>
      </section>
    </div>`;
}

// ---------------------------------------------------------------- 3. RESEARCH OVERVIEW VIEW
function renderResearchOverview() {
  const tabs = S.tabs.map((t) => `
    <div class="card" style="cursor:pointer;" data-tab="${t.id}">
      <header>
        <h4 style="margin:0;font-size:16px;">${esc(t.name)}</h4>
        <span class="tag ${t.status}">${esc(t.status)}</span>
      </header>
      <div style="font-size:13px;color:var(--muted);margin-top:6px;">
        ${t.links.length ? `Linked to: ${t.links.length} tabs` : "Not linked to other tabs"}
      </div>
      <div style="margin-top:12px;display:flex;justify-content:flex-end;">
        <button class="primary" data-tab="${t.id}">Open Tab</button>
      </div>
    </div>`).join("") || '<div class="empty">No research tabs yet. Create one on the left or above!</div>';

  $("#view").innerHTML = `
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
      <h2 style="font-size:22px;">Research Tabs (${S.tabs.length})</h2>
    </div>
    <div class="grid">${tabs}</div>`;
}

// ---------------------------------------------------------------- 4. NOTES VIEW
function renderNotesView() {
  $("#view").innerHTML = `
    <div class="card" style="margin-bottom:18px;">
      <header><h3>+ Quick Note</h3></header>
      <form id="note-form" class="inline" autocomplete="off">
        <input id="note-title-input" placeholder="Title (optional)" style="flex:0 0 200px;">
        <input id="note-input" placeholder="Write a note or ask Millie to take one…" aria-label="Write a note" required>
        <button class="primary">Save note</button>
      </form>
    </div>
    <div class="grid" id="notes-grid">
      ${S.notes.map((n) => `
        <div class="card">
          <header>
            <h4 style="margin:0;font-size:15px;">${esc(n.title || "Note")}</h4>
            <button class="ghost danger" data-del-note="${n.id}">✕</button>
          </header>
          <p style="white-space:pre-wrap;margin:8px 0;font-size:14px;">${esc(n.content)}</p>
          <small class="muted">${new Date(n.created_at * 1000).toLocaleString([], { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" })}</small>
        </div>`).join("") || '<div class="empty">No notes yet. Say “Take a note…” or type above.</div>'}
    </div>`;
}

// ---------------------------------------------------------------- 5. TIMERS & ALARMS VIEW
function renderTimersView() {
  const rem = S.reminders.map((r) => `
    <div class="item ${r.fired ? "fired" : ""}">
      <span class="tag ${r.kind === "alarm" ? "alarm" : ""}">${r.kind === "alarm" ? "Alarm" : "Reminder"}</span>
      <div class="grow">${esc(r.text)}<br><small>${fmtDue(r.due_at)}${r.fired ? " · fired" : ` · rings in <b data-due="${r.due_at}"></b>`}</small></div>
      <button class="ghost danger" data-del-reminder="${r.id}" aria-label="Delete">✕</button>
    </div>`).join("") || '<div class="empty">Nothing scheduled. Say “Set a 10 second timer”.</div>';

  const sws = S.stopwatches.map((s) => `
    <div class="item">
      <div class="grow">${esc(s.label)}<br>
        <span class="mono" style="font-size:18px;font-weight:700;" data-start="${s.started_at}" ${s.stopped_at ? `data-stop="${s.stopped_at}"` : ""}></span>
      </div>
      ${s.stopped_at ? "" : `<button data-stop-sw="${s.id}">Stop</button>`}
      <button class="ghost danger" data-del-sw="${s.id}" aria-label="Delete stopwatch">✕</button>
    </div>`).join("") || '<div class="empty">No stopwatches running. Say “Start a stopwatch”.</div>';

  $("#view").innerHTML = `
    <div class="grid">
      <section class="card">
        <header><h3>Reminders & Alarms</h3></header>
        <div id="reminders-list">${rem}</div>
      </section>

      <section class="card">
        <header><h3>Stopwatches</h3></header>
        <div id="sw-list">${sws}</div>
        <form id="sw-form" class="inline" autocomplete="off" style="margin-top:12px;">
          <input id="sw-input" placeholder="Stopwatch label (optional)" aria-label="Stopwatch label">
          <button class="primary">Start</button>
        </form>
      </section>
    </div>`;
}

// ---------------------------------------------------------------- 6. CHAT / ACTIVITY VIEW
function renderChatView() {
  $("#view").innerHTML = `
    <div class="card">
      <header><h3>Live Activity & Conversation History</h3></header>
      <div id="log" class="log" style="max-height:550px;overflow-y:auto;"></div>
    </div>`;
  renderLog();
}

function renderLog() {
  const el = $("#log");
  if (!el) return;
  el.innerHTML = S.log.slice().reverse().map((l) => `<div class="${l.role === "user" ? "you" : "millie"}">${esc(l.text)}</div>`).join("")
    || '<div class="empty">Your conversation history with Millie will appear here.</div>';
}

// ---------------------------------------------------------------- Research Tab View (individual tab)
function buildTab() {
  $("#view").innerHTML = `
    <div class="tabhead">
      <h2 id="t-name"></h2><span id="t-status" class="tag"></span>
      <button class="danger ghost" id="t-delete" style="margin-left:auto">Delete tab</button>
    </div>
    <div class="linksrow">
      <span>Uses findings from:</span><span id="t-links"></span>
      <select id="t-linksel" aria-label="Link another tab"></select>
      <button id="t-linkbtn">Link</button>
    </div>
    <div class="runbox">
      <textarea id="t-instr" placeholder="What should this tab research? For example: survey literature on local TTS and wakewords."></textarea>
      <div class="row"><button class="primary" id="t-run">Start research</button></div>
    </div>
    <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(340px,1fr))">
      <section class="card"><header><h3>Findings</h3></header><div id="t-findings"></div></section>
      <section class="card"><header><h3>Activity</h3></header><div id="t-activity" class="activity"></div></section>
    </div>
    <section class="card" style="margin-top:16px">
      <header><h3>Files and images</h3></header>
      <div id="t-files"></div>
      <div class="dropzone">
        <input type="file" id="t-file" multiple aria-label="Upload files">
        <label class="check"><input type="checkbox" id="t-ocr" checked> Read text in images (OCR)</label>
      </div>
      <div class="muted" style="margin-top:6px">PDF, Word (.docx), PowerPoint and text files are read automatically.</div>
    </section>`;
}

async function loadTab(id) {
  try {
    S.tabData = await api(`/api/tabs/${id}`);
  } catch {
    return go("files");
  }
  if (S.view !== id) return;
  renderTabDynamic();
}

function renderTabDynamic() {
  const d = S.tabData;
  if (!d || !$("#t-name")) return;
  $("#t-name").textContent = d.tab.name;
  const st = $("#t-status");
  st.textContent = d.running ? "researching" : d.tab.status;
  st.className = `tag ${d.tab.status}`;
  $("#t-links").innerHTML = d.links.map((l) => `<button class="chip" data-tab="${l.id}" title="Open ${esc(l.name)}">${esc(l.name)}</button>`).join("") || "nothing yet";
  const linked = new Set([d.tab.id, ...d.links.map((l) => l.id)]);
  $("#t-linksel").innerHTML = S.tabs.filter((t) => !linked.has(t.id)).map((t) => `<option value="${t.id}">${esc(t.name)}</option>`).join("") || '<option value="">No other tabs</option>';
  $("#t-run").disabled = d.running;
  $("#t-run").textContent = d.running ? "Researching…" : "Start research";

  $("#t-findings").innerHTML = d.insights.slice().reverse().map((i) => `
    <div class="insight ${i.kind === "summary" ? "summary" : ""}">
      <h4>${esc(i.title)}</h4>
      <p>${esc(i.content)}</p>
      ${i.sources.length ? `<div class="src">${i.sources.map((u) => `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(u)}</a>`).join("<br>")}</div>` : ""}
    </div>`).join("") || '<div class="empty">Findings appear here as the agent saves them.</div>';

  const act = $("#t-activity");
  const stick = act.scrollTop + act.clientHeight >= act.scrollHeight - 30;
  act.innerHTML = d.messages.map((m) => `<div class="m ${esc(m.role)}">${esc(m.content)}</div>`).join("") || '<div class="empty">No activity yet.</div>';
  if (stick) act.scrollTop = act.scrollHeight;

  $("#t-files").innerHTML = d.files.map((f) => `
    <div class="item">
      <span class="tag">${esc(f.kind)}</span>
      <div class="grow">${esc(f.filename)}<br>
        <small>${f.extracted_text.length ? `${f.extracted_text.length.toLocaleString()} chars extracted` : "no text found"}</small>
      </div>
    </div>`).join("") || '<div class="empty">No files attached to this tab.</div>';
}

function showSystem() {
  const c = S.caps || {};
  $("#sysinfo").innerHTML = `Wake: ${c.wake ? esc(c.wake_name) : "not installed"}<br>VAD: ${esc(c.vad || "?")}<br>TTS: ${esc(c.tts || "?")}`;
}

// ---------------------------------------------------------------- Global Event Listeners
document.addEventListener("click", async (e) => {
  const btn = e.target.closest("button") || e.target.closest("[data-nav]") || e.target.closest("[data-tab]") || e.target.closest("[data-file-select]");
  if (!btn) return;

  try {
    // 1. Navigation clicks
    if (btn.dataset.nav) {
      return go(btn.dataset.nav);
    }
    if (btn.dataset.tab) {
      return go(Number(btn.dataset.tab));
    }

    // 2. Quick app launcher clicks
    if (btn.dataset.openApp) {
      const app = btn.dataset.openApp;
      await api("/api/apps/open", { method: "POST", json: { app_name: app } });
      toast(`Launched <b>${esc(app)}</b>`);
      return;
    }

    // 2b. Spotify quick controls
    if (btn.dataset.spotifyAction) {
      const act = btn.dataset.spotifyAction;
      await api("/api/spotify/control", { method: "POST", json: { action: act } });
      toast(`Spotify: <b>${esc(act)}</b>`);
      return;
    }


    // 3. File Suite Subtabs
    if (btn.dataset.fstab) {
      S.fsSubtab = btn.dataset.fstab;
      renderFilesView();
      return;
    }

    // 4. File selection
    if (btn.dataset.fileSelect) {
      const p = btn.dataset.fileSelect;
      const data = await api(`/api/workspace/files/content?path=${encodeURIComponent(p)}`);
      S.selectedFile = data;
      S.fsSubtab = "preview";
      renderFilesView();
      return;
    }

    // 4b. File download (↓ button in the file list)
    if (btn.dataset.fileDl) {
      location.href = downloadUrl(btn.dataset.fileDl);
      return;
    }
    if (btn.id === "fs-choose-files") {
      $("#fs-upload-input").click();
      return;
    }

    // 5. File delete
    if (btn.dataset.fileDel) {
      const p = btn.dataset.fileDel;
      if (confirm(`Delete file '${p}' from workspace?`)) {
        await api(`/api/workspace/files?path=${encodeURIComponent(p)}`, { method: "DELETE" });
        if (S.selectedFile && S.selectedFile.rel_path === p) S.selectedFile = null;
        toast(`Deleted file <b>${esc(p)}</b>`);
      }
      return;
    }

    // 6. Refresh files
    if (btn.id === "fs-refresh-btn") {
      await reloadData();
      toast("Workspace refreshed");
      return;
    }

    // 7. Ask Millie to fill file
    if (btn.id === "fs-ask-millie-btn") {
      const path = $("#fs-new-path")?.value.trim() || "document.docx";
      const prompt = `Please create and fill the workspace file '${path}' with comprehensive content.`;
      $("#text-input").value = prompt;
      $("#text-input").focus();
      toast("Prompt copied to chat input. Hit Send or speak to Millie!");
      return;
    }

    // 8. Knowledge Category filter
    if (btn.dataset.kbCat !== undefined) {
      S.kbCategory = btn.dataset.kbCat;
      await reloadData();
      return;
    }

    // 9. Knowledge delete
    if (btn.dataset.delKb) {
      await api(`/api/knowledge/${btn.dataset.delKb}`, { method: "DELETE" });
      toast("Fact removed from Knowledge Base");
      return;
    }

    // 10. Notes / Alarms / Stopwatches delete & stop
    if (btn.dataset.delReminder) {
      await api(`/api/reminders/${btn.dataset.delReminder}`, { method: "DELETE" });
    } else if (btn.dataset.delNote) {
      await api(`/api/notes/${btn.dataset.delNote}`, { method: "DELETE" });
    } else if (btn.dataset.stopSw) {
      await api(`/api/stopwatches/${btn.dataset.stopSw}/stop`, { method: "POST" });
    } else if (btn.dataset.delSw) {
      await api(`/api/stopwatches/${btn.dataset.delSw}`, { method: "DELETE" });
    }

    // 11. Research tab actions
    if (btn.id === "t-run") {
      const instr = $("#t-instr").value.trim();
      if (!instr) { toast("Tell this tab what to research first."); return; }
      await api(`/api/tabs/${S.view}/run`, { method: "POST", json: { instructions: instr } });
      $("#t-instr").value = "";
      await loadTab(S.view);
    } else if (btn.id === "t-linkbtn") {
      const v = $("#t-linksel").value;
      if (!v) return;
      await api(`/api/tabs/${S.view}/link`, { method: "POST", json: { linked_tab_id: Number(v) } });
    } else if (btn.id === "t-delete") {
      if (confirm(`Delete “${S.tabData.tab.name}” and everything in it?`)) {
        await api(`/api/tabs/${S.view}`, { method: "DELETE" });
        await go("research");
      }
    }
  } catch (err) {
    toast(esc(err.message), "alarm");
  }
});

// File input handlers
document.addEventListener("change", async (e) => {
  // 1. Research tab file upload
  if (e.target.id === "t-file") {
    const input = e.target, ocr = $("#t-ocr").checked;
    for (const f of input.files) {
      const form = new FormData();
      form.append("file", f);
      form.append("ocr", ocr);
      try {
        await api(`/api/tabs/${S.view}/upload`, { method: "POST", form });
        toast(`Added ${esc(f.name)}`);
      } catch (err) {
        toast(`Failed to add ${esc(f.name)}: ${esc(err.message)}`, "alarm");
      }
    }
    input.value = "";
    return;
  }

  // 2. Workspace general file upload
  if (e.target.id === "fs-upload-input") {
    const files = e.target.files;
    for (const f of files) {
      const form = new FormData();
      form.append("file", f);
      try {
        const res = await api("/api/workspace/upload", { method: "POST", form });
        toast(`Uploaded <b>${esc(res.rel_path)}</b>`);
      } catch (err) {
        toast(`Upload failed for ${esc(f.name)}: ${esc(err.message)}`, "alarm");
      }
    }
    e.target.value = "";
    S.fsSubtab = "preview";
    await reloadData();
  }
});

// Drag and drop handler for workspace files
document.addEventListener("dragover", (e) => {
  const dz = e.target.closest("#fs-dropzone");
  if (dz) {
    e.preventDefault();
    dz.style.borderColor = "var(--primary)";
    dz.style.background = "var(--line)";
  }
});

document.addEventListener("dragleave", (e) => {
  const dz = e.target.closest("#fs-dropzone");
  if (dz) {
    dz.style.borderColor = "var(--line)";
    dz.style.background = "";
  }
});

document.addEventListener("drop", async (e) => {
  const dz = e.target.closest("#fs-dropzone");
  if (!dz) return;
  e.preventDefault();
  dz.style.borderColor = "var(--line)";
  dz.style.background = "";
  const files = e.dataTransfer?.files;
  if (!files || !files.length) return;
  for (const f of files) {
    const form = new FormData();
    form.append("file", f);
    try {
      const res = await api("/api/workspace/upload", { method: "POST", form });
      toast(`Uploaded <b>${esc(res.rel_path)}</b>`);
    } catch (err) {
      toast(`Upload failed for ${esc(f.name)}: ${esc(err.message)}`, "alarm");
    }
  }
  S.fsSubtab = "preview";
  await reloadData();
});

// Search input handlers
document.addEventListener("input", (e) => {
  if (e.target.id === "fs-search") {
    S.fileFilter = e.target.value;
    const tree = $(".fs-tree-list");
    if (tree) tree.innerHTML = visibleFiles().visible.map(fileRow).join("") || `<div class="empty">No matching files.</div>`;
  }

  if (e.target.id === "kb-search-input") {
    S.kbQuery = e.target.value;
    clearTimeout(window._kbTimeout);
    window._kbTimeout = setTimeout(async () => {
      await reloadData();
    }, 250);
  }
});

// Form submit handlers
document.addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    // 1. New research tab
    if (e.target.id === "newtab-form") {
      const name = $("#newtab-name").value.trim();
      if (!name) return;
      const tab = await api("/api/tabs", { method: "POST", json: { name } });
      $("#newtab-name").value = "";
      await reloadData();
      await go(tab.id);
      return;
    }

    // 2. Chat / typed voice input
    if (e.target.id === "text-form") {
      const v = $("#text-input").value.trim();
      if (!v) return;
      ensureCtx();
      send({ type: "text", text: v });
      $("#text-input").value = "";
      return;
    }

    // 3. New workspace file
    if (e.target.id === "fs-new-file-form") {
      const path = $("#fs-new-path").value.trim();
      const content = $("#fs-new-content").value;
      if (!path) return;
      await api("/api/workspace/files", { method: "POST", json: { path, content } });
      toast(`Created file <b>${esc(path)}</b>`);
      const fileData = await api(`/api/workspace/files/content?path=${encodeURIComponent(path)}`);
      S.selectedFile = fileData;
      S.fsSubtab = "preview";
      await reloadData();
      return;
    }

    // 4. Convert file
    if (e.target.id === "fs-convert-form") {
      const src = $("#fs-convert-src").value;
      const ext = $("#fs-convert-ext").value;
      const out = $("#fs-convert-out").value.trim();
      if (!src) return;
      const res = await api("/api/workspace/convert", { method: "POST", json: { src_path: src, target_ext: ext, out_name: out } });
      toast(`Converted to <b>${esc(res.rel_path)}</b>`);
      const fileData = await api(`/api/workspace/files/content?path=${encodeURIComponent(res.rel_path)}`);
      S.selectedFile = fileData;
      S.fsSubtab = "preview";
      await reloadData();
      return;
    }

    // 5. Merge files
    if (e.target.id === "fs-merge-form") {
      const checked = $$('input[name="merge_file"]:checked').map((c) => c.value);
      const out = $("#fs-merge-out").value.trim();
      if (checked.length < 2) {
        toast("Select at least 2 files to merge.", "alarm");
        return;
      }
      if (!out) {
        toast("Please provide an output filename.", "alarm");
        return;
      }
      const res = await api("/api/workspace/merge", { method: "POST", json: { src_paths: checked, out_name: out } });
      toast(`Merged into <b>${esc(res.rel_path)}</b>`);
      const fileData = await api(`/api/workspace/files/content?path=${encodeURIComponent(res.rel_path)}`);
      S.selectedFile = fileData;
      S.fsSubtab = "preview";
      await reloadData();
      return;
    }

    // 6. Add knowledge
    if (e.target.id === "kb-add-form") {
      const title = $("#kb-new-title").value.trim();
      const category = $("#kb-new-cat").value;
      const rawTags = $("#kb-new-tags").value.split(",").map((t) => t.trim()).filter(Boolean);
      const content = $("#kb-new-content").value.trim();
      if (!title || !content) return;
      await api("/api/knowledge", { method: "POST", json: { title, content, category, tags: rawTags } });
      toast(`Added <b>${esc(title)}</b> to Knowledge Base`);
      $("#kb-new-title").value = "";
      $("#kb-new-tags").value = "";
      $("#kb-new-content").value = "";
      await reloadData();
      return;
    }

    // 7. Add Note
    if (e.target.id === "note-form") {
      const content = $("#note-input").value.trim();
      const title = $("#note-title-input")?.value.trim() || "";
      if (!content) return;
      await api("/api/notes", { method: "POST", json: { content, title } });
      $("#note-input").value = "";
      if ($("#note-title-input")) $("#note-title-input").value = "";
      await reloadData();
      return;
    }

    // 8. Start Stopwatch
    if (e.target.id === "sw-form") {
      const label = $("#sw-input").value.trim() || "Stopwatch";
      await api("/api/stopwatches", { method: "POST", json: { label } });
      $("#sw-input").value = "";
      await reloadData();
      return;
    }
  } catch (err) {
    toast(esc(err.message), "alarm");
  }
});

// Voice buttons
$("#btn-mic").onclick = async () => {
  try {
    S.micOn ? stopMic() : await startMic();
  } catch (err) {
    toast(`Microphone unavailable: ${esc(err.message)}. Allow mic access.`, "alarm", true);
  }
};
$("#btn-talk").onclick = () => { ensureCtx(); send({ type: "ptt" }); };
$("#btn-stop").onclick = () => { stopAudio(); send({ type: "stop" }); };

// Live clocks: stopwatches and reminder countdowns
setInterval(() => {
  const now = Date.now() / 1000;
  document.querySelectorAll("[data-start]").forEach((el) => {
    el.textContent = fmtDur((el.dataset.stop ? Number(el.dataset.stop) : now) - Number(el.dataset.start));
  });
  document.querySelectorAll("[data-due]").forEach((el) => {
    el.textContent = fmtDur(Number(el.dataset.due) - now);
  });
}, 500);

// ---------------------------------------------------------------- Boot
(async function boot() {
  setState("off");
  renderLatency(null);
  connect();
  try {
    const h = await api("/api/health");
    const msgs = [];
    if (!h.llm_configured) msgs.push("No LLM key found in <code>.env</code>. Set <code>LLM_API_KEY</code> (see <code>.env.example</code>).");
    if (h.wake === false) msgs.push("Wake word model is not installed; use Push-to-Talk or type.");
    $("#banner").innerHTML = msgs.map((m) => `<div class="banner">${m}</div>`).join("");
  } catch { /* server starting */ }

  await reloadData();
  await go("files");
})();
