/* 前端逻辑（原生 JS，无构建步骤） */
const API = window.API || "/api/v1";
const $ = (s) => document.querySelector(s);

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2500);
}

const esc = (s) => (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(path, options = {}) {
  const res = await fetch(API + path, {
    headers: options.body && !(options.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
    ...options,
  });
  if (!res.ok) {
    let msg = `请求失败 ${res.status}`;
    try { msg = (await res.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

const collection = () => $("#collection").value || "default";

/* ---------------- 问答 ---------------- */
function addMsg(role, text) {
  const box = $("#messages");
  const tip = box.querySelector(".tip");
  if (tip) tip.remove();
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  box.appendChild(wrap);
  box.scrollTop = box.scrollHeight;
  return wrap;
}

function renderSources(wrap, sources) {
  if (!sources || !sources.length) return;
  const box = document.createElement("div");
  box.className = "sources";
  sources.forEach((s) => {
    const item = document.createElement("div");
    item.className = "source";
    const label = [s.document_name, s.page ? `第${s.page}页` : "", s.source].filter(Boolean).join(" · ");
    item.innerHTML = `<b>[${s.index}]</b> ${esc(label)}<br>${esc(s.snippet)}`;
    box.appendChild(item);
  });
  wrap.querySelector(".bubble").appendChild(box);
}

async function send() {
  const input = $("#question");
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  addMsg("user", q);
  const wrap = addMsg("assistant", "");
  const bubble = wrap.querySelector(".bubble");
  let acc = "";
  try {
    const res = await fetch(`${API}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, collection: collection() }),
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop() || "";
      for (const p of parts) {
        if (!p.startsWith("data:")) continue;
        const evt = JSON.parse(p.slice(5).trim());
        if (evt.type === "sources") renderSources(wrap, evt.sources);
        if (evt.type === "token") {
          acc += evt.content;
          bubble.textContent = acc;
          $("#messages").scrollTop = $("#messages").scrollHeight;
        }
        if (evt.type === "error") throw new Error(evt.error);
      }
    }
  } catch (e) {
    bubble.textContent = "请求失败：" + e.message;
  }
}

/* ---------------- 知识库 ---------------- */
async function loadDocs() {
  try {
    const docs = await api(`/documents`);
    $("#docRows").innerHTML = docs
      .map(
        (d) => `<tr>
          <td>${esc(d.name)}</td>
          <td>${d.doc_type}</td>
          <td><span class="badge ${d.status}">${d.status}</span></td>
          <td>${d.chunk_count}</td>
          <td><button class="btn ghost" data-id="${d.id}">删除</button></td>
        </tr>`
      )
      .join("");
  } catch (e) {
    toast("加载文档失败：" + e.message);
  }
}

async function upload(files) {
  for (const f of files) {
    const item = document.createElement("div");
    item.className = "queue-item";
    item.innerHTML = `<span>${esc(f.name)}</span><span class="muted">上传中…</span>`;
    $("#queue").appendChild(item);
    const fd = new FormData();
    fd.append("file", f);
    fd.append("collection", collection());
    try {
      await api("/documents/upload", { method: "POST", body: fd });
      item.querySelector(".muted").textContent = "已提交，后台索引中";
      setTimeout(loadDocs, 3000);
      setTimeout(loadDocs, 10000);
    } catch (e) {
      item.querySelector(".muted").textContent = "失败：" + e.message;
    }
  }
}

/* ---------------- 评测 ---------------- */
let evalTimer = null;
async function runEval() {
  $("#runEval").disabled = true;
  try {
    const data = await api("/eval/run", {
      method: "POST",
      body: JSON.stringify({
        collection: collection(),
        size: Number($("#size").value) || 20,
        use_ragas: $("#useRagas").checked,
      }),
    });
    toast("评测已启动");
    clearInterval(evalTimer);
    evalTimer = setInterval(async () => {
      const run = await api(`/eval/runs/${data.run_id}`);
      renderEval(run);
      if (run.status !== "running") {
        clearInterval(evalTimer);
        $("#runEval").disabled = false;
      }
    }, 3000);
  } catch (e) {
    toast("评测失败：" + e.message);
    $("#runEval").disabled = false;
  }
}

function renderEval(run) {
  const kv = Object.entries(run.metrics || {})
    .map(([k, v]) => `<div class="k">${esc(k)}</div><div>${esc(String(v))}</div>`)
    .join("");
  $("#evalResult").innerHTML = `<div class="card-head"><h2>评测 ${run.status}</h2></div>
    <div class="kv">${kv || `<div class="k">状态</div><div>${run.status}</div>`}</div>
    ${run.error ? `<div class="muted">${esc(run.error)}</div>` : ""}`;
}

/* ---------------- 事件 ---------------- */
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    ["chat", "docs", "eval"].forEach((v) => $("#view-" + v).classList.toggle("hidden", v !== t.dataset.view));
    if (t.dataset.view === "docs") loadDocs();
  })
);

$("#chatForm").addEventListener("submit", (e) => { e.preventDefault(); send(); });
$("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } });
$("#pick").addEventListener("click", () => $("#file").click());
$("#file").addEventListener("change", (e) => upload(Array.from(e.target.files)));
const dz = $("#dropzone");
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("drag"); upload(Array.from(e.dataTransfer.files)); });
$("#refresh").addEventListener("click", loadDocs);
$("#docRows").addEventListener("click", async (e) => {
  if (e.target.tagName !== "BUTTON") return;
  if (!confirm("确认删除该文档及其分片？")) return;
  try {
    await api(`/documents/${e.target.dataset.id}`, { method: "DELETE" });
    toast("已删除");
    loadDocs();
  } catch (err) { toast("删除失败：" + err.message); }
});
$("#runEval").addEventListener("click", runEval);
loadDocs();
