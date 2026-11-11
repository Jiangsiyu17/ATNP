(() => {
  const $ = (id) => document.getElementById(id);
  const els = {
    shell: $("qa-shell"), welcome: $("qa-welcome"), conversation: $("qa-conversation"),
    turns: $("qa-history-turns"), question: $("qa-question"), input: $("qa-input"),
    form: $("qa-form"), progress: $("qa-progress"), progressText: $("qa-progress-text"), progressList: $("qa-progress-list"),
    answer: $("qa-answer"), body: $("qa-answer-body"), meta: $("qa-meta"), notice: $("qa-notice"),
    refs: $("qa-references"), refCount: $("qa-reference-count"), refList: $("qa-reference-list"),
    error: $("qa-error"), errorText: $("qa-error-text"), retry: $("qa-retry"),
    send: $("qa-send"), model: $("qa-model"), modelNote: $("qa-model-note"), history: $("qa-history"), empty: $("qa-empty"),
    dot: $("qa-status-dot"), status: $("qa-status")
  };
  const KEY = "atnp.database.expertQa.v1";
  let conversations = load(), active = null, controller = null, requestMeta = null, progressEvents = [], lastQuestion = "", answerText = "";

  function load() { try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch (_) { return []; } }
  function save() { localStorage.setItem(KEY, JSON.stringify(conversations.slice(0, 12))); }
  function id() { return `conv_${Date.now()}_${Math.random().toString(16).slice(2)}`; }
  function esc(value) { const d = document.createElement("div"); d.textContent = value || ""; return d.innerHTML; }
  function renderHistory() {
    els.history.replaceChildren(); els.empty.hidden = conversations.length > 0;
    conversations.forEach((c) => { const b = document.createElement("button"); b.textContent = c.title; b.className = c.id === active ? "active" : ""; b.onclick = () => open(c.id); els.history.append(b); });
  }
  function renderText(value) {
    const fragment = document.createDocumentFragment();
    String(value || "").replace(/\r/g, "").split("\n").forEach((line) => {
      if (!line.trim()) return;
      const heading = line.match(/^#{1,3}\s+(.+)/); const list = line.match(/^[-*•]\s+(.+)/);
      const node = document.createElement(heading ? "h3" : list ? "li" : "p");
      const text = (heading || list || [null, line])[1];
      const parts = text.split(/(\[R\d+\])/g);
      parts.forEach((part) => { const match = part.match(/^\[R(\d+)\]$/); if (!match) node.append(document.createTextNode(part)); else { const a = document.createElement("button"); a.className = "qa-citation"; a.textContent = part; a.onclick = () => { els.refs.open = true; document.getElementById(`qa-ref-${match[1]}`)?.scrollIntoView({ behavior: "smooth" }); }; node.append(a); } });
      if (list) { let ul = fragment.lastElementChild; if (!ul || ul.tagName !== "UL") { ul = document.createElement("ul"); fragment.append(ul); } ul.append(node); } else fragment.append(node);
    });
    return fragment;
  }
  function renderReferences(refs) {
    els.refList.replaceChildren(); els.refCount.textContent = refs.length; els.refs.hidden = !refs.length;
    refs.forEach((r) => { const box = document.createElement("div"); box.className = "qa-ref"; box.id = `qa-ref-${String(r.id || "").replace(/^R/, "")}`; const title = r.url ? `<a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title || r.paper_id || "未命名文献")}</a>` : esc(r.title || r.paper_id || "未命名文献"); box.innerHTML = `<b>[${esc(String(r.id || "").replace(/^R/, ""))}]</b> ${title}<small>${r.doi ? `DOI ${esc(r.doi)}` : r.pmid ? `PMID ${esc(r.pmid)}` : esc(r.paper_id || "")}</small>`; els.refList.append(box); });
  }
  function renderProgress() {
    els.progressList.replaceChildren();
    progressEvents.forEach((event) => {
      const row = document.createElement("div"); row.className = `qa-progress-item ${event.state || "running"}`;
      const mark = document.createElement("span"); mark.className = "qa-progress-mark"; mark.textContent = event.state === "completed" ? "✓" : event.state === "error" ? "!" : "";
      const copy = document.createElement("div"); const message = document.createElement("span"); message.textContent = event.message || event.stage || "处理中"; copy.append(message);
      const details = event.details || {}; const detail = [details.intent && `问题类型：${details.intent}`, details.model && `模型：${details.model}`, Array.isArray(details.keywords) && details.keywords.length && `关键词：${details.keywords.slice(0, 4).join("、")}`].filter(Boolean).join(" · ");
      if (detail) { const small = document.createElement("small"); small.textContent = detail; copy.append(small); }
      row.append(mark, copy); els.progressList.append(row);
    });
  }
  function updateProgress(event) {
    const i = progressEvents.findIndex((item) => item.stage === event.stage);
    if (i >= 0) progressEvents[i] = { ...progressEvents[i], ...event }; else progressEvents.push(event);
    els.progressText.textContent = event.message || "正在检索知识库…"; renderProgress();
  }
  function completeProgress(result) {
    if (Array.isArray(result?._progress) && result._progress.length) progressEvents = result._progress;
    progressEvents = progressEvents.map((item) => ({ ...item, state: item.state === "running" ? "completed" : item.state }));
    if (!progressEvents.length) { els.progress.hidden = true; return; }
    els.progress.classList.add("completed"); els.progressText.textContent = "检索与生成完成"; renderProgress();
  }
  function showResult(question, result) {
    completeProgress(result); els.progress.hidden = false; els.answer.hidden = false; els.error.hidden = true; answerText = result.answer || result.message || "";
    els.body.replaceChildren(); els.body.append(renderText(answerText)); renderReferences(Array.isArray(result.references) ? result.references : []);
    els.meta.textContent = [result.compound, result.references?.length ? `${result.references.length} 篇文献` : "", Number.isFinite(result.elapsed_ms) ? `${(result.elapsed_ms / 1000).toFixed(1)} 秒` : ""].filter(Boolean).join(" · ");
    const notice = result.message || (result.status === "needs_clarification" ? "请补充明确的天然产物名称。" : ""); els.notice.textContent = notice; els.notice.hidden = !notice;
    els.question.textContent = question; els.welcome.hidden = true; els.conversation.hidden = false;
  }
  function open(cid) { const c = conversations.find((x) => x.id === cid); if (!c) return; active = cid; const turn = c.turns.at(-1); els.turns.replaceChildren(); c.turns.slice(0, -1).forEach((t) => { const p = document.createElement("p"); p.className = "qa-question"; p.innerHTML = `<small>你的问题</small><br>${esc(t.question)}`; els.turns.append(p); }); if (turn) showResult(turn.question, turn.result); renderHistory(); }
  function sse(response, onEvent) { const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ""; return (async () => { while (true) { const { value, done } = await reader.read(); buffer += decoder.decode(value || new Uint8Array(), { stream: !done }); const blocks = buffer.split("\n\n"); buffer = blocks.pop() || ""; blocks.forEach((block) => { let event = "message", data = ""; block.split("\n").forEach((line) => { if (line.startsWith("event:")) event = line.slice(6).trim(); if (line.startsWith("data:")) data += line.slice(5).trim(); }); if (data) { try { onEvent(event, JSON.parse(data)); } catch (_) {} } }); if (done) break; } })(); }
  async function ask(question) {
    question = question.trim(); if (!question || controller) return; lastQuestion = question; if (!active) { active = id(); conversations.unshift({ id: active, title: question, turns: [] }); }
    els.welcome.hidden = true; els.conversation.hidden = false; els.question.textContent = question; els.answer.hidden = true; els.error.hidden = true; els.progress.hidden = false; els.progress.classList.remove("completed"); progressEvents = []; renderProgress(); els.progressText.textContent = "正在连接问答服务…"; els.input.value = ""; els.input.style.height = "auto"; els.send.disabled = true; controller = new AbortController();
    try {
      const result = await window.atnpSseClient.ask({
        question, conversationId: active, model: els.model.value || null, signal: controller.signal,
        onMeta: (meta) => { requestMeta = meta; },
        onProgress: (progress) => { updateProgress(progress); }
      });
      conversations.find((c) => c.id === active).turns.push({ question, result }); save(); showResult(question, result); renderHistory();
    } catch (error) { if (error.name !== "AbortError") { els.progress.hidden = true; els.error.hidden = false; els.errorText.textContent = error.message || "问答服务暂时不可用。"; } }
    finally { controller = null; requestMeta = null; els.send.disabled = false; }
  }
  async function loadModels() { try { const r = await fetch("/expert-qa/api/models"); const d = await r.json(); els.model.replaceChildren(...(d.items || []).map((m) => { const o = new Option(m.label || m.id, m.id); return o; })); els.model.disabled = false; els.modelNote.textContent = "回答由所选模型生成，并以 ATNP 文献证据为依据。"; } catch (_) { els.model.replaceChildren(new Option("使用默认模型", "")); } }
  async function health() { try { const r = await fetch("/expert-qa/api/health"); const d = await r.json(); els.dot.className = d.status === "ok" ? "ok" : ""; els.status.textContent = d.status === "ok" ? "问答服务正常" : "模型服务需要检查"; } catch (_) { els.status.textContent = "无法连接问答服务"; } }
  els.form.onsubmit = async (e) => { e.preventDefault(); if (controller) { controller.abort(); if (requestMeta?.request_id && requestMeta?.cancel_token) { try { await window.atnpSseClient.cancel(requestMeta.request_id, requestMeta.cancel_token); } catch (_) {} } return; } ask(els.input.value); }; els.input.oninput = () => { els.input.style.height = "auto"; els.input.style.height = `${Math.min(els.input.scrollHeight, 130)}px`; }; els.input.onkeydown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); els.form.requestSubmit(); } }; document.querySelectorAll("[data-question]").forEach((b) => b.onclick = () => ask(b.dataset.question)); $("qa-new").onclick = () => { active = null; els.welcome.hidden = false; els.conversation.hidden = true; renderHistory(); }; $("qa-clear").onclick = () => { conversations = []; save(); active = null; renderHistory(); }; $("qa-menu").onclick = () => els.shell.classList.toggle("qa-open"); $("qa-copy").onclick = () => navigator.clipboard?.writeText(answerText); els.retry.onclick = () => ask(lastQuestion);
  renderHistory(); loadModels(); health();
})();
