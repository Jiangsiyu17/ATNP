window.atnpSseClient = (() => {
  async function consume(response, onEvent) {
    if (!response.body) throw new Error("浏览器不支持读取流式响应。");
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    const handle = (block) => {
      if (!block || block.startsWith(":")) return;
      let name = "message"; const lines = [];
      block.replace(/\r\n/g, "\n").split("\n").forEach((line) => {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        if (line.startsWith("data:")) lines.push(line.slice(5).trimStart());
      });
      if (lines.length) onEvent(name, JSON.parse(lines.join("\n")));
    };
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream: !done}).replace(/\r\n/g, "\n");
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) { handle(buffer.slice(0, cut)); buffer = buffer.slice(cut + 2); }
      if (done) break;
    }
    if (buffer.trim()) handle(buffer.trim());
  }
  async function ask(options) {
    const apiBase = options.apiBase || "/expert-qa/api";
    const response = await fetch(`${apiBase}/chat/stream`, {
      method: "POST", headers: {"Content-Type": "application/json"}, signal: options.signal,
      body: JSON.stringify({
        question: options.question, conversation_id: options.conversationId,
        context_compound: options.contextCompound || null, model: options.model || null,
        validate_claims: null, task_router: true, search_agent: true,
        include_debug: false, max_tokens: 768
      })
    });
    if (!response.ok) { let detail = ""; try { detail = (await response.json()).detail || ""; } catch (_) {} const e = new Error(detail || `请求失败（HTTP ${response.status}）`); e.status = response.status; throw e; }
    let result = null, streamError = null;
    await consume(response, (event, data) => {
      if (event === "meta") options.onMeta?.(data);
      else if (event === "progress") options.onProgress?.(data);
      else if (event === "result") result = data;
      else if (event === "error") streamError = data;
    });
    if (streamError) { const e = new Error(streamError.detail || "问答服务出错"); e.status = streamError.status_code; throw e; }
    if (!result) throw new Error("服务未返回回答。");
    return result;
  }
  async function cancel(requestId, cancelToken, apiBase = "/expert-qa/api") {
    const response = await fetch(`${apiBase}/chat/${encodeURIComponent(requestId)}/cancel`, {method: "POST", headers: {"X-Cancel-Token": cancelToken}});
    if (!response.ok) throw new Error(`取消失败（HTTP ${response.status}）`);
    return response.json();
  }
  return {ask, cancel};
})();
