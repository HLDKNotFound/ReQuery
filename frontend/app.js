const API_BASE = window.location.origin;

const EXAMPLES = {
  1: {
    query: "What did the paper say about training?",
    context: "Discussion about the Transformer architecture in Attention Is All You Need."
  },
  2: {
    query: "How does it work?",
    context: "The previous discussion is about Retrieval-Augmented Generation."
  },
  3: {
    query: "When did it first air on television?",
    context: "Discussion about The Simpsons and its early animation shorts."
  }
};

function setExample(id) {
  const ex = EXAMPLES[id];
  if (ex) {
    document.getElementById("query-input").value = ex.query;
    document.getElementById("context-input").value = ex.context;
  }
}

async function triggerRewrite() {
  const query = document.getElementById("query-input").value.trim();
  const context = document.getElementById("context-input").value.trim();
  const k = parseInt(document.getElementById("k-slider").value, 10);
  const temp = parseFloat(document.getElementById("temp-slider").value);

  if (!query) {
    alert("Please enter a user query.");
    return;
  }

  const btn = document.getElementById("rewrite-btn");
  btn.disabled = true;
  btn.innerText = "Generating...";

  try {
    const res = await fetch(`${API_BASE}/rewrite`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        context: context || null,
        k: k,
        temperature: temp,
        max_new_tokens: 64
      })
    });

    if (!res.ok) {
      throw new Error(`API error: ${res.statusText}`);
    }

    const data = await res.json();
    renderRewrittenQueries(data.rewritten_queries);
    
    // Show metrics
    const metricsBar = document.getElementById("metrics-bar");
    metricsBar.classList.remove("hidden");
    document.getElementById("metric-latency").innerText = `${data.latency_ms} ms`;
    document.getElementById("metric-k").innerText = `${data.k_generated}`;
    document.getElementById("retrieval-lat-pill").style.display = "none";

  } catch (err) {
    console.error(err);
    alert(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerText = "Generate Rewrites";
  }
}

async function triggerRAGSearch() {
  const query = document.getElementById("query-input").value.trim();
  const context = document.getElementById("context-input").value.trim();
  const k = parseInt(document.getElementById("k-slider").value, 10);
  const temp = parseFloat(document.getElementById("temp-slider").value);
  const fusion = document.getElementById("fusion-select").value;

  if (!query) {
    alert("Please enter a user query.");
    return;
  }

  const btn = document.getElementById("search-btn");
  btn.disabled = true;
  btn.innerText = "Searching...";

  try {
    const res = await fetch(`${API_BASE}/rag/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        context: context || null,
        k: k,
        temperature: temp,
        fusion_method: fusion,
        top_k_docs: 5
      })
    });

    if (!res.ok) {
      throw new Error(`API error: ${res.statusText}`);
    }

    const data = await res.json();
    renderRewrittenQueries(data.rewritten_queries);
    renderDocuments(data.retrieved_documents);

    // Show metrics
    const metricsBar = document.getElementById("metrics-bar");
    metricsBar.classList.remove("hidden");
    document.getElementById("metric-latency").innerText = `${data.rewriter_latency_ms} ms`;
    document.getElementById("metric-k").innerText = `${data.rewritten_queries.length}`;
    document.getElementById("metric-ret-latency").innerText = `${data.retrieval_latency_ms} ms`;
    document.getElementById("retrieval-lat-pill").style.display = "flex";
    document.getElementById("fusion-tag").innerText = `${data.fusion_method.toUpperCase()} Top-5`;

  } catch (err) {
    console.error(err);
    alert(`Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerText = "Execute RAG Retrieval";
  }
}

function renderRewrittenQueries(queries) {
  const container = document.getElementById("queries-list");
  if (!queries || queries.length === 0) {
    container.innerHTML = '<p class="placeholder-text">No queries returned.</p>';
    return;
  }

  container.innerHTML = queries.map((q, idx) => `
    <div class="query-card">
      <span class="query-num">Q${idx + 1}</span>
      <span class="query-text">${escapeHtml(q)}</span>
      <button class="copy-btn" title="Copy to clipboard" onclick="copyText('${escapeHtml(q).replace(/'/g, "\\'")}')">📋</button>
    </div>
  `).join("");
}

function renderDocuments(docs) {
  const container = document.getElementById("documents-list");
  if (!docs || docs.length === 0) {
    container.innerHTML = '<p class="placeholder-text">No documents retrieved.</p>';
    return;
  }

  container.innerHTML = docs.map((doc, idx) => `
    <div class="doc-card">
      <div class="doc-header">
        <span class="doc-rank">#${doc.rank || idx + 1}</span>
        <span class="doc-score">Score: ${doc.score}</span>
      </div>
      <div class="doc-text">${escapeHtml(doc.text)}</div>
    </div>
  `).join("");
}

function copyText(text) {
  navigator.clipboard.writeText(text).then(() => {
    alert("Copied to clipboard!");
  }).catch(() => {});
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
