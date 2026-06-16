const state = {
  selectedClassName: null,
  isLoaded: false,
  currentMode: "live",
};

function updateNavButtonsState() {
  const loaded = state.isLoaded;
  document.getElementById("nextBtn").disabled = !loaded;
  document.getElementById("restartBtn").disabled = !loaded;
  document.getElementById("exportBtn").disabled = !loaded;
  document.getElementById("gotoBtn").disabled = !loaded;
  document.getElementById("gotoInput").disabled = !loaded;

  document.getElementById("nextBtn").style.opacity = loaded ? "1" : "0.5";
  document.getElementById("restartBtn").style.opacity = loaded ? "1" : "0.5";
  document.getElementById("exportBtn").style.opacity = loaded ? "1" : "0.5";
  document.getElementById("gotoBtn").style.opacity = loaded ? "1" : "0.5";
  document.getElementById("gotoInput").style.opacity = loaded ? "1" : "0.5";
}

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: {} };
  if (body !== null) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function setError(msg) {
  const errorBox = document.getElementById("errorBox");
  if (msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove("hidden");
  } else {
    errorBox.classList.add("hidden");
  }
}

function toggleModeFields() {
  const mode = document.getElementById("modeSelect").value;
  const isLive = mode === "live";

  document.getElementById("configLabel").classList.toggle("hidden", !isLive);
  document.getElementById("backboneLabel").classList.toggle("hidden", !isLive);
  document.getElementById("dataRootLabel").classList.toggle("hidden", !isLive);
  document.getElementById("nSamplesLabel").classList.toggle("hidden", !isLive);
  document.getElementById("recordsLabel").classList.toggle("hidden", isLive);

  state.currentMode = mode;
}

function updateStatusInfo(s) {
  const html = `
    <span>Dataset: <strong>${s.dataset || "—"}</strong></span> |
    <span>Sample: <strong>${s.sample_idx + 1}/${s.num_samples}</strong></span> |
    <span>Acc: <strong>${s.running_acc.toFixed(1)}%</strong></span> |
    <span>CLIP Acc: <strong>${(s.text_running_acc ?? 0).toFixed(1)}%</strong></span>
  `;
  document.getElementById("statusInfo").innerHTML = html;
}

function renderPredictionPanel(s) {
  const gtCard = `
    <div class="pred-card gt">
      <div class="label">Ground Truth</div>
      <div class="value">${s.target_name}</div>
    </div>
  `;

  const predClass = s.correct ? "right" : "wrong";
  const predCard = `
    <div class="pred-card ${predClass}" id="predCard">
      <div class="label">${s.correct ? "✓ Correct" : "✗ Incorrect"}</div>
      <div class="value">${s.predicted_name}</div>
    </div>
  `;

  const clipName = s.classnames ? s.classnames[s.clip_pred] : String(s.clip_pred);
  const stats = `
    <div class="pred-stats">
      <div><strong>Running Acc:</strong> ${s.running_acc.toFixed(1)}%</div>
      <div><strong>CLIP Acc:</strong> ${(s.text_running_acc ?? 0).toFixed(1)}%</div>
      <div><strong>CLIP ZS:</strong> ${clipName}</div>
      <div><strong>CLIP Conf:</strong> ${(s.clip_conf * 100).toFixed(1)}%</div>
    </div>
  `;

  const imageHtml = `
    <div class="sample-image-container">
      <div class="label">Sample Image</div>
      <canvas id="sampleCanvas" width="224" height="224"></canvas>
      <div class="sample-label">${s.target_name} → ${s.predicted_name}</div>
    </div>
  `;

  document.querySelector(".pred-header").innerHTML = imageHtml + gtCard + '<div class="pred-arrow">→</div>' + predCard + stats;

  // Draw image if available
  if (s.image_data) {
    const canvas = document.getElementById("sampleCanvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, 224, 224);
    };
    img.src = s.image_data;
  } else {
    const canvas = document.getElementById("sampleCanvas");
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No image data", 112, 112);
  }
}

function renderScoreVisualization(s) {
  const sorted = [...s.classes].sort((a, b) => b.final_logit - a.final_logit);
  const top5 = sorted.slice(0, 5);

  const minLogit = Math.min(...top5.map(c => c.final_logit));
  const maxLogit = Math.max(...top5.map(c => c.final_logit));
  const range = maxLogit - minLogit || 1;

  // Text-only prediction: argmax of text_score across all classes.
  const textOnly = [...s.classes].reduce((best, c) => c.text_score > best.text_score ? c : best, s.classes[0]);
  const finalPred = sorted[0];
  const textFixed = (textOnly.class_id !== finalPred.class_id);
  const textPredCorrect = textOnly.class_id === s.target;
  const finalPredCorrect = finalPred.class_id === s.target;

  let html = `
    <div class="proto-impact-bar ${textFixed ? (finalPredCorrect ? "impact-fixed" : "impact-changed") : "impact-same"}">
      <strong>Text-only:</strong> ${textOnly.class_name} ${textPredCorrect ? "✓" : "✗"}
      &nbsp;→&nbsp;
      <strong>Final:</strong> ${finalPred.class_name} ${finalPredCorrect ? "✓" : "✗"}
      &nbsp;|&nbsp;
      ${textFixed
        ? (finalPredCorrect ? "<span style='color:#10b981'>Proto branch fixed the prediction</span>"
                            : "<span style='color:#f59e0b'>Proto branch changed prediction (still wrong)</span>")
        : "<span style='color:#94a3b8'>Proto branch did not change prediction</span>"}
    </div>
  `;

  const gate = s.update_gate || null;
  if (gate) {
    const gateClass = gate.passed ? "gate-pass" : "gate-fail";
    const gateText = gate.passed ? "Prototype update: YES" : "Prototype update: NO";
    const gateClassName = gate.class_name || "—";
    const reason = gate.reason || (gate.passed
      ? "Update executed."
      : "Update skipped by gating conditions.");
    const confPassed = (gate.best_conf || 0) > (gate.conf_thresh || 0);
    const marginPassed = (gate.margin || 0) >= (gate.margin_thresh || 0);
    const bankInfo = (gate.max_k || 0) > 0
      ? `bank K=${gate.bank_k || 0}/${gate.max_k}`
      : "";
    html += `
      <div class="update-gate ${gateClass}">
        <strong>${gateText}</strong>
        <span>class=${gateClassName}</span>
        ${bankInfo ? `<span>${bankInfo}</span>` : ""}
        <span>confidence: ${(100 * (gate.best_conf || 0)).toFixed(1)}% ${confPassed ? "(pass)" : "(fail)"}</span>
        <span>margin: ${(100 * (gate.margin || 0)).toFixed(1)}% ${marginPassed ? "(pass)" : "(fail)"}</span>
        <span>required: confidence>${(100 * (gate.conf_thresh || 0)).toFixed(1)}%, margin≥${(100 * (gate.margin_thresh || 0)).toFixed(1)}%</span>
        <span><strong>why:</strong> ${reason}</span>
      </div>
    `;
  }

  for (const c of top5) {
    const pct = ((c.final_logit - minLogit) / range) * 100;
    const isGT = c.class_id === s.target;
    const isPred = c.class_id === s.predicted;
    const isTextPred = c.class_id === textOnly.class_id;
    let fillClass = "";
    if (isGT) fillClass = "gt-class";
    if (isPred && !s.correct) fillClass = "pred-class";
    const textTag = (isTextPred && textFixed) ? `<span class="text-pred-tag">text-only</span>` : "";

    html += `
      <div class="score-bar">
        <div class="score-label">${c.class_name}${textTag}</div>
        <div class="score-bar-container">
          <div class="score-bar-fill ${fillClass}" style="width: ${pct}%">
            ${c.final_logit.toFixed(2)}
          </div>
        </div>
        <div class="score-value">${(c.prob * 100).toFixed(1)}%</div>
      </div>
    `;
  }
  document.getElementById("scoreVisualization").innerHTML = html;
}

function inferPatchGrid(topMatches) {
  let maxIdx = -1;
  for (const m of topMatches || []) {
    for (const idx of (m.top_patch_indices || [])) {
      if (idx > maxIdx) maxIdx = idx;
    }
  }
  if (maxIdx < 0) return 14;
  const grid = Math.round(Math.sqrt(maxIdx + 1));
  return grid > 0 ? grid : 14;
}

function patchStripHtml(match, grid, prefix) {
  const idxs = (match.top_patch_indices || []).slice(0, 3);
  if (!idxs.length) return "";
  return `
    <div class="patch-strip">
      ${idxs.map((idx, i) => `
        <div>
          <canvas
            class="patch-thumb ${i === 0 ? "centered" : ""}"
            width="42"
            height="42"
            title="${i === 0 ? "Best match patch" : "Patch"} idx=${idx}"
            data-patch-index="${idx}"
            data-patch-grid="${grid}"
            data-patch-key="${prefix}_${match.class_id}_${match.proto_idx}_${idx}_${i}"
          ></canvas>
        </div>
      `).join("")}
      <span class="patch-label">idx: ${idxs.join(", ")}</span>
    </div>
  `;
}

function protoIdentityHtml(match) {
  if (!match.proto_patch_data) {
    return '<div class="proto-identity"><span class="patch-label">Prototype patch: not captured yet</span></div>';
  }
  const idxText = (match.proto_patch_idx !== null && match.proto_patch_idx !== undefined && match.proto_patch_idx >= 0)
    ? `${match.proto_patch_idx}`
    : "—";
  const simText = (match.proto_patch_sim !== null && match.proto_patch_sim !== undefined && match.proto_patch_sim >= 0)
    ? `${Number(match.proto_patch_sim).toFixed(3)}`
    : "—";
  return `
    <div class="proto-identity">
      <img class="proto-thumb" src="${match.proto_patch_data}" alt="Prototype representative patch" title="From patch idx=${idxText} (sim=${simText})" />
      <span class="patch-label" title="Patch from this prototype's training image (idx=patch grid position, sim=cosine similarity to proto center)">Proto patch idx=${idxText} sim=${simText}</span>
    </div>
  `;
}

function drawPatchThumbs(imageData) {
  const canvases = Array.from(document.querySelectorAll("canvas.patch-thumb"));
  if (!canvases.length) return;

  const paintPlaceholder = () => {
    for (const c of canvases) {
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, c.width, c.height);
      ctx.strokeStyle = "#334155";
      ctx.strokeRect(0, 0, c.width, c.height);
    }
  };

  if (!imageData) {
    paintPlaceholder();
    return;
  }

  const img = new Image();
  img.onload = () => {
    for (const c of canvases) {
      const patchIdx = Number(c.dataset.patchIndex || -1);
      const grid = Number(c.dataset.patchGrid || 14);
      const ctx = c.getContext("2d");

      if (patchIdx < 0 || grid <= 0) {
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(0, 0, c.width, c.height);
        continue;
      }

      const patchW = img.width / grid;
      const patchH = img.height / grid;
      const px = patchIdx % grid;
      const py = Math.floor(patchIdx / grid);
      const sx = px * patchW;
      const sy = py * patchH;

      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.drawImage(img, sx, sy, patchW, patchH, 0, 0, c.width, c.height);
    }
  };
  img.src = imageData;
}

function renderAccordion(s) {
  const tooltips = {
    "Text Score": "CLIP text embedding contribution (~0–100)",
    "Raw Proto": "Class-level prototype evidence before alpha/quality/tau scaling",
    "Alpha": "Per-class evidence gate from update history",
    "Proto Term": "Actual prototype contribution added to text score after weighting",
    "Final Logit": "Final Logit = Text Score + Proto Term",
    "Tau Eff": "Sample-level proto scaling factor (adaptive tau if enabled)",
    "Softmax Prob": "Probability from softmax(final_logits), 0–100%"
  };

  const sorted = [...s.classes].sort((a, b) => b.final_logit - a.final_logit);
  let html = "";

  for (const c of sorted) {
    const isGT = c.class_id === s.target;
    const isPred = c.class_id === s.predicted;
    const itemClass = isGT ? "gt" : isPred ? "pred" : "";
    const maxK = c.max_k || s.update_gate?.max_k || c.bank_K;

    const classProtoMap = s.class_prototypes || {};
    const classAllProtos = classProtoMap[String(c.class_id)] || [];
    const fgThresh = s.proto_formula?.foreground_appw_thresh ?? 0.5;
    const foregroundProtos = classAllProtos.filter(p => (p.app_w || 0) >= fgThresh);
    const backgroundProtos = classAllProtos.filter(p => (p.app_w || 0) < fgThresh);

    // Keep this-sample match context visible inside foreground/background lists.
    const matchedMap = new Map();
    (s.top_matches || []).forEach(m => {
      if (m.class_id === c.class_id) matchedMap.set(m.proto_idx, m);
    });

    foregroundProtos.sort((a, b) => (b.app_w || 0) - (a.app_w || 0));
    backgroundProtos.sort((a, b) => (a.app_w || 0) - (b.app_w || 0));
    
    // Build HTML for prototypes
    let protoHtml = `
      <div class="proto-section-title proto-section-title-active">
        <strong>Foreground prototypes (${foregroundProtos.length}/${c.bank_K}, app_w ≥ ${fgThresh.toFixed(3)})</strong>
      </div>`;
    
    const patchGrid = inferPatchGrid(s.top_matches);

    // Foreground prototypes based on app_w
    if (foregroundProtos.length) {
      protoHtml += foregroundProtos.map(m => {
        const matched = matchedMap.get(m.proto_idx);
        const maxSim = matched?.max_sim ?? m.max_sim ?? 0;
        const centered = matched?.centered_score ?? m.centered_score ?? 0;
        const contrib = matched?.contrib ?? m.contrib ?? 0;
        return `
        <div class="proto-item" style="background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10b981;">
          <strong style="color: #10b981;">FG Proto ${m.proto_idx}</strong>: 
          <span title="Max cosine sim using one-to-one patch assignment per class">max_sim=${maxSim.toFixed(3)}</span>, 
          <span title="# samples this prototype matched / # samples used to update this class">app=${m.appearance.toFixed(0)}/${m.update_samples || 0}</span>, 
          <span title="Appearance ratio vs total update samples for this class">app_w=${(m.app_w || 0).toFixed(3)}</span>,
          <span title="Confidence weight (0.5 to 1.0 based on bank size)">ew=${m.evidence_weight.toFixed(2)}</span>,
          <span title="Centered score = alpha×quality_gate×raw_score">centered=${centered.toFixed(4)}</span>,
          <span title="Proto contribution to final logit = τ × centered_score">contrib=${contrib.toFixed(3)}</span>
          ${protoIdentityHtml(matched || m)}
          ${patchStripHtml(matched || m, patchGrid, "class")}
        </div>
      `;
      }).join("");
    } else {
      protoHtml += `
        <div class="proto-empty">No foreground prototypes for this class under current app_w threshold.</div>
      `;
    }

    // Show non-active prototypes right below active ones using neutral styling.
    protoHtml += `
      <div class="proto-section-title proto-section-title-inactive">
        <strong>Background prototypes (${backgroundProtos.length}, app_w &lt; ${fgThresh.toFixed(3)})</strong>
      </div>`;

    if (backgroundProtos.length > 0) {
      const shownInactive = backgroundProtos.slice(0, 30);
      protoHtml += shownInactive.map(m => {
        const matched = matchedMap.get(m.proto_idx);
        const maxSim = matched?.max_sim ?? m.max_sim ?? 0;
        return `
        <div class="proto-item proto-item-inactive">
          <strong style="color: #94a3b8;">BG Proto ${m.proto_idx}</strong>:
          <span title="Max cosine sim using one-to-one patch assignment per class"> max_sim=${maxSim.toFixed(3)}</span>,
          <span title="# samples this prototype matched / # samples used to update this class"> app=${m.appearance.toFixed(0)}/${m.update_samples || 0}</span>,
          <span title="Appearance ratio vs total update samples for this class"> app_w=${(m.app_w || 0).toFixed(3)}</span>,
          <span title="Confidence weight (0.5 to 1.0 based on bank size)"> ew=${m.evidence_weight.toFixed(2)}</span>
          ${protoIdentityHtml(matched || m)}
        </div>
      `;
      }).join("");

      if (backgroundProtos.length > shownInactive.length) {
        protoHtml += `
          <div class="proto-empty">+ ${backgroundProtos.length - shownInactive.length} more background prototypes</div>
        `;
      }
    } else {
      protoHtml += `
        <div class="proto-empty">
          No background prototypes.
        </div>`;
    }

    html += `
      <div class="accordion-item ${itemClass}">
        <div class="accordion-header" data-class-id="${c.class_id}">
          <span>
            <strong>${c.class_name}</strong> 
            <span style="color: #94a3b8; font-size: 0.9rem;">(K=${c.bank_K}/${maxK}, logit=${c.final_logit.toFixed(3)})</span>
          </span>
          <span class="accordion-icon">▼</span>
        </div>
        <div class="accordion-content">
          <div class="class-details">
            <div class="detail-row" title="${tooltips['Text Score']}">
              <span class="detail-label">Text Score</span>
              <span class="detail-value">${c.text_score.toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Raw Proto']}">
              <span class="detail-label">Raw Proto</span>
              <span class="detail-value">${c.raw_proto.toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Alpha']}">
              <span class="detail-label">Alpha</span>
              <span class="detail-value">${(c.alpha ?? c.class_penalty).toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Proto Term']}">
              <span class="detail-label">Proto Term</span>
              <span class="detail-value">${c.proto_term.toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Tau Eff']}">
              <span class="detail-label">Tau Eff</span>
              <span class="detail-value">${c.tau_eff.toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Final Logit']}">
              <span class="detail-label">Final Logit</span>
              <span class="detail-value">${c.final_logit.toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Proto Term']}" style="grid-column: 1 / -1; border-left-color: #f59e0b;">
              <span class="detail-label">Proto Term Formula (tau_proto * alpha * quality_gate * raw_proto)</span>
              <span class="detail-value">${(s.proto_formula?.tau_proto ?? c.tau_proto ?? 0).toFixed(4)} × ${(c.alpha ?? c.class_penalty).toFixed(4)} × ${(s.proto_formula?.quality_gate ?? c.quality_gate ?? 0).toFixed(4)} × ${(c.delta_proto ?? c.raw_proto).toFixed(4)} = ${c.proto_term.toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Final Logit']}" style="grid-column: 1 / -1; border-left-color: #f59e0b;">
              <span class="detail-label">Formula (final_logit = text_score + proto_term)</span>
              <span class="detail-value">${c.text_score.toFixed(4)} + (${c.proto_term.toFixed(4)}) = ${c.final_logit.toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Softmax Prob']}">
              <span class="detail-label">Softmax Prob</span>
              <span class="detail-value">${(c.prob * 100).toFixed(2)}%</span>
            </div>
            <div class="protos-list">
              <strong style="color: #0ea5e9;">Prototype role by app_w (foreground/background), with this-sample match stats:</strong>
              ${protoHtml}
            </div>
          </div>
        </div>
      </div>
    `;
  }

  document.getElementById("classAccordion").innerHTML = html;

  // Wire accordion headers
  document.querySelectorAll(".accordion-header").forEach(header => {
    header.addEventListener("click", () => {
      const content = header.nextElementSibling;
      const isActive = content.classList.contains("active");

      // Keep other class panels as-is; only toggle the clicked one.
      content.classList.toggle("active", !isActive);
      header.classList.toggle("active", !isActive);
    });
  });
}

function renderMatchTable(s) {
  const tbody = document.querySelector("#matchTable tbody");
  tbody.innerHTML = "";
  const patchGrid = inferPatchGrid(s.top_matches);
  s.top_matches.slice(0, 20).forEach((m) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${m.class_name}</td>
      <td>${m.proto_idx}</td>
      <td>${m.appearance.toFixed(0)}</td>
      <td>${m.evidence_weight.toFixed(2)}</td>
      <td>${m.max_sim.toFixed(4)}</td>
      <td>${m.centered_score.toFixed(4)}</td>
      <td>${m.contrib.toFixed(3)}</td>
      <td>
        ${protoIdentityHtml(m)}
        ${patchStripHtml(m, patchGrid, "table")}
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function renderState(s) {
  updateStatusInfo(s);
  renderPredictionPanel(s);
  renderScoreVisualization(s);
  renderAccordion(s);
  renderMatchTable(s);
  drawPatchThumbs(s.image_data || null);

  const gotoInput = document.getElementById("gotoInput");
  gotoInput.min = "1";
  gotoInput.max = String(Math.max(1, s.num_samples || 1));
  gotoInput.value = String((s.sample_idx || 0) + 1);

  state.isLoaded = true;
  updateNavButtonsState();
  document.getElementById("contentSection").classList.remove("hidden");
}

async function refreshStatus() {
  const res = await api("/api/status");
  const dsSelect = document.getElementById("datasetSelect");
  dsSelect.innerHTML = (res.datasets || []).map(d => `<option value="${d}">${d}</option>`).join("");
  if (res.loaded) {
    renderState(res.state);
  }
}

async function loadData() {
  const mode = document.getElementById("modeSelect").value;
  const payload = {
    mode,
    dataset: document.getElementById("datasetSelect").value,
    config: document.getElementById("configInput").value,
    backbone: document.getElementById("backboneInput").value,
    data_root: document.getElementById("dataRootInput").value,
    n_samples: Number(document.getElementById("nSamplesInput").value),
    records: document.getElementById("recordsInput").value || null,
  };
  const res = await api("/api/load", "POST", payload);
  document.getElementById("loadInfo").textContent = `✓ Loaded ${res.state.num_samples} samples from ${res.source}`;
  renderState(res.state);
}

async function nextSample() {
  const res = await api("/api/next", "POST", { selected_class_name: state.selectedClassName });
  renderState(res.state);
}

async function restart() {
  const res = await api("/api/restart", "POST", { selected_class_name: state.selectedClassName });
  renderState(res.state);
}

async function saveCurrent() {
  const res = await api("/api/export-current", "POST", {});
  const fileName = res.path.split("/").pop();
  document.getElementById("loadInfo").textContent = `💾 Saved: ${fileName}`;
}

async function goToSample() {
  const input = document.getElementById("gotoInput");
  const sampleOneBased = Number(input.value);

  if (!Number.isInteger(sampleOneBased) || sampleOneBased < 1) {
    throw new Error("Sample index must be an integer >= 1.");
  }

  const maxSamples = Number(input.max || 1);
  if (sampleOneBased > maxSamples) {
    throw new Error(`Sample index must be <= ${maxSamples}.`);
  }

  const targetIdx = sampleOneBased - 1;
  const res = await api("/api/set-index", "POST", {
    idx: targetIdx,
    selected_class_name: state.selectedClassName,
  });
  renderState(res.state);
}

function setupModal() {
  const modal = document.getElementById("helpModal");
  const helpBtn = document.getElementById("helpBtn");
  const closeBtn = document.querySelector(".close");

  helpBtn.addEventListener("click", () => {
    modal.classList.remove("hidden");
  });

  closeBtn.addEventListener("click", () => {
    modal.classList.add("hidden");
  });

  window.addEventListener("click", (e) => {
    if (e.target === modal) {
      modal.classList.add("hidden");
    }
  });
}

function wire() {
  setupModal();

  document.getElementById("modeSelect").addEventListener("change", toggleModeFields);

  document.getElementById("loadBtn").addEventListener("click", async () => {
    try {
      setError("");
      document.getElementById("loadBtn").disabled = true;
      document.getElementById("loadBtn").textContent = "Loading...";
      await loadData();
    } catch (err) {
      setError("Error loading data: " + err.message);
    } finally {
      document.getElementById("loadBtn").disabled = false;
      document.getElementById("loadBtn").textContent = "Load";
    }
  });

  document.getElementById("nextBtn").addEventListener("click", async () => {
    if (!state.isLoaded) {
      setError("Please load data first using the Load button.");
      return;
    }
    try {
      setError("");
      await nextSample();
    } catch (err) {
      setError("Error: " + err.message);
    }
  });

  document.getElementById("restartBtn").addEventListener("click", async () => {
    if (!state.isLoaded) {
      setError("Please load data first using the Load button.");
      return;
    }
    try {
      setError("");
      await restart();
    } catch (err) {
      setError("Error: " + err.message);
    }
  });

  document.getElementById("exportBtn").addEventListener("click", async () => {
    if (!state.isLoaded) {
      setError("Please load data first using the Load button.");
      return;
    }
    try {
      setError("");
      await saveCurrent();
    } catch (err) {
      setError("Error saving: " + err.message);
    }
  });

  document.getElementById("gotoBtn").addEventListener("click", async () => {
    if (!state.isLoaded) {
      setError("Please load data first using the Load button.");
      return;
    }
    try {
      setError("");
      await goToSample();
    } catch (err) {
      setError("Error jumping to sample: " + err.message);
    }
  });

  document.getElementById("gotoInput").addEventListener("keydown", async (e) => {
    if (e.key !== "Enter") return;
    if (!state.isLoaded) return;
    try {
      setError("");
      await goToSample();
    } catch (err) {
      setError("Error jumping to sample: " + err.message);
    }
  });
}

(async function boot() {
  wire();
  toggleModeFields();
  updateNavButtonsState();
  try {
    await refreshStatus();
  } catch (err) {
    setError("Failed to initialize: " + err.message);
  }
})();
