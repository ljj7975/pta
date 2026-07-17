const state = {
  selectedClassName: null,
  isLoaded: false,
  currentMode: "live",
  // Config editor state
  originalConfig: null,     // Original YAML config (for reset)
  currentConfigDir: null,   // Currently selected config dir
  currentDataset: null,     // Currently selected dataset
  tempConfigPath: null,     // Path to current temp config file
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
  const pca = s.per_class_accuracy || {};
  const targetPca = pca[s.target_name];
  const targetAcc = targetPca ? `${targetPca.correct}/${targetPca.total}` : "—";
  const html = `
    <span>Dataset: <strong>${s.dataset || "—"}</strong></span> |
    <span>Sample: <strong>${s.sample_idx + 1}/${s.num_samples}</strong></span> |
    <span>Acc: <strong>${s.running_acc.toFixed(1)}%</strong></span> |
    <span>CLIP Acc: <strong>${(s.text_running_acc ?? 0).toFixed(1)}%</strong></span> |
    <span>${s.target_name}: <strong>${targetAcc}</strong></span>
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
  // s.classes is sorted by final_logit descending — build class_id-indexed
  // arrays so that arr[class_id] gives the correct score for that class,
  // regardless of sort order.
  function _scoreArray(scoreFn) {
    const C = s.classes.length;
    return Array.from({length: C}, (_, cid) => {
      const cls = s.classes.find(c => c.class_id === cid);
      return cls ? scoreFn(cls) : 0;
    });
  }
  const textOnlyArr = s.text_only_logits || _scoreArray(c => c.text_score);
  const ptaArr = s.pta_logits || _scoreArray(c => c.final_logit);
  const fullArr = s.full_logits || _scoreArray(c => c.final_logit_full ?? c.final_logit);

  const textOnlyPred = s.classes.reduce((best, c) => (textOnlyArr[c.class_id] > textOnlyArr[best.class_id] ? c : best), s.classes[0]);
  const ptaPred = s.classes.reduce((best, c) => (ptaArr[c.class_id] > ptaArr[best.class_id] ? c : best), s.classes[0]);
  const fullPred = s.classes.reduce((best, c) => (fullArr[c.class_id] > fullArr[best.class_id] ? c : best), s.classes[0]);

  const textPredCorrect = textOnlyPred.class_id === s.target;
  const ptaPredCorrect = ptaPred.class_id === s.target;
  const fullPredCorrect = fullPred.class_id === s.target;

  let html = `
    <div class="proto-impact-bar ${fullPred.class_id !== textOnlyPred.class_id ? (fullPredCorrect ? "impact-fixed" : "impact-changed") : "impact-same"}">
      <strong>Text-only:</strong> ${textOnlyPred.class_name} ${textPredCorrect ? "✓" : "✗"}
      &nbsp;→&nbsp;
      <strong>PTA:</strong> ${ptaPred.class_name} ${ptaPredCorrect ? "✓" : "✗"}
      &nbsp;→&nbsp;
      <strong>Full:</strong> ${fullPred.class_name} ${fullPredCorrect ? "✓" : "✗"}
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

  function scoreSection(title, logitArr, predClassId) {
    // Each section independently picks its own top-5 sorted by its own scores
    const indexed = s.classes.map(c => ({ ...c, val: logitArr[c.class_id] }));
    const sectionTop5 = indexed.sort((a, b) => b.val - a.val).slice(0, 5);

    const sectionScores = sectionTop5.map(c => c.val);
    const sMin = Math.min(...sectionScores);
    const sMax = Math.max(...sectionScores);
    const sRange = sMax - sMin || 1;

    let bars = "";
    for (const c of sectionTop5) {
      const val = c.val;
      const pct = ((val - sMin) / sRange) * 100;
      const isGT = c.class_id === s.target;
      const isPred = c.class_id === predClassId;
      let fillClass = "";
      if (isGT) fillClass = "gt-class";
      if (isPred && c.class_id !== s.target) fillClass = "pred-class";
      const tag = (c.class_id === predClassId && c.class_id !== fullPred.class_id)
        ? `<span class="text-pred-tag">predicted</span>` : "";
      bars += `
        <div class="score-bar">
          <div class="score-label">${c.class_name}${tag}</div>
          <div class="score-bar-container">
            <div class="score-bar-fill ${fillClass}" style="width: ${pct}%">${val.toFixed(2)}</div>
          </div>
        </div>
      `;
    }
    return `
      <div class="score-section">
        <div class="score-section-title">${title}</div>
        ${bars}
      </div>
    `;
  }

  const tauText = s.tau_text ?? 1;
  const tauImg = s.tau_image_proto ?? 100;
  const tauPch = s.tau_patch_proto ?? 10;

  html += scoreSection(`Text Only (τ_text=${tauText} × CLIP)`, textOnlyArr, textOnlyPred.class_id);
  html += scoreSection(`PTA (text + image, τ_img=${tauImg})`, ptaArr, ptaPred.class_id);
  html += scoreSection(`Full (+ patch, τ_pch=${tauPch})`, fullArr, fullPred.class_id);

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
    "Text Score": "Raw CLIP text-image cosine similarity (×100)",
    "Tau Image": "Weight for image-level prototype (global parameter)",
    "Tau Patch": "Weight for patch-level prototype (global parameter)",
    "Image Score": "tau_image × raw_image_proto_logit",
    "Patch Score": "tau_patch × alpha × quality_gate × raw_proto",
    "Alpha": "Per-class evidence gate from update history (0→alpha_max)",
    "Quality Gate": "Variance-based gate: var(proto_scores) / (var + eps)",
    "Final Logit": "tau_text × text + tau_image × image_proto (production PTA)",
    "Final (Full)": "Final logit + Patch Score (includes patch contribution)",
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

    const pca = s.per_class_accuracy || {};
    const classPca = pca[c.class_name];
    const classAccStr = classPca ? `${classPca.correct}/${classPca.total}` : "";

    html += `
      <div class="accordion-item ${itemClass}">
        <div class="accordion-header" data-class-id="${c.class_id}">
          <span>
            <strong>${c.class_name}</strong> 
            <span style="color: #94a3b8; font-size: 0.9rem;">(K=${c.bank_K}/${maxK}, logit=${c.final_logit.toFixed(3)})</span>
            ${classAccStr ? `<span style="color: #0ea5e9; font-size: 0.85rem; margin-left: 8px;">acc: ${classAccStr}</span>` : ""}
          </span>
          <span class="accordion-icon">▼</span>
        </div>
        <div class="accordion-content">
          <div class="class-details">
            <div class="detail-row" title="${tooltips['Text Score']}">
              <span class="detail-label">Text Score</span>
              <span class="detail-value">${c.text_score.toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Tau Image']}">
              <span class="detail-label">Tau Image</span>
              <span class="detail-value">${(s.tau_image_proto ?? 100).toFixed(1)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Tau Patch']}">
              <span class="detail-label">Tau Patch</span>
              <span class="detail-value">${(s.tau_patch_proto ?? 10).toFixed(1)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Image Score']}">
              <span class="detail-label">Image Score</span>
              <span class="detail-value">${c.image_score.toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Patch Score']}">
              <span class="detail-label">Patch Score</span>
              <span class="detail-value">${c.patch_score.toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Alpha']}">
              <span class="detail-label">Alpha</span>
              <span class="detail-value">${(c.alpha ?? c.class_penalty).toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Quality Gate']}">
              <span class="detail-label">Quality Gate</span>
              <span class="detail-value">${(s.proto_formula?.quality_gate ?? c.quality_gate ?? 0).toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Final Logit']}" style="border-left-color: #0ea5e9;">
              <span class="detail-label">Final Logit</span>
              <span class="detail-value">${c.final_logit.toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Final (Full)']}" style="border-left-color: #f59e0b;">
              <span class="detail-label">Final (Full)</span>
              <span class="detail-value">${(c.final_logit_full ?? c.final_logit).toFixed(4)}</span>
            </div>
            <div class="detail-row" title="${tooltips['Final Logit']}" style="grid-column: 1 / -1; border-left-color: #f59e0b;">
              <span class="detail-label">Decomposition</span>
              <span class="detail-value">final = text(${c.text_score.toFixed(2)}) + image(${c.image_score.toFixed(2)}) + patch(${c.patch_score.toFixed(2)}) = ${(c.final_logit_full ?? c.final_logit).toFixed(2)}</span>
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
  dsSelect.removeEventListener("change", dsSelect._configHandler);
  dsSelect._configHandler = async () => {
    const configDir = document.getElementById("configInput").value;
    const dataset = dsSelect.value;
    if (configDir && dataset) {
      await loadConfigParams(configDir, dataset);
    }
  };
  dsSelect.addEventListener("change", dsSelect._configHandler);
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
      document.getElementById("loadingBar").classList.remove("hidden");
      document.getElementById("loadingStatus").classList.remove("hidden");
      const nSamples = Number(document.getElementById("nSamplesInput").value) || 200;
      const mode = document.getElementById("modeSelect").value;
      document.getElementById("loadingStatus").textContent =
        mode === "live"
          ? `Processing ${nSamples} samples... (may take a few minutes)`
          : "Loading records...";
      await loadData();
    } catch (err) {
      setError("Error loading data: " + err.message);
    } finally {
      document.getElementById("loadBtn").disabled = false;
      document.getElementById("loadBtn").textContent = "Load";
      document.getElementById("loadingBar").classList.add("hidden");
      document.getElementById("loadingStatus").classList.add("hidden");
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
      state.isLoaded = false;
      updateNavButtonsState();
      setError("Session lost (server reloaded). Please load data again.");
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
      state.isLoaded = false;
      updateNavButtonsState();
      setError("Session lost (server reloaded). Please load data again.");
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
      state.isLoaded = false;
      updateNavButtonsState();
      setError("Session lost (server reloaded). Please load data again.");
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
      state.isLoaded = false;
      updateNavButtonsState();
      setError("Session lost (server reloaded). Please load data again.");
    }
  });

  document.getElementById("gotoInput").addEventListener("keydown", async (e) => {
    if (e.key !== "Enter") return;
    if (!state.isLoaded) return;
    try {
      setError("");
      await goToSample();
    } catch (err) {
      state.isLoaded = false;
      updateNavButtonsState();
      setError("Session lost (server reloaded). Please load data again.");
    }
  });

  document.getElementById("configEditorToggle").addEventListener("click", toggleConfigEditor);
  document.getElementById("applyConfigBtn").addEventListener("click", async () => {
    try {
      await applyConfigAndRerun();
    } catch (err) {
      setError("Error applying config: " + err.message);
    }
  });
  document.getElementById("resetConfigBtn").addEventListener("click", async () => {
    try {
      await resetConfig();
    } catch (err) {
      setError("Error resetting config: " + err.message);
    }
  });
}

function toggleConfigEditor() {
  const content = document.getElementById("configEditorContent");
  const icon = document.querySelector("#configEditorToggle .toggle-icon");
  content.classList.toggle("open");
  icon.classList.toggle("open");
}

async function loadConfigParams(configDir, dataset) {
  try {
    setError("");
    const data = await api("/api/config/load", "POST", { config_dir: configDir, dataset });
    state.originalConfig = data.config;
    state.currentConfigDir = configDir;
    state.currentDataset = dataset;
    renderConfigParams(data.config);
    updateYamlPreview(data.config);
    document.getElementById("applyConfigBtn").disabled = false;
    document.getElementById("resetConfigBtn").disabled = false;
  } catch (err) {
    document.getElementById("configParams").innerHTML =
      '<p class="muted">Could not load config: ' + err.message + "</p>";
    document.getElementById("yamlPreview").textContent = "";
    document.getElementById("applyConfigBtn").disabled = true;
    document.getElementById("resetConfigBtn").disabled = true;
  }
}

function flattenConfig(obj, prefix) {
  const entries = [];
  for (const [key, value] of Object.entries(obj)) {
    const dotKey = prefix ? `${prefix}.${key}` : key;
    if (typeof value === "object" && value !== null && !Array.isArray(value)) {
      entries.push(...flattenConfig(value, dotKey));
    } else {
      entries.push({ key: dotKey, value });
    }
  }
  return entries;
}

function renderConfigParams(config) {
  const container = document.getElementById("configParams");
  container.innerHTML = "";
  const leaves = flattenConfig(config, "");
  for (const { key, value } of leaves) {
    const row = document.createElement("div");
    row.className = "config-param-row";
    row.dataset.key = key;
    row.dataset.original = String(value);

    const label = document.createElement("label");
    label.textContent = key;

    const input = document.createElement("input");
    input.value = String(value);
    if (typeof value === "number") {
      input.type = "number";
      input.step = value % 1 === 0 ? "1" : "any";
    } else {
      input.type = "text";
    }

    const typeSpan = document.createElement("span");
    typeSpan.className = "param-type";
    typeSpan.textContent = typeof value;

    input.addEventListener("input", () => {
      const original = row.dataset.original;
      const current = input.value;
      const changed = current !== original;
      row.classList.toggle("changed", changed);
      const cfg = collectConfigFromForm();
      updateYamlPreview(cfg);
    });

    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(typeSpan);
    container.appendChild(row);
  }
}

function unflattenConfig(flat) {
  const result = {};
  for (const [dotKey, value] of Object.entries(flat)) {
    const parts = dotKey.split(".");
    let cur = result;
    for (let i = 0; i < parts.length - 1; i++) {
      if (!(parts[i] in cur)) cur[parts[i]] = {};
      cur = cur[parts[i]];
    }
    cur[parts[parts.length - 1]] = value;
  }
  return result;
}

function collectConfigFromForm() {
  const flat = {};
  for (const row of document.querySelectorAll(".config-param-row")) {
    const key = row.dataset.key;
    const input = row.querySelector("input");
    let value = input.value;
    if (input.type === "number") {
      value = Number(value);
    } else if (value === "true") {
      value = true;
    } else if (value === "false") {
      value = false;
    }
    flat[key] = value;
  }
  return unflattenConfig(flat);
}

function updateYamlPreview(config) {
  document.getElementById("yamlPreview").textContent = buildYamlString(config);
}

function buildYamlString(obj, indent) {
  indent = indent || "";
  const lines = [];
  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === "object" && value !== null && !Array.isArray(value)) {
      lines.push(`${indent}${key}:`);
      lines.push(buildYamlString(value, indent + "  "));
    } else if (typeof value === "string") {
      lines.push(`${indent}${key}: "${value}"`);
    } else if (typeof value === "boolean") {
      lines.push(`${indent}${key}: ${value ? "true" : "false"}`);
    } else {
      lines.push(`${indent}${key}: ${value}`);
    }
  }
  return lines.join("\n");
}

function collectFormOverrides() {
  const overrides = {};
  for (const row of document.querySelectorAll(".config-param-row")) {
    if (!row.classList.contains("changed")) continue;
    const key = row.dataset.key;
    const input = row.querySelector("input");
    let value = input.value;
    if (input.type === "number") {
      value = Number(value);
    } else if (value === "true") {
      value = true;
    } else if (value === "false") {
      value = false;
    }
    overrides[key] = value;
  }
  return overrides;
}

async function applyConfigAndRerun() {
  const overrides = collectFormOverrides();
  if (Object.keys(overrides).length === 0) {
    document.getElementById("configStatus").textContent = "No changes to apply";
    return;
  }
  try {
    setError("");
    document.getElementById("configStatus").textContent = "Saving config...";
    const data = await api("/api/config/save-temp", "POST", {
      config_dir: state.currentConfigDir,
      dataset: state.currentDataset,
      overrides,
    });
    state.tempConfigPath = data.temp_path;
    document.getElementById("configInput").value = data.temp_path;
    document.getElementById("configStatus").textContent = "⚡ Config saved, reloading...";
    await loadData();
    document.getElementById("configStatus").textContent = "⚡ Applied and reloaded";
  } catch (err) {
    setError("Error applying config: " + err.message);
    document.getElementById("configStatus").textContent = "Error: " + err.message;
  }
}

async function resetConfig() {
  if (state.originalConfig) {
    renderConfigParams(state.originalConfig);
    updateYamlPreview(state.originalConfig);
    document.getElementById("configStatus").textContent = "🔄 Reset to defaults";
  }
}

(async function boot() {
  wire();
  toggleModeFields();
  updateNavButtonsState();
  try {
    await refreshStatus();
    const configDir = document.getElementById("configInput").value;
    const dsSelect = document.getElementById("datasetSelect");
    if (configDir && dsSelect.value) {
      await loadConfigParams(configDir, dsSelect.value);
    }
  } catch (err) {
    setError("Failed to initialize: " + err.message);
  }
})();
