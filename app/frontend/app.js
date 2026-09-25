const state = {
  features: [],
  sample: null,
  sampleStats: null,
  pcaReferences: [],
  patientCoords: null,
  currentPrediction: null,
  chatHistory: [],
  aiConfig: {
    provider_name: "auto",
    api_key: "",
    model_name: "",
    base_url: ""
  },
  attachedFiles: [],
  availableProviders: []
};

// UI Elements
const statusText = document.getElementById("statusText");
const reloadBtn = document.getElementById("reloadBtn");
const themeToggleBtn = document.getElementById("themeToggleBtn");
const aiSettingsBtn = document.getElementById("aiSettingsBtn");
const dropZone = document.getElementById("dropZone");
const csvFile = document.getElementById("csvFile");
const loadSample = document.getElementById("loadSample");
const predictBtn = document.getElementById("predictBtn");
const inputPreview = document.getElementById("inputPreview");
const sampleStats = document.getElementById("sampleStats");
const errorBox = document.getElementById("errorBox");

const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultContent = document.getElementById("resultContent");
const predBadge = document.getElementById("predBadge");
const confidence = document.getElementById("confidence");
const probChart = document.getElementById("probChart");
const topGenes = document.getElementById("topGenes");
const demoProfileSelect = document.getElementById("demoProfileSelect");
const suppressedGenes = document.getElementById("suppressedGenes");
const insightSummary = document.getElementById("insightSummary");
const insightList = document.getElementById("insightList");

const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const sendChatBtn = document.getElementById("sendChatBtn");
const chatFileInput = document.getElementById("chatFileInput");
const attachFileBtn = document.getElementById("attachFileBtn");
const attachedFilesPreview = document.getElementById("attachedFilesPreview");
const activeProviderBadge = document.getElementById("activeProviderBadge");

// AI Settings Modal Elements
const aiSettingsModal = document.getElementById("aiSettingsModal");
const closeAiModalBtn = document.getElementById("closeAiModalBtn");
const saveAiSettingsBtn = document.getElementById("saveAiSettingsBtn");
const resetAiSettingsBtn = document.getElementById("resetAiSettingsBtn");
const aiProviderSelect = document.getElementById("aiProviderSelect");
const apiKeyGroup = document.getElementById("apiKeyGroup");
const aiApiKeyInput = document.getElementById("aiApiKeyInput");
const aiModelGroup = document.getElementById("aiModelGroup");
const aiModelInput = document.getElementById("aiModelInput");
const aiBaseUrlGroup = document.getElementById("aiBaseUrlGroup");
const aiBaseUrlInput = document.getElementById("aiBaseUrlInput");
const aiConfigStatus = document.getElementById("aiConfigStatus");

let pcaChartInstance = null;

const cohortColors = {
  "BRCA": "#ec4899", // Pink
  "KIRC": "#3b82f6", // Blue
  "COAD": "#10b981", // Green
  "LUAD": "#f59e0b", // Orange
  "PRAD": "#8b5cf6"  // Purple

};

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.textContent = "";
  errorBox.hidden = true;
}

function setSample(values, source) {
  clearPrediction();
  state.sample = values;
  state.sampleStats = summarizeSample(values);
  const expected = state.features.length;
  const isMatch = expected > 0 && (values.length === expected || values.length === 20531);
  predictBtn.disabled = values.length === 0 || !isMatch;

  const countNote = values.length === 20531
    ? ` (full raw profile, mapped to ${expected})`
    : (expected ? ` (expected ${expected})` : "");
  inputPreview.textContent = `${source}\nFeatures: ${values.length}${countNote}\nPreview: ${values.slice(0, 12).map((v) => Number(v).toFixed(4)).join(", ")}`;

  renderSampleStats(state.sampleStats, expected);
  if (expected && !isMatch) {
    showError(`Feature count mismatch: got ${values.length}, expected ${expected} (or 20,531 raw features).`);
  }
}

function clearPrediction() {
  state.currentPrediction = null;
  state.patientCoords = null;
  state.chatHistory = [];
  
  if (state.attachedFiles.length === 0) {
    chatInput.disabled = true;
    sendChatBtn.disabled = true;
  }
  chatInput.value = "";
  
  chatMessages.innerHTML = `
    <div class="chat-bubble model">
      Hello! I am your AI clinical assistant. Load a sample and click **Predict** to populate its context. You can also click 📎 to attach clinical pathology notes (.txt, .md), genomic variant files (.json), or expression tables (.csv, .tsv) for multi-modal synthesis.
    </div>
  `;
  
  resultPlaceholder.hidden = false;
  resultContent.hidden = true;
  
  if (pcaChartInstance) {
    pcaChartInstance.data.datasets = getChartDatasets();
    pcaChartInstance.update();
  }
}

async function loadPcaReference() {
  try {
    const res = await fetch("/pca_reference");
    state.pcaReferences = await res.json();
  } catch (error) {
    console.error("Failed to load PCA references:", error);
  }
}

async function refreshStatus() {
  clearError();
  const healthResponse = await fetch("/health");
  const health = await healthResponse.json();
  statusText.textContent = health.detail || "Ready.";

  const featureResponse = await fetch("/features");
  const featureData = await featureResponse.json();
  state.features = featureData.feature_names || [];

  await loadPcaReference();
  initPcaChart();
}

function parseCsvRowToNumbers(row) {
  return row.map((cell) => Number(cell)).filter((value) => Number.isFinite(value));
}

function parseCsvByFeatureHeaders(rows) {
  if (rows.length < 2 || state.features.length === 0) {
    return null;
  }
  const header = rows[0].map((cell) => cell.trim());
  const headerSet = new Set(header);
  const overlap = state.features.filter((feature) => headerSet.has(feature)).length;
  if (overlap < Math.min(10, Math.floor(state.features.length * 0.3))) {
    return null;
  }

  const valueRow = rows[1];
  const indexByGene = new Map();
  header.forEach((name, idx) => {
    if (name) indexByGene.set(name, idx);
  });
  const orderedValues = state.features.map((gene) => {
    const idx = indexByGene.get(gene);
    if (idx === undefined) return NaN;
    return Number(valueRow[idx]);
  });
  if (orderedValues.some((value) => !Number.isFinite(value))) {
    throw new Error("CSV header matched features, but values are missing/non-numeric for at least one gene.");
  }
  return orderedValues;
}

function parseCsvFirstNumericRow(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) {
    throw new Error("CSV is empty.");
  }

  const rows = lines.map((line) => line.split(",").map((cell) => cell.trim()));
  const byHeader = parseCsvByFeatureHeaders(rows);
  if (byHeader) return byHeader;

  const numericRows = rows.map(parseCsvRowToNumbers).filter((row) => row.length > 0);

  if (!numericRows.length) {
    throw new Error("No numeric values found in CSV.");
  }
  return numericRows[0];
}

function summarizeSample(values) {
  const count = values.length;
  const arr = values.map(Number).filter((value) => Number.isFinite(value));
  if (!arr.length) {
    return { count, finite: 0, min: NaN, max: NaN, mean: NaN, std: NaN, zeros: 0 };
  }
  const mean = arr.reduce((acc, value) => acc + value, 0) / arr.length;
  const variance = arr.reduce((acc, value) => acc + (value - mean) ** 2, 0) / arr.length;
  return {
    count,
    finite: arr.length,
    min: Math.min(...arr),
    max: Math.max(...arr),
    mean,
    std: Math.sqrt(variance),
    zeros: arr.filter((value) => value === 0).length,
  };
}

function formatNum(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "-";
}

function renderSampleStats(stats, expectedCount) {
  if (!stats) {
    sampleStats.innerHTML = "";
    return;
  }
  const rows = [
    ["Features", `${stats.count}${expectedCount ? ` / ${expectedCount}` : ""}`],
    ["Mean", formatNum(stats.mean)],
    ["Std Dev", formatNum(stats.std)],
    ["Min", formatNum(stats.min)],
    ["Max", formatNum(stats.max)],
    ["Zeros", `${stats.zeros}`],
  ];
  sampleStats.innerHTML = rows
    .map(
      ([label, value]) => `
      <div class="stat-tile">
        <div class="stat-label">${label}</div>
        <div class="stat-value">${value}</div>
      </div>`
    )
    .join("");
}

// DROPACTION IMPLEMENTATION
function setupDropZone() {
  dropZone.addEventListener("click", () => csvFile.click());
  dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      csvFile.click();
    }
  });
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("active");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("active");
    });
  });
  dropZone.addEventListener("drop", async (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    await handleFile(file);
  });
}

async function handleFile(file) {
  clearError();
  try {
    const text = await file.text();
    const values = parseCsvFirstNumericRow(text);
    setSample(values, file.name);
  } catch (error) {
    showError(error.message);
  }
}

csvFile.addEventListener("change", async () => {
  const file = csvFile.files[0];
  if (!file) return;
  await handleFile(file);
});

let demoProfilesCache = null;

async function fetchDemoProfiles() {
  if (demoProfilesCache) return demoProfilesCache;
  try {
    const res = await fetch("/demo_profiles");
    if (res.ok) {
      demoProfilesCache = await res.json();
      return demoProfilesCache;
    }
  } catch (err) {
    console.warn("Could not fetch /demo_profiles, trying static file", err);
  }
  try {
    const res = await fetch("/frontend/demo_profiles.json");
    demoProfilesCache = await res.json();
    return demoProfilesCache;
  } catch (e) {
    console.error("Failed to load demo profiles", e);
    return null;
  }
}

loadSample.addEventListener("click", async () => {
  clearError();
  const selectedKey = demoProfileSelect ? demoProfileSelect.value : "brca";
  const profiles = await fetchDemoProfiles();
  if (profiles && profiles[selectedKey]) {
    const p = profiles[selectedKey];
    setSample(p.gene_values, `${p.name} — ${p.description}`);
  } else {
    showError("Could not load curated demo profile. Please ensure demo_profiles.json is generated.");
  }
});

reloadBtn.addEventListener("click", async () => {
  clearError();
  await fetch("/reload", { method: "POST" });
  await refreshStatus();
});

// PREDICT PIPELINE
predictBtn.addEventListener("click", async () => {
  clearError();
  if (!state.sample) {
    showError("Load a sample first.");
    return;
  }

  try {
    predictBtn.disabled = true;
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_id: "frontend_sample", gene_values: state.sample }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    state.currentPrediction = payload;
    state.patientCoords = payload.pca_coords;
    
    renderResult(payload);
    
    // Update scatter plot with patient coordinates
    if (pcaChartInstance) {
      pcaChartInstance.data.datasets = getChartDatasets();
      pcaChartInstance.update();
    }

    // Enable Chat Input
    chatInput.disabled = false;
    sendChatBtn.disabled = false;
  } catch (error) {
    showError(error.message);
  } finally {
    predictBtn.disabled = false;
  }
});

function renderResult(result) {
  resultPlaceholder.hidden = true;
  resultContent.hidden = false;

  predBadge.textContent = result.predicted_class;
  confidence.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;

  probChart.innerHTML = "";
  Object.entries(result.class_probabilities).forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span>${label}</span>
      <div class="bar-track"><div class="bar-fill" style="width: ${Math.max(0, Math.min(100, value * 100))}%"></div></div>
      <span>${(value * 100).toFixed(1)}%</span>
    `;
    probChart.appendChild(row);
  });

  topGenes.innerHTML = "";
  (result.top_features || []).forEach((item) => {
    const row = document.createElement("div");
    row.className = "gene-row";
    const sign = item.attribution > 0 ? "+" : "";
    const attrText = item.attribution !== undefined && item.attribution !== null ? ` (${sign}${item.attribution.toFixed(3)})` : "";
    row.innerHTML = `<span class="gene-name">${item.gene}${attrText}</span><span class="gene-value">${item.expression}</span>`;
    topGenes.appendChild(row);
  });

  if (suppressedGenes) {
    suppressedGenes.innerHTML = "";
    if (result.suppressed_features && result.suppressed_features.length > 0) {
      result.suppressed_features.forEach((item) => {
        const row = document.createElement("div");
        row.className = "gene-row";
        const sign = item.attribution > 0 ? "+" : "";
        const attrText = item.attribution !== undefined && item.attribution !== null ? ` (${sign}${item.attribution.toFixed(3)})` : "";
        row.innerHTML = `<span class="gene-name" style="color: #ef4444;">${item.gene}${attrText}</span><span class="gene-value">${item.expression}</span>`;
        suppressedGenes.appendChild(row);
      });
    } else {
      suppressedGenes.innerHTML = '<div style="font-size: 0.8rem; color: #94a3b8; padding: 4px;">No significant suppressors</div>';
    }
  }

  renderInsights(result);
}

function renderInsights(result) {
  const probabilities = Object.entries(result.class_probabilities).sort((a, b) => b[1] - a[1]);
  const top1 = probabilities[0] || ["N/A", 0];
  const top2 = probabilities[1] || ["N/A", 0];
  const margin = top1[1] - top2[1];

  let confidenceBand = "high";
  if (result.confidence < 0.75) confidenceBand = "low";
  else if (result.confidence < 0.9) confidenceBand = "moderate";

  const summary = result.low_confidence
    ? `AI interpretation: the model leans toward ${result.predicted_class}, but confidence is below the reliability threshold.`
    : `AI interpretation: the sample most strongly matches ${result.predicted_class} with ${confidenceBand} confidence.`;

  const topGeneSignal = (result.top_features || [])
    .slice(0, 3)
    .map((entry) => `${entry.gene} (${entry.expression})`)
    .join(", ");

  const suppressedSignal = (result.suppressed_features || [])
    .slice(0, 2)
    .map((entry) => `${entry.gene} (${entry.expression})`)
    .join(", ");

  const statSignals = state.sampleStats
    ? `Input spread looks ${state.sampleStats.std > 1.2 ? "broad" : "compact"} (std ${formatNum(state.sampleStats.std)}), with ${state.sampleStats.zeros} zero-value features.`
    : "Input summary is unavailable.";

  const insights = [
    `Class separation: top class ${top1[0]} ${(top1[1] * 100).toFixed(1)}% vs second ${top2[0]} ${(top2[1] * 100).toFixed(1)}% (margin ${(margin * 100).toFixed(1)}%).`,
    statSignals,
    `Top activating drivers: ${topGeneSignal || "none available"}.`,
  ];

  if (suppressedSignal) {
    insights.push(`Suppressed biomarkers / tumor suppressors: ${suppressedSignal}.`);
  }

  if (result.warning) {
    insights.unshift(result.warning);
  }

  insightSummary.textContent = summary;
  insightList.innerHTML = insights.map((item) => `<div class="insight-item">${item}</div>`).join("");
}

// CHART MANAGEMENT
function getChartDatasets() {
  const datasets = [];

  // Group reference points by label
  const groups = {};
  state.pcaReferences.forEach(pt => {
    if (!groups[pt.label]) groups[pt.label] = [];
    groups[pt.label].push({ x: pt.x, y: pt.y });
  });

  // Add reference cohorts
  Object.keys(groups).forEach(label => {
    datasets.push({
      label: label,
      data: groups[label],
      backgroundColor: cohortColors[label] || "#64748b",
      pointRadius: 4,
      pointHoverRadius: 6,
      opacity: 0.6
    });
  });

  // Add patient sample if it exists
  if (state.patientCoords) {
    datasets.push({
      label: "Patient Sample",
      data: [state.patientCoords],
      backgroundColor: "#e11d48", // Crimson red
      borderColor: "#ffffff",
      borderWidth: 2,
      pointRadius: 10,
      pointStyle: "rectRot", // rot square
      pointHoverRadius: 12,
      showLine: false
    });
  }

  return datasets;
}

function initPcaChart() {
  const canvas = document.getElementById("pcaChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (pcaChartInstance) {
    pcaChartInstance.destroy();
  }

  const isDark = document.body.getAttribute("data-theme") === "dark";
  const textColor = isDark ? "#94a3b8" : "#64748b";
  const gridColor = isDark ? "#2d3b55" : "#e2e8f0";

  pcaChartInstance = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: getChartDatasets()
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          labels: {
            color: textColor,
            font: { family: "Inter", weight: "600", size: 11 }
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `${context.dataset.label}: (${context.parsed.x.toFixed(2)}, ${context.parsed.y.toFixed(2)})`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: "Inter" } },
          title: { display: true, text: "PCA Component 1", color: textColor, font: { weight: "600", family: "Inter" } }
        },
        y: {
          grid: { color: gridColor },
          ticks: { color: textColor, font: { family: "Inter" } },
          title: { display: true, text: "PCA Component 2", color: textColor, font: { weight: "600", family: "Inter" } }
        }
      }
    }
  });
}

function updateChartColors() {
  if (!pcaChartInstance) return;
  const isDark = document.body.getAttribute("data-theme") === "dark";
  const textColor = isDark ? "#94a3b8" : "#64748b";
  const gridColor = isDark ? "#2d3b55" : "#e2e8f0";

  pcaChartInstance.options.plugins.legend.labels.color = textColor;
  pcaChartInstance.options.scales.x.grid.color = gridColor;
  pcaChartInstance.options.scales.x.ticks.color = textColor;
  pcaChartInstance.options.scales.x.title.color = textColor;
  pcaChartInstance.options.scales.y.grid.color = gridColor;
  pcaChartInstance.options.scales.y.ticks.color = textColor;
  pcaChartInstance.options.scales.y.title.color = textColor;

  pcaChartInstance.data.datasets = getChartDatasets();
  pcaChartInstance.update();
}

// CHAT ASSISTANT
let bubbleCounter = 0;
function appendChatBubble(role, text, providerUsed = null, structuredInsights = null) {
  const bubbleId = `bubble-${bubbleCounter++}`;
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  bubble.id = bubbleId;

  if (role === "model" && text !== "Analyzing...") {
    let contentHtml = "";
    if (providerUsed) {
      contentHtml += `<span class="provider-tag">${providerUsed}</span><br>`;
    }
    contentHtml += formatMarkdown(text);

    if (structuredInsights && structuredInsights.ihc_recommendations && structuredInsights.ihc_recommendations.length > 0) {
      const ihcTags = structuredInsights.ihc_recommendations.slice(0, 5).map(m => `<code>${m}</code>`).join(" ");
      contentHtml += `
        <div class="chat-structured-card">
          <div class="card-title">🔬 Confirmatory IHC Recommendations</div>
          <div>${ihcTags}</div>
        </div>
      `;
    }
    bubble.innerHTML = contentHtml;
  } else {
    bubble.textContent = text;
  }

  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return bubbleId;
}

function removeChatBubble(id) {
  const bubble = document.getElementById(id);
  if (bubble) bubble.remove();
}

function formatMarkdown(text) {
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/^### (.*?)$/gm, '<h4 style="margin: 6px 0 4px; color: var(--accent);">$1</h4>')
    .replace(/^## (.*?)$/gm, '<h3 style="margin: 8px 0 6px;">$1</h3>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');

  // Format blockquotes / notices
  html = html.replace(/(?:^|<br>)&gt;\s*(.*?)(?=<br>|$)/g, '<div class="chat-notice">$1</div>');

  // Format bullet lists
  html = html.replace(/(?:^|<br>)\s*-\s+(.*?)(?=<br>|$)/g, '<li>$1</li>');
  if (html.includes('<li>')) {
    html = html.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
  }
  return html;
}

async function sendChatMessage() {
  const message = chatInput.value.trim();
  if (!message && state.attachedFiles.length === 0) return;

  const userQuery = message || "Please interpret the attached file evidence in the clinical context.";
  chatInput.value = "";
  appendChatBubble("user", userQuery);
  const priorHistory = [...state.chatHistory];
  state.chatHistory.push({ role: "user", content: userQuery });

  const loadingId = appendChatBubble("model", "Analyzing...");

  try {
    const payloadBody = {
      message: userQuery,
      history: priorHistory,
      prediction: state.currentPrediction,
      attached_files: state.attachedFiles.map(f => ({
        filename: f.filename,
        content: f.content,
        file_type: f.file_type
      })),
      provider_config: state.aiConfig
    };

    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadBody)
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Failed to contact AI assistant.");
    }

    removeChatBubble(loadingId);
    appendChatBubble("model", payload.reply, payload.provider_used, payload.structured_insights);
    state.chatHistory.push({ role: "model", content: payload.reply });

    // Clear attached files after successful chat submission
    state.attachedFiles = [];
    renderAttachedFilesPreview();
  } catch (error) {
    removeChatBubble(loadingId);
    appendChatBubble("model", `Error: ${error.message}`);
  }
}

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    sendChatMessage();
  }
});
sendChatBtn.addEventListener("click", sendChatMessage);

// ATTACHED FILES HANDLING
function setupChatAttachments() {
  if (!attachFileBtn || !chatFileInput) return;

  attachFileBtn.addEventListener("click", () => {
    chatFileInput.click();
  });

  chatFileInput.addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    for (const file of files) {
      if (file.size > 5 * 1024 * 1024) {
        showError(`File "${file.name}" exceeds maximum allowed size of 5 MB.`);
        continue;
      }

      const ext = file.name.split(".").pop().toLowerCase();
      try {
        const textContent = await file.text();
        state.attachedFiles.push({
          filename: file.name,
          content: textContent,
          file_type: ext
        });
      } catch (err) {
        showError(`Failed to read file ${file.name}: ${err.message}`);
      }
    }

    chatFileInput.value = "";
    renderAttachedFilesPreview();

    // Enable chat input if files are attached
    if (state.attachedFiles.length > 0) {
      chatInput.disabled = false;
      sendChatBtn.disabled = false;
      chatInput.focus();
    }
  });
}

function renderAttachedFilesPreview() {
  if (!attachedFilesPreview) return;
  if (state.attachedFiles.length === 0) {
    attachedFilesPreview.hidden = true;
    attachedFilesPreview.innerHTML = "";
    return;
  }

  attachedFilesPreview.hidden = false;
  attachedFilesPreview.innerHTML = state.attachedFiles.map((f, idx) => `
    <div class="file-attachment-chip" data-idx="${idx}">
      <span class="file-chip-name" title="${f.filename}">📄 ${f.filename}</span>
      <span class="file-chip-type">${f.file_type}</span>
      <button type="button" class="file-chip-remove" data-idx="${idx}" title="Remove file">&times;</button>
    </div>
  `).join("");

  attachedFilesPreview.querySelectorAll(".file-chip-remove").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.getAttribute("data-idx"), 10);
      state.attachedFiles.splice(idx, 1);
      renderAttachedFilesPreview();
      if (state.attachedFiles.length === 0 && !state.currentPrediction) {
        chatInput.disabled = true;
        sendChatBtn.disabled = true;
      }
    });
  });
}

// QUICK PROMPT ACTION CHIPS
function setupQuickChips() {
  document.querySelectorAll(".chip-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const prompt = btn.getAttribute("data-prompt");
      chatInput.value = prompt;
      if (!chatInput.disabled) {
        sendChatMessage();
      } else {
        showError("Please load a patient profile or attach a clinical file first.");
      }
    });
  });
}

// AI CONFIGURATION & MODAL
function loadAiConfig() {
  try {
    const saved = localStorage.getItem("detectai_ai_config");
    if (saved) {
      state.aiConfig = Object.assign({}, state.aiConfig, JSON.parse(saved));
    }
  } catch (err) {
    console.warn("Failed to load saved AI config:", err);
  }
  updateProviderBadge();
}

function updateProviderBadge() {
  if (!activeProviderBadge) return;
  const nameMap = {
    offline: "Offline Bio-Agent",
    gemini: "Google Gemini",
    openai: "OpenAI",
    anthropic: "Claude",
    openai_compatible: "OpenAI-Compatible"
  };
  const providerKey = state.aiConfig.provider_name || "auto";
  if (providerKey === "auto") {
    const def = state.serverDefaultProvider || "offline";
    activeProviderBadge.textContent = `${nameMap[def] || def} (Auto)`;
  } else {
    activeProviderBadge.textContent = nameMap[providerKey] || providerKey;
  }
}

async function fetchAiProviders() {
  try {
    const res = await fetch("/ai/providers");
    if (res.ok) {
      const data = await res.json();
      state.availableProviders = data.providers || [];
      state.serverDefaultProvider = data.active_default || "offline";
      updateProviderBadge();
    }
  } catch (err) {
    console.warn("Could not fetch available AI providers:", err);
  }
}

function updateAiModalFields() {
  const selected = aiProviderSelect.value;
  if (selected === "auto") {
    apiKeyGroup.style.display = "none";
    aiBaseUrlGroup.style.display = "none";
    aiModelGroup.style.display = "none";
  } else if (selected === "offline") {
    apiKeyGroup.style.display = "none";
    aiBaseUrlGroup.style.display = "none";
    aiModelGroup.style.display = "none";
  } else if (selected === "openai_compatible") {
    apiKeyGroup.style.display = "block";
    aiBaseUrlGroup.style.display = "block";
    aiModelGroup.style.display = "block";
  } else {
    apiKeyGroup.style.display = "block";
    aiBaseUrlGroup.style.display = "none";
    aiModelGroup.style.display = "block";
  }
}

function openAiModal() {
  aiProviderSelect.value = state.aiConfig.provider_name || "auto";
  aiApiKeyInput.value = state.aiConfig.api_key || "";
  aiModelInput.value = state.aiConfig.model_name || "";
  aiBaseUrlInput.value = state.aiConfig.base_url || "";
  aiConfigStatus.style.display = "none";
  updateAiModalFields();
  aiSettingsModal.hidden = false;
}

function closeAiModal() {
  aiSettingsModal.hidden = true;
}

function saveAiConfig() {
  state.aiConfig = {
    provider_name: aiProviderSelect.value,
    api_key: aiApiKeyInput.value.trim(),
    model_name: aiModelInput.value.trim(),
    base_url: aiBaseUrlInput.value.trim()
  };
  localStorage.setItem("detectai_ai_config", JSON.stringify(state.aiConfig));
  updateProviderBadge();
  closeAiModal();
}

function resetAiConfig() {
  state.aiConfig = {
    provider_name: "auto",
    api_key: "",
    model_name: "",
    base_url: ""
  };
  localStorage.removeItem("detectai_ai_config");
  updateProviderBadge();
  closeAiModal();
}

if (aiSettingsBtn) aiSettingsBtn.addEventListener("click", openAiModal);
if (closeAiModalBtn) closeAiModalBtn.addEventListener("click", closeAiModal);
if (saveAiSettingsBtn) saveAiSettingsBtn.addEventListener("click", saveAiConfig);
if (resetAiSettingsBtn) resetAiSettingsBtn.addEventListener("click", resetAiConfig);
if (aiProviderSelect) aiProviderSelect.addEventListener("change", updateAiModalFields);
if (aiSettingsModal) {
  aiSettingsModal.addEventListener("click", (e) => {
    if (e.target === aiSettingsModal) closeAiModal();
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !aiSettingsModal.hidden) closeAiModal();
  });
}

// THEME SELECTION
function initTheme() {
  const savedTheme = localStorage.getItem("theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);
  document.body.setAttribute("data-theme", savedTheme);
  themeToggleBtn.textContent = savedTheme === "dark" ? "☀️ Light" : "🌙 Dark";
}

themeToggleBtn.addEventListener("click", () => {
  const currentTheme = document.documentElement.getAttribute("data-theme");
  const nextTheme = currentTheme === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", nextTheme);
  document.body.setAttribute("data-theme", nextTheme);
  localStorage.setItem("theme", nextTheme);
  themeToggleBtn.textContent = nextTheme === "dark" ? "☀️ Light" : "🌙 Dark";
  updateChartColors();
});

// STARTUP
initTheme();
loadAiConfig();
fetchAiProviders();
setupChatAttachments();
setupQuickChips();
setupDropZone();
refreshStatus().catch((error) => showError(error.message));
