const state = {
  features: [],
  sample: null,
  sampleStats: null,
};

const statusText = document.getElementById("statusText");
const reloadBtn = document.getElementById("reloadBtn");
const dropZone = document.getElementById("dropZone");
const csvFile = document.getElementById("csvFile");
const loadSample = document.getElementById("loadSample");
const predictBtn = document.getElementById("predictBtn");
const inputPreview = document.getElementById("inputPreview");
const sampleStats = document.getElementById("sampleStats");
const errorBox = document.getElementById("errorBox");
const resultCard = document.getElementById("resultCard");
const predBadge = document.getElementById("predBadge");
const confidence = document.getElementById("confidence");
const probChart = document.getElementById("probChart");
const topGenes = document.getElementById("topGenes");
const insightSummary = document.getElementById("insightSummary");
const insightList = document.getElementById("insightList");

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.textContent = "";
  errorBox.hidden = true;
}

function setSample(values, source) {
  state.sample = values;
  state.sampleStats = summarizeSample(values);
  const expected = state.features.length;
  const exactMatch = expected > 0 && values.length === expected;
  predictBtn.disabled = values.length === 0 || !exactMatch;

  inputPreview.textContent = `${source}
Features: ${values.length}${expected ? ` (expected ${expected})` : ""}
Preview: ${values.slice(0, 12).map((v) => Number(v).toFixed(4)).join(", ")}`;

  renderSampleStats(state.sampleStats, expected);
  if (expected && !exactMatch) {
    showError(`Feature count mismatch: got ${values.length}, expected ${expected}.`);
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
    ["Feature Count", `${stats.count}${expectedCount ? ` / ${expectedCount}` : ""}`],
    ["Mean", formatNum(stats.mean)],
    ["Std", formatNum(stats.std)],
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

loadSample.addEventListener("click", () => {
  clearError();
  if (!state.features.length) {
    showError("No feature list is loaded yet. Train the model first, then reload artifacts.");
    return;
  }
  const values = state.features.map((_, index) => {
    const wave = Math.sin(index / 19) * 0.6;
    const trend = (index % 17) / 30;
    return Number((wave + trend).toFixed(5));
  });
  setSample(values, "Generated demo sample");
});

reloadBtn.addEventListener("click", async () => {
  clearError();
  await fetch("/reload", { method: "POST" });
  await refreshStatus();
});

predictBtn.addEventListener("click", async () => {
  clearError();
  resultCard.hidden = true;
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

    renderResult(payload);
  } catch (error) {
    showError(error.message);
  } finally {
    predictBtn.disabled = false;
  }
});

function renderResult(result) {
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
  result.top_features.forEach((item) => {
    const row = document.createElement("div");
    row.className = "gene-row";
    row.innerHTML = `<span class="gene-name">${item.gene}</span><span class="gene-value">${item.expression}</span>`;
    topGenes.appendChild(row);
  });

  renderInsights(result);
  resultCard.hidden = false;
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

  const statSignals = state.sampleStats
    ? `Input spread looks ${state.sampleStats.std > 1.2 ? "broad" : "compact"} (std ${formatNum(state.sampleStats.std)}), with ${state.sampleStats.zeros} zero-value features.`
    : "Input summary is unavailable.";

  const insights = [
    `Class separation: top class ${top1[0]} ${(top1[1] * 100).toFixed(1)}% vs second ${top2[0]} ${(top2[1] * 100).toFixed(1)}% (margin ${(margin * 100).toFixed(1)}%).`,
    statSignals,
    `Top expression drivers in this sample: ${topGeneSignal || "none available"}.`,
  ];

  if (result.warning) {
    insights.unshift(result.warning);
  }

  insightSummary.textContent = summary;
  insightList.innerHTML = insights.map((item) => `<div class="insight-item">${item}</div>`).join("");
}

setupDropZone();
refreshStatus().catch((error) => showError(error.message));
