const API_BASE = window.API_BASE || "http://localhost:8000";
const form = document.getElementById("uploadForm");
const statusEl = document.getElementById("status");
const resultSection = document.getElementById("resultSection");
const resultTable = document.getElementById("resultTable");
const rawCsv = document.getElementById("rawCsv");
const downloadDiv = document.getElementById("download");
const submitBtn = document.getElementById("submitBtn");
const fileInput = document.getElementById("fileInput");
const fileInfo = document.getElementById("fileInfo");
const uploadArea = document.getElementById("uploadArea");
const progressContainer = document.getElementById("progressContainer");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");

async function checkHealth() {
  try {
    const r = await fetch(API_BASE + "/health");
    if (r.ok) {
      const j = await r.json();
      statusEl.innerHTML = `<span style="color: #48bb78;">✅ System Ready</span> • Model: <code>${j.model}</code>`;
    }
  } catch (e) {
    statusEl.innerHTML =
      '<span style="color: #f56565;">❌ Cannot reach backend</span>';
  }
}
checkHealth();

// File input change handler
fileInput.addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (file) {
    displayFileInfo(file);
    submitBtn.disabled = false;
  } else {
    hideFileInfo();
    submitBtn.disabled = true;
  }
});

// Drag and drop functionality
uploadArea.addEventListener("dragover", function (e) {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", function (e) {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", function (e) {
  e.preventDefault();
  uploadArea.classList.remove("dragover");

  const files = e.dataTransfer.files;
  if (files.length > 0 && files[0].type === "application/pdf") {
    fileInput.files = files;
    displayFileInfo(files[0]);
    submitBtn.disabled = false;
  }
});

function displayFileInfo(file) {
  const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
  fileInfo.innerHTML = `
    <h4>📄 ${file.name}</h4>
    <p>Size: ${sizeInMB} MB • Type: PDF Document</p>
  `;
  fileInfo.classList.remove("hidden");
}

function hideFileInfo() {
  fileInfo.classList.add("hidden");
}

function showProgress(text = "Processing...") {
  progressText.textContent = text;
  progressContainer.classList.remove("hidden");
}

function updateProgress(percentage, text) {
  progressFill.style.width = percentage + "%";
  if (text) progressText.textContent = text;
}

function hideProgress() {
  progressContainer.classList.add("hidden");
  progressFill.style.width = "0%";
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) {
    return;
  }

  submitBtn.disabled = true;
  resultSection.classList.add("hidden");

  // Show progress
  showProgress("Uploading file...");
  updateProgress(10, "Uploading file...");

  statusEl.innerHTML =
    '<span style="color: #667eea;">📤 Uploading and processing your PDF...</span>';

  const fd = new FormData();
  fd.append("file", fileInput.files[0]);

  // Simulate progress updates
  setTimeout(() => updateProgress(30, "Extracting text from PDF..."), 500);
  setTimeout(() => updateProgress(60, "Analyzing with AI model..."), 1500);
  setTimeout(() => updateProgress(80, "Generating alloy data table..."), 3000);

  let resp;
  try {
    resp = await fetch(API_BASE + "/extract", { method: "POST", body: fd });
  } catch (err) {
    statusEl.innerHTML =
      '<span style="color: #f56565;">❌ Network error: ' + err + "</span>";
    hideProgress();
    submitBtn.disabled = false;
    return;
  }

  if (!resp.ok) {
    let txt = await resp.text();
    statusEl.innerHTML =
      '<span style="color: #f56565;">❌ Server error: ' +
      resp.status +
      "</span>";
    hideProgress();
    submitBtn.disabled = false;
    return;
  }

  updateProgress(95, "Finalizing results...");

  const data = await resp.json();

  setTimeout(() => {
    updateProgress(100, "Complete!");
    statusEl.innerHTML =
      '<span style="color: #48bb78;">✅ Extraction completed successfully!</span>';

    renderTable(data.csv_text);
    rawCsv.textContent = data.csv_text;
    downloadDiv.innerHTML = `<a href="${data.download_url}" download class="dl">📥 Download CSV File</a>`;

    resultSection.classList.remove("hidden");
    hideProgress();
    submitBtn.disabled = false;

    // Reset form
    setTimeout(() => {
      fileInput.value = "";
      hideFileInfo();
      submitBtn.disabled = true;
    }, 1000);
  }, 500);
});

function renderTable(csv) {
  if (!csv) {
    resultTable.innerHTML = "";
    return;
  }
  const lines = csv.split(/\n/).filter((l) => l.trim());
  if (!lines.length) {
    resultTable.innerHTML = "";
    return;
  }
  const header = lines[0].split("|").map((h) => h.trim());
  let thead =
    "<thead><tr>" +
    header.map((h) => `<th>${escapeHtml(h)}</th>`).join("") +
    "</tr></thead>";
  let bodyRows = "";
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split("|");
    if (cols.length !== header.length) continue;
    bodyRows +=
      "<tr>" +
      cols.map((c) => `<td>${escapeHtml(c.trim())}</td>`).join("") +
      "</tr>";
  }
  resultTable.innerHTML = thead + "<tbody>" + bodyRows + "</tbody>";
}

function escapeHtml(str) {
  return str.replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[
        c
      ] || c)
  );
}
