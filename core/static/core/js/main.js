/**
 * main.js — DIP Lab Tool shared utilities
 */

// ─── Spinner ──────────────────────────────────────────────────────────────
function showSpinner() {
  document.getElementById('globalSpinner').classList.remove('d-none');
}

function hideSpinner() {
  document.getElementById('globalSpinner').classList.add('d-none');
}

// ─── AJAX helper ─────────────────────────────────────────────────────────
async function postAPI(url, body) {
  try {
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value
      || (typeof CSRF !== 'undefined' ? CSRF : '');
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrfToken,
      },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
      return err;
    }
    return await response.json();
  } catch (err) {
    return { error: err.message || 'Network error' };
  }
}

// ─── Upload zone setup ────────────────────────────────────────────────────
function setupUpload(zoneId, inputId, callback) {
  const zone = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => {
    zone.classList.remove('drag-over');
  });

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      readImageFile(file, callback);
    }
  });

  input.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) readImageFile(file, callback);
  });
}

function readImageFile(file, callback) {
  const reader = new FileReader();
  reader.onload = (e) => {
    callback(e.target.result);
  };
  reader.readAsDataURL(file);
}

// ─── Download helper ──────────────────────────────────────────────────────
function downloadImage(dataUrl, filename = 'result.png') {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ─── Sidebar toggle (mobile) ──────────────────────────────────────────────
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebar = document.getElementById('sidebar');

if (sidebarToggle && sidebar) {
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });

  // Close sidebar when clicking outside on mobile
  document.addEventListener('click', (e) => {
    if (window.innerWidth < 768 &&
        !sidebar.contains(e.target) &&
        !sidebarToggle.contains(e.target)) {
      sidebar.classList.remove('open');
    }
  });
}

// ─── Theory modal helper ──────────────────────────────────────────────────
function openTheoryModal(title, bodyHtml) {
  const titleEl = document.getElementById('theoryModalTitle');
  const bodyEl = document.getElementById('theoryModalBody');
  if (titleEl) titleEl.textContent = title;
  if (bodyEl) bodyEl.innerHTML = bodyHtml;
  const modal = new bootstrap.Modal(document.getElementById('theoryModal'));
  modal.show();
  if (window.MathJax) {
    MathJax.typesetPromise([bodyEl]).catch(console.error);
  }
}

// ─── Chart helpers ────────────────────────────────────────────────────────
function makeBarChart(canvasId, labels, data, label, color = '#4f6ef7') {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label,
        data,
        backgroundColor: color,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } },
    },
  });
}

function makeLineChart(canvasId, labels, datasets) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: false } },
    },
  });
}

// ─── Metrics display helper ───────────────────────────────────────────────
function displayMetrics(metrics, containerId) {
  const el = document.getElementById(containerId);
  if (!el || !metrics) return;
  el.innerHTML = `
    <div class="row g-2">
      <div class="col-4"><div class="metric-badge">PSNR: <strong>${metrics.psnr ?? '—'} dB</strong></div></div>
      <div class="col-4"><div class="metric-badge">MSE: <strong>${metrics.mse ?? '—'}</strong></div></div>
      <div class="col-4"><div class="metric-badge">SSIM: <strong>${metrics.ssim ?? '—'}</strong></div></div>
    </div>`;
  el.classList.remove('d-none');
}

// ─── Image comparison viewer ──────────────────────────────────────────────
function showSideBySide(origSrc, resultSrc, origLabel = 'Original', resultLabel = 'Processed', containerId = 'resultArea') {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.className = 'row g-3';
  el.innerHTML = `
    <div class="col-6">
      <p class="small text-muted mb-1">${origLabel}</p>
      <img src="${origSrc}" class="img-fluid rounded border" style="width:100%">
    </div>
    <div class="col-6">
      <p class="small text-muted mb-1">${resultLabel}</p>
      <img src="${resultSrc}" class="img-fluid rounded border" style="width:100%">
      <div class="mt-2">
        <button class="btn btn-sm btn-success" onclick="downloadImage('${resultSrc}','result.png')">
          <i class="bi bi-download me-1"></i>Download
        </button>
      </div>
    </div>`;
}
