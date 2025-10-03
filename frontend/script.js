const API_BASE = window.API_BASE || 'http://localhost:8000'; // Adjust if backend runs elsewhere
const form = document.getElementById('uploadForm');
const statusEl = document.getElementById('status');
const resultSection = document.getElementById('resultSection');
const resultTable = document.getElementById('resultTable');
const rawCsv = document.getElementById('rawCsv');
const downloadDiv = document.getElementById('download');
const submitBtn = document.getElementById('submitBtn');

async function checkHealth(){
  try{
  const r = await fetch(API_BASE + '/health');
    if(r.ok){ const j = await r.json(); statusEl.innerText = `Ready. Model: ${j.model}`; }
  }catch(e){ statusEl.innerText='Cannot reach backend.'; }
}
checkHealth();

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  const fileInput = document.getElementById('fileInput');
  if(!fileInput.files.length){ return; }
  submitBtn.disabled = true;
  statusEl.textContent = 'Uploading & processing...';
  resultSection.classList.add('hidden');

  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  let resp;
  try {
  resp = await fetch(API_BASE + '/extract', { method:'POST', body: fd });
  } catch(err){
    statusEl.textContent = 'Network error: '+err;
    submitBtn.disabled = false;
    return;
  }
  if(!resp.ok){
    let txt = await resp.text();
    statusEl.textContent = 'Server error: '+resp.status+' '+txt.slice(0,200);
    submitBtn.disabled = false;
    return;
  }
  const data = await resp.json();
  statusEl.textContent = 'Done.';
  renderTable(data.csv_text);
  rawCsv.textContent = data.csv_text;
  downloadDiv.innerHTML = `<a href="${data.download_url}" download class="dl">Download CSV</a>`;
  resultSection.classList.remove('hidden');
  submitBtn.disabled = false;
});

function renderTable(csv){
  if(!csv){ resultTable.innerHTML=''; return; }
  const lines = csv.split(/\n/).filter(l=>l.trim());
  if(!lines.length){ resultTable.innerHTML=''; return; }
  const header = lines[0].split('|').map(h=>h.trim());
  let thead = '<thead><tr>'+header.map(h=>`<th>${escapeHtml(h)}</th>`).join('')+'</tr></thead>';
  let bodyRows = '';
  for(let i=1;i<lines.length;i++){
    const cols = lines[i].split('|');
    if(cols.length!==header.length) continue;
    bodyRows += '<tr>'+cols.map(c=>`<td>${escapeHtml(c.trim())}</td>`).join('')+'</tr>';
  }
  resultTable.innerHTML = thead + '<tbody>' + bodyRows + '</tbody>';
}

function escapeHtml(str){
  return str.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]||c));
}
