// Minimal Progressive ML frontend helpers for RL Dashboard
// Provides: initProgressiveML, getProgressivePrediction, startProgressiveTraining

(function(){
  async function fetchJSON(url, opts){
    const r = await fetch(url, opts||{});
    let j = null;
    try { j = await r.json(); } catch(_) { j = { status: r.ok? 'success':'error', message: await r.text() }; }
    return j;
  }

  async function initProgressiveML(){
    try{
      // Readiness/status line
      const j = await fetchJSON('/api/ml/progressive/status');
      const el = document.getElementById('progressive-ml-status-display');
      if(!el) return;
      if(j && j.status === 'success'){
        const d = j.data || {};
        const parts = [];
        parts.push('Status: ' + (d.status || 'unknown'));
        if(typeof d.jobs_running === 'number') parts.push(`Jobs: ${d.jobs_running}`);
        if(d.last_updated) parts.push(`Updated: ${new Date(d.last_updated).toLocaleString()}`);
        el.textContent = parts.join(' • ');
      } else {
        el.textContent = 'Progressive ML: backend unavailable';
      }
    }catch(_){
      const el = document.getElementById('progressive-ml-status-display');
      if(el) el.textContent = 'Progressive ML: error reading status';
    }
  }

  async function getProgressivePrediction(){
    const sym = (document.getElementById('progressive-symbol')?.value || '').trim().toUpperCase();
    const mode = (document.getElementById('progressive-mode')?.value || 'progressive');
    if(!sym){ alert('Please enter a stock symbol'); return; }
    const out = document.getElementById('progressive-prediction-results');
    if(out) out.textContent = 'Loading...';
    try{
      const j = await fetchJSON(`/api/ml/progressive/predict/${encodeURIComponent(sym)}?mode=${encodeURIComponent(mode)}`, { method: 'POST' });
      if(out){
        if(j && j.status === 'success'){
          const d = j.data || j.result || j;
          out.textContent = JSON.stringify(d, null, 2);
        } else {
          out.textContent = (j && (j.detail || j.message)) ? String(j.detail || j.message) : 'Prediction failed';
        }
      }
    }catch(e){ if(out) out.textContent = 'Prediction error: ' + e.message; }
  }

  async function startProgressiveTraining(){
    const sym = (document.getElementById('progressive-symbol')?.value || '').trim().toUpperCase();
    const sel = document.getElementById('progressive-model-types');
    const mode = (document.getElementById('progressive-mode')?.value || 'progressive');
    if(!sym){ alert('Please enter a stock symbol'); return; }
    let types = ['lstm'];
    if(sel && sel.options){
      types = Array.from(sel.options).filter(o=>o.selected).map(o=>o.value);
      if(types.length === 0) types = ['lstm'];
    }
    const panel = document.getElementById('progressive-training-results');
    if(panel) panel.textContent = 'Starting training...';
    try{
      const qs = new URLSearchParams({ symbol: sym, model_types: types.join(','), mode });
      const j = await fetchJSON(`/api/ml/progressive/train?${qs.toString()}`, { method: 'POST' });
      if(panel){
        if(j && j.status === 'success'){
          panel.textContent = 'Training started...';
        } else {
          panel.textContent = (j && (j.detail || j.message)) ? String(j.detail || j.message) : 'Failed to start training';
        }
      }
      // Poll basic status (coarse)
      let tries = 0;
      const poll = async () => {
        tries += 1;
        const s = await fetchJSON('/api/ml/progressive/training/status');
        if(panel){ panel.textContent = JSON.stringify(s, null, 2); }
        if(tries < 30 && s && s.status !== 'error' && s?.data?.running){
          setTimeout(poll, 2000);
        }
      };
      setTimeout(poll, 1500);
    }catch(e){ if(panel) panel.textContent = 'Training error: ' + e.message; }
  }

  // Small helpers for Backtest explainer (if not present on RL page)
  function openBacktestExplainer(){ const m = document.getElementById('backtest-explainer-modal'); if(m) m.style.display='block'; }
  function closeBacktestExplainer(){ const m = document.getElementById('backtest-explainer-modal'); if(m) m.style.display='none'; }

  // Expose to window
  window.initProgressiveML = initProgressiveML;
  window.getProgressivePrediction = getProgressivePrediction;
  window.startProgressiveTraining = startProgressiveTraining;
  window.openBacktestExplainer = window.openBacktestExplainer || openBacktestExplainer;
  window.closeBacktestExplainer = window.closeBacktestExplainer || closeBacktestExplainer;
})();
