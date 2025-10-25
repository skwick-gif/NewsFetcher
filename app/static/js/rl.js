(function(){
  async function loadStatus(){
    try {
      const r = await fetch('/api/rl/status');
      const j = await r.json();
      document.getElementById('rl-status').textContent = j?.status || 'Idle';
      document.getElementById('rl-positions').textContent = JSON.stringify(j?.positions || []);
      document.getElementById('rl-pnl').textContent = (j?.pnl ?? '—');
      const d = (j?.decisions || []).map(x=>`${x.time} ${x.symbol} ${x.action} @${x.price}`).join('<br/>');
      document.getElementById('rl-decisions').innerHTML = d || '—';
    } catch(e){
      document.getElementById('rl-status').textContent = 'Error';
    }
  }
  loadStatus();
})();