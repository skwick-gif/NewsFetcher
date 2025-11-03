// Progressive ML dashboard helpers
(function(){
  let currentBacktestJobId = null;
  let backtestPollingInterval = null;
  let backtestChart = null;

  async function fetchJSON(url, opts){
    const r = await fetch(url, opts || {});
    try {
      return await r.json();
    } catch (err) {
      return {
        status: r.ok ? 'success' : 'error',
        message: await r.text().catch(() => String(err))
      };
    }
  }

  function getSelectedModels(){
    const sel = document.getElementById('progressive-model-types');
    if(!sel || !sel.options) return ['lstm'];
    const picked = Array.from(sel.options).filter(opt => opt.selected).map(opt => opt.value);
    return picked.length ? picked : ['lstm'];
  }

  function readIndicatorParams(){
    const val = id => document.getElementById(id)?.value;
    return {
      rsi_period: parseInt(val('ind-rsi-period') || '14', 10),
      macd_fast: parseInt(val('ind-macd-fast') || '12', 10),
      macd_slow: parseInt(val('ind-macd-slow') || '26', 10),
      macd_signal: parseInt(val('ind-macd-signal') || '9', 10),
      sma_periods: val('ind-sma-periods') || '5,10,20,50',
      ema_periods: val('ind-ema-periods') || '5,10,20,50',
      bb_period: parseInt(val('ind-bb-period') || '20', 10),
      bb_std: parseFloat(val('ind-bb-std') || '2')
    };
  }

  function showProgressCard(){
    const progress = document.getElementById('backtest-progress');
    if(progress) progress.style.display = 'block';
    const results = document.getElementById('backtest-results-container');
    if(results) results.style.display = 'none';
    const chart = document.getElementById('backtest-chart-container');
    if(chart) chart.style.display = 'none';
    const plan = document.getElementById('backtest-plan-card');
    if(plan) plan.style.display = 'none';
    const champ = document.getElementById('backtest-champion-card');
    if(champ) champ.style.display = 'none';
    const preds = document.getElementById('backtest-current-preds');
    if(preds) preds.innerHTML = '';
    const forward = document.getElementById('backtest-forward-results');
    if(forward) forward.innerHTML = '';
  }

  function updateProgressStatus(text, pct){
    const status = document.getElementById('backtest-status-text');
    const fill = document.getElementById('backtest-progress-fill');
    if(typeof pct === 'number' && fill){
      const clamped = Math.max(0, Math.min(100, pct));
      fill.style.width = clamped + '%';
    }
    if(status && text){
      status.textContent = text;
    }
  }

  async function startBacktesting(){
    const symbol = (document.getElementById('progressive-symbol')?.value || '').trim().toUpperCase();
    const startDate = document.getElementById('backtest-start-date')?.value;
    const endDate = document.getElementById('backtest-end-date')?.value;
    const testPeriod = parseInt(document.getElementById('backtest-test-period')?.value || '0', 10);
    const maxIterations = parseInt(document.getElementById('backtest-max-iterations')?.value || '0', 10);
    const targetAccuracy = parseFloat(document.getElementById('backtest-target-accuracy')?.value || '0') / 100;
    const autoStop = !!document.getElementById('backtest-auto-stop')?.checked;
    const selectedModels = getSelectedModels();

    if(!symbol){ alert('Please enter a stock symbol'); return; }
    if(!startDate || !endDate){ alert('Please select both start and end dates'); return; }
    if(new Date(startDate) >= new Date(endDate)){ alert('Start date must be before end date'); return; }
    if(Number.isNaN(testPeriod) || testPeriod < 1 || testPeriod > 365){ alert('Test period must be between 1 and 365 days'); return; }
    if(Number.isNaN(maxIterations) || maxIterations < 1 || maxIterations > 50){ alert('Max iterations must be between 1 and 50'); return; }
    if(Number.isNaN(targetAccuracy) || targetAccuracy < 0.1 || targetAccuracy > 1){ alert('Target accuracy must be between 10% and 100%'); return; }
    if(!selectedModels.length){ alert('Please select at least one model type'); return; }

    showProgressCard();
    updateProgressStatus('Starting backtest...', 0);

    const payload = {
      symbol,
      train_start_date: startDate,
      train_end_date: endDate,
      test_period_days: testPeriod,
      max_iterations: maxIterations,
      target_accuracy: targetAccuracy,
      auto_stop: autoStop,
      model_types: selectedModels,
      indicator_params: readIndicatorParams()
    };

    try {
      const resp = await fetch('/api/ml/progressive/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json().catch(() => ({}));
      if(!resp.ok){
        const detail = data?.detail || data?.message || JSON.stringify(data);
        throw new Error(detail || `HTTP ${resp.status}`);
      }

      if(data.preflight || data.plan || data.adjusted){
        displayPlanAndPreflight(data);
      }

      if(data.status === 'backtest_started' && data.job_id){
        currentBacktestJobId = data.job_id;
        if(backtestPollingInterval){
          clearInterval(backtestPollingInterval);
        }
        backtestPollingInterval = setInterval(() => pollBacktestProgress(data.job_id), 1500);
        updateProgressStatus('Backtest started...', 5);
      } else if(data.status === 'success' && data.backtest_results){
        await displayBacktestResults(data.backtest_results);
      } else {
        updateProgressStatus('Backtest failed to start', 0);
        alert('Backtesting failed: ' + (data.error || data.detail || 'Unknown error'));
      }
    } catch (err) {
      console.error('Error starting backtest:', err);
      updateProgressStatus('Error: ' + err.message, 0);
      alert('Error starting backtest: ' + err.message);
    }
  }

  async function startAutoBacktesting(){
    const symbol = (document.getElementById('progressive-symbol')?.value || '').trim().toUpperCase();
    const selectedModels = getSelectedModels();
    if(!symbol){ alert('Please enter a stock symbol'); return; }
    if(!selectedModels.length){ alert('Select at least one model'); return; }

    showProgressCard();
    updateProgressStatus('Planning & starting…', 0);

    const payload = {
      symbol,
      auto_plan: true,
      deep_mode: true,
      ensure_fresh_data: true,
      auto_adjust_iterations: true,
      desired_iterations: 10,
      desired_test_period_days: 14,
      model_types: selectedModels,
      indicator_params: readIndicatorParams(),
      train_start_date: '2000-01-01',
      train_end_date: '2099-12-31',
      test_period_days: 14,
      max_iterations: 10,
      target_accuracy: 0.85,
      auto_stop: true
    };

    try {
      const resp = await fetch('/api/ml/progressive/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json().catch(() => ({}));
      if(!resp.ok){
        const detail = data?.detail || data?.message || JSON.stringify(data);
        throw new Error(detail || `HTTP ${resp.status}`);
      }
      if(data.preflight || data.plan || data.adjusted){
        displayPlanAndPreflight(data);
      }
      if(data.status === 'backtest_started' && data.job_id){
        currentBacktestJobId = data.job_id;
        if(backtestPollingInterval){
          clearInterval(backtestPollingInterval);
        }
        backtestPollingInterval = setInterval(() => pollBacktestProgress(data.job_id), 1500);
        updateProgressStatus('Backtest started...', 5);
      } else {
        updateProgressStatus('Backtest failed to start', 0);
      }
    } catch (err) {
      console.error('Auto backtest failed:', err);
      alert('Auto backtest failed: ' + err.message);
      const progress = document.getElementById('backtest-progress');
      if(progress) progress.style.display = 'none';
    }
  }

  async function stopBacktesting(){
    try {
      if(backtestPollingInterval){
        clearInterval(backtestPollingInterval);
        backtestPollingInterval = null;
      }
      if(currentBacktestJobId){
        await fetch(`/api/ml/progressive/backtest/cancel/${currentBacktestJobId}`, { method: 'POST' });
      }
      updateProgressStatus('Cancelling...', undefined);
    } catch (err) {
      console.error('Stop backtest error:', err);
      updateProgressStatus('Stop requested', undefined);
    }
  }

  async function pollBacktestProgress(jobId){
    try {
      const resp = await fetch(`/api/ml/progressive/backtest/status/${jobId}`);
      if(!resp.ok){
        throw new Error(`HTTP ${resp.status}`);
      }
      const st = await resp.json();

      const pct = Math.round(st.progress || 0);
      let etaText = '';
      if(st.eta_seconds !== null && st.eta_seconds !== undefined){
        const mins = Math.floor(st.eta_seconds / 60);
        const secs = Math.round(st.eta_seconds % 60);
        etaText = mins > 0 ? ` • ETA: ${mins}m ${secs}s` : ` • ETA: ${secs}s`;
      }
      updateProgressStatus(`${st.current_step || 'Running...'} • ${pct}%${etaText}`, pct);

      if(st.preflight || st.plan || st.adjusted || st.note){
        displayPlanAndPreflight(st);
      }
      if(st.champion){
        displayChampion(st.champion);
      }
      if(st.current_predictions && st.current_predictions.predictions){
        renderCurrentPredictions(st.current_predictions);
      }

      if(st.status === 'completed'){
        if(backtestPollingInterval){
          clearInterval(backtestPollingInterval);
          backtestPollingInterval = null;
        }
        if(st.result){
          await displayBacktestResults(st.result);
          if(st.champion){ displayChampion(st.champion); }
          if(st.current_predictions && st.current_predictions.predictions){
            renderCurrentPredictions(st.current_predictions);
          }
        } else {
          const progress = document.getElementById('backtest-progress');
          if(progress) progress.style.display = 'none';
          const results = document.getElementById('backtest-results-container');
          if(results) results.style.display = 'none';
          const chart = document.getElementById('backtest-chart-container');
          if(chart) chart.style.display = 'none';
          alert('Backtest completed but no result received');
        }
      } else if(st.status === 'failed'){
        if(backtestPollingInterval){
          clearInterval(backtestPollingInterval);
          backtestPollingInterval = null;
        }
        updateProgressStatus(`Failed: ${st.error || 'Unknown error'}`, pct);
        alert(`Backtest failed: ${st.error || 'Unknown error'}`);
      } else if(st.status === 'cancelled'){
        if(backtestPollingInterval){
          clearInterval(backtestPollingInterval);
          backtestPollingInterval = null;
        }
        updateProgressStatus('⏹ Backtest cancelled', pct);
        const progress = document.getElementById('backtest-progress');
        if(progress) progress.style.display = 'none';
      }
    } catch (err) {
      console.error('Error polling backtest progress:', err);
    }
  }

  function renderCurrentPredictions(cp){
    try {
      const box = document.getElementById('backtest-current-preds');
      if(!box) return;
      const horizons = ['1d','7d','30d'];
      const summary = horizons.map(hk => {
        const p = cp.predictions[hk];
        if(!p) return null;
        const sig = p.signal || p.direction || '-';
        const conf = typeof p.confidence === 'number' ? Math.round(p.confidence * 100) + '%' : '—';
        const pct = typeof p.price_change_pct === 'number' ? `${p.price_change_pct >= 0 ? '+' : ''}${(p.price_change_pct * 100).toFixed(1)}%` : '—';
        return `${hk.toUpperCase()}: ${sig} • Conf ${conf} • ${pct}`;
      }).filter(Boolean).join(' | ');

      const riskLines = horizons.map(hk => {
        const p = cp.predictions[hk];
        if(!p || !p.risk) return null;
        const r = p.risk;
        const slp = typeof r.stop_loss_pct === 'number' ? (r.stop_loss_pct * 100).toFixed(1) + '%' : '—';
        const tpp = typeof r.take_profit_pct === 'number' ? (r.take_profit_pct * 100).toFixed(1) + '%' : '—';
        const sl = typeof r.stop_loss === 'number' ? '$' + Number(r.stop_loss).toFixed(2) : '—';
        const tp = typeof r.take_profit === 'number' ? '$' + Number(r.take_profit).toFixed(2) : '—';
        const basis = r.basis ? ` • basis: ${r.basis}` : '';
        const rr = r.rr ? ` • RR: 1:${r.rr}` : '';
        return `${hk.toUpperCase()}: SL ${sl} (${slp}) • TP ${tp} (${tpp})${rr}${basis}`;
      }).filter(Boolean).join('\n');

      const tipLines = [
        'Explanation:',
        '• BUY/SELL/HOLD = signal derived from expected move and direction probability.',
        '• Conf = blended confidence.',
        '• %+ = expected price change (capped per horizon).',
        'Caps: 1D ±10%, 7D ±20%, 30D ±40%.'
      ];
      const capped = horizons.filter(hk => cp.predictions[hk]?.capped).map(hk => hk.toUpperCase());
      if(capped.length) tipLines.push(`Capped: ${capped.join(', ')}`);
      if(riskLines) tipLines.push('', 'Risk (per horizon):', riskLines);
      const tooltip = tipLines.join('\n');

      const compactRisk = horizons.map(hk => {
        const p = cp.predictions[hk];
        if(!p || !p.risk) return null;
        const r = p.risk;
        const sl = typeof r.stop_loss === 'number' ? '$' + Number(r.stop_loss).toFixed(2) : '—';
        const tp = typeof r.take_profit === 'number' ? '$' + Number(r.take_profit).toFixed(2) : '—';
        return `${hk.toUpperCase()}: ${sl}/${tp}`;
      }).filter(Boolean).join(' | ');

      box.innerHTML = summary
        ? `<div><span title="${tooltip}">Current predictions → ${summary}</span>${compactRisk ? `<div style="opacity:0.75; font-size:0.85em; margin-top:2px;">SL/TP → ${compactRisk}</div>` : ''}</div>`
        : '';
      const champ = document.getElementById('backtest-champion-card');
      if(champ) champ.style.display = 'block';
    } catch (err) {
      console.warn('render current preds failed', err);
    }
  }

  async function displayBacktestResults(results){
    if(!results || (!results.all_iterations && !results.iterations)){
      try {
        const symbol = (document.getElementById('progressive-symbol')?.value || '').trim().toUpperCase();
        if(symbol){
          const fallbackResp = await fetch(`/api/ml/progressive/backtest/results/${symbol}`);
          if(fallbackResp.ok){
            const fallbackData = await fallbackResp.json();
            if(fallbackData.status === 'success' && fallbackData.results){
              results = fallbackData.results;
            }
          }
        }
      } catch (err) {
        console.warn('Backtest fallback fetch failed:', err);
      }
    }

    if(!results || (!results.all_iterations && !results.iterations)){
      const msg = results && results.error ? `Backtest error: ${results.error}` : 'No results to display';
      alert(msg);
      const progress = document.getElementById('backtest-progress');
      if(progress) progress.style.display = 'none';
      const resultsBox = document.getElementById('backtest-results-container');
      if(resultsBox) resultsBox.style.display = 'none';
      const chartBox = document.getElementById('backtest-chart-container');
      if(chartBox) chartBox.style.display = 'none';
      return;
    }

    if(!results.all_iterations && results.iterations){
      try {
        const best = results.iterations.reduce((acc, it) => {
          if(typeof it.accuracy === 'number' && it.accuracy > (acc.accuracy ?? -1)) return it;
          return acc;
        }, {});
        results.all_iterations = results.iterations;
        if(!results.best_iteration && best && typeof best.iteration === 'number'){
          results.best_iteration = best.iteration;
        }
      } catch (err) {
        console.warn('Failed to normalize legacy backtest results:', err);
        results.all_iterations = results.iterations;
      }
    }

    const progress = document.getElementById('backtest-progress');
    if(progress) progress.style.display = 'none';
    const resultsBox = document.getElementById('backtest-results-container');
    if(resultsBox) resultsBox.style.display = 'block';
    const chartBox = document.getElementById('backtest-chart-container');
    if(chartBox) chartBox.style.display = 'block';

    const tbody = document.getElementById('backtest-results-tbody');
    if(tbody){ tbody.innerHTML = ''; }

    try {
      const ths = document.querySelectorAll('#backtest-results-table thead th');
      if(ths && ths.length >= 6){
        ths[0].title = 'Iteration number in the expanding-window backtest';
        ths[1].title = 'Last date included in training for this iteration';
        ths[2].title = 'Direction accuracy over the test window';
        ths[3].title = 'Validation loss (lower is better)';
        ths[4].title = 'Training duration for this iteration';
        ths[5].title = 'Best iteration marker';
      }
    } catch (err) {}

    results.all_iterations.forEach(iter => {
      if(!tbody) return;
      const row = tbody.insertRow();
      const isBest = iter.iteration === results.best_iteration;
      row.innerHTML = `
        <td style="padding: 8px;">${iter.iteration}</td>
        <td style="padding: 8px;">${iter.train_until || ''}</td>
        <td style="padding: 8px; color: ${typeof iter.accuracy === 'number' && iter.accuracy >= 0.85 ? '#10b981' : '#f59e0b'};">${typeof iter.accuracy === 'number' ? (iter.accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
        <td style="padding: 8px;">${typeof iter.val_loss === 'number' ? iter.val_loss.toFixed(4) : 'N/A'}</td>
        <td style="padding: 8px;">${typeof iter.training_time === 'number' ? iter.training_time.toFixed(1) + 's' : 'N/A'}</td>
        <td style="padding: 8px;">${isBest ? '✅ Best' : ''}</td>
      `;
    });

    createBacktestChart(results);
  }

  function createBacktestChart(results){
    if(typeof Chart === 'undefined'){
      console.warn('Chart.js not available for backtest chart');
      return;
    }
    const ctx = document.getElementById('backtest-chart')?.getContext('2d');
    if(!ctx) return;

    if(backtestChart){
      backtestChart.destroy();
    }

    const iterations = results.all_iterations.map(iter => `Iter ${iter.iteration ?? '?'}`);
    const accuracies = results.all_iterations.map(iter => typeof iter.accuracy === 'number' ? iter.accuracy * 100 : 0);
    const losses = results.all_iterations.map(iter => typeof iter.val_loss === 'number' ? iter.val_loss : 0);

    backtestChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: iterations,
        datasets: [
          {
            label: 'Accuracy (%)',
            data: accuracies,
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            yAxisID: 'y',
            tension: 0.3
          },
          {
            label: 'Loss',
            data: losses,
            borderColor: 'rgb(239, 68, 68)',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            yAxisID: 'y1',
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          title: { display: true, text: 'Backtest Performance Over Iterations', color: '#e2e8f0' },
          legend: { labels: { color: '#e2e8f0' } }
        },
        scales: {
          x: { ticks: { color: '#e2e8f0' }, grid: { color: 'rgba(255,255,255,0.1)' } },
          y: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'Accuracy (%)', color: '#e2e8f0' },
            ticks: { color: '#e2e8f0' },
            grid: { color: 'rgba(255,255,255,0.1)' }
          },
          y1: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'Loss', color: '#e2e8f0' },
            ticks: { color: '#e2e8f0' },
            grid: { display: false }
          }
        }
      }
    });
  }

  function displayPlanAndPreflight(obj){
    try {
      const card = document.getElementById('backtest-plan-card');
      const badge = document.getElementById('backtest-preflight-badge');
      const content = document.getElementById('backtest-plan-content');
      if(!card || !content) return;
      const pf = obj.preflight || {};
      const plan = obj.plan || pf.plan || {};
      const adjusted = obj.adjusted ? 'Yes' : 'No';
      const last = pf.last_data_date || '-';
      const tpd = pf.test_period_days || plan.test_period_days || '-';
      const req = pf.requested_max_iterations || plan.requested_iterations || '-';
      const feasible = pf.feasible_max_iterations !== undefined ? pf.feasible_max_iterations : '-';
      const lines = [
        plan.train_start_date ? `Train: ${plan.train_start_date} → ${plan.train_end_date}` : (obj.train_start_date && obj.train_end_date ? `Train: ${obj.train_start_date} → ${obj.train_end_date}` : ''),
        `Test window: ${tpd} days • Requested iters: ${req} • Feasible: ${feasible}`,
        `Adjusted: ${adjusted}`,
        `Last data: ${last}`,
        obj.note ? obj.note : ''
      ].filter(Boolean);

      try {
        const scout = obj.scout || (obj.result && obj.result.scout) || null;
        if(scout && scout.chosen){
          const ch = scout.chosen;
          lines.push(`Chosen config → Window ${ch.window_days}d • Seq ${ch.sequence_length} • Profile ${ch.profile} • Fwd Acc ${(ch.accuracy*100).toFixed(1)}% • N=${ch.predictions}`);
        }
      } catch (err) {}

      content.innerHTML = lines.join('<br>');
      if(badge) badge.textContent = 'Planned';
      card.style.display = 'block';
    } catch (err) {
      console.warn('displayPlanAndPreflight failed', err);
    }
  }

  function displayChampion(champion){
    try {
      const card = document.getElementById('backtest-champion-card');
      const content = document.getElementById('backtest-champion-content');
      if(!card || !content) return;
      const meta = champion.meta || {};
      const bestIter = meta.best_iteration || '-';
      const acc = meta.summary?.best_accuracy !== undefined ? (meta.summary.best_accuracy * 100).toFixed(1) + '%' : 'N/A';
      const dir = champion.dir || champion.path || champion;
      content.innerHTML = `Best Iteration: <strong>${bestIter}</strong> • Best Accuracy: <strong>${acc}</strong><br>Path: <code style="font-size:0.85em;">${dir}</code>`;
      card.style.display = 'block';
    } catch (err) {
      console.warn('displayChampion failed', err);
    }
  }

  async function runChampionForwardTest(){
    try {
      const symbol = (document.getElementById('progressive-symbol')?.value || '').trim().toUpperCase();
      if(!symbol){ alert('Please enter a stock symbol first'); return; }
      const resp = await fetch(`/api/ml/progressive/champion/forward_test/${symbol}`, { method: 'POST' });
      const data = await resp.json().catch(() => ({}));
      if(!resp.ok){
        const detail = data?.detail || data?.message || JSON.stringify(data);
        throw new Error(detail || `HTTP ${resp.status}`);
      }
      const box = document.getElementById('backtest-forward-results');
      if(box){
        const m = data.metrics || {};
        const acc = typeof m.direction_accuracy === 'number' ? (m.direction_accuracy * 100).toFixed(1) + '%' : (typeof m.accuracy === 'number' ? (m.accuracy * 100).toFixed(1) + '%' : 'N/A');
        const mae = m.mae != null ? Number(m.mae).toFixed(4) : 'N/A';
        const rmse = m.rmse != null ? Number(m.rmse).toFixed(4) : 'N/A';
        const mape = m.mape != null ? Number(m.mape).toFixed(2) + '%' : 'N/A';
        const n = m.predictions_made ?? m.test_samples ?? '-';
        box.innerHTML = `Forward ${data.forward_start} → ${data.forward_end}: Accuracy ${acc}, MAE ${mae}, RMSE ${rmse}, MAPE ${mape}, N=${n}`;
        const champ = document.getElementById('backtest-champion-card');
        if(champ) champ.style.display = 'block';
      }
    } catch (err) {
      alert('Forward test failed: ' + err.message);
    }
  }

  async function loadBacktestHistory(){
    try {
      const symbol = (document.getElementById('progressive-symbol')?.value || '').trim().toUpperCase();
      if(!symbol) return;
      const resp = await fetch(`/api/ml/progressive/backtest/history/${symbol}`);
      const data = await resp.json().catch(() => ({}));
      const listDiv = document.getElementById('backtest-history-list');
      if(!listDiv) return;
      if(!data || data.status !== 'success' || !Array.isArray(data.items) || data.items.length === 0){
        listDiv.innerHTML = '<div style="opacity:0.7;">No history found</div>';
        return;
      }
      listDiv.innerHTML = data.items.map(item => {
        const acc = typeof item.best_accuracy === 'number' ? (item.best_accuracy * 100).toFixed(1) + '%' : 'N/A';
        const modified = item.modified ? new Date(item.modified).toLocaleString() : '';
        return `
          <div style="display:flex; justify-content: space-between; align-items:center; padding:6px 8px; border-bottom:1px solid rgba(255,255,255,0.08);">
            <div>
              <div style="font-size:0.9em;">${item.file}</div>
              <small style="opacity:0.7;">${modified} • Best: ${acc} • Iters: ${item.iterations ?? '—'}</small>
            </div>
            <button onclick="loadBacktestResultFile('${symbol}', '${item.file}')" style="background:#2563eb; border:none; color:#fff; padding:6px 10px; border-radius:6px; cursor:pointer; font-size:0.8em;">Load</button>
          </div>
        `;
      }).join('');
    } catch (err) {
      console.error('Failed to load history:', err);
    }
  }

  async function loadBacktestResultFile(symbol, file){
    try {
      const resp = await fetch(`/api/ml/progressive/backtest/result_by_file/${symbol}/${file}`);
      if(!resp.ok) return;
      const data = await resp.json().catch(() => ({}));
      if(data && data.status === 'success' && data.results){
        await displayBacktestResults(data.results);
      }
    } catch (err) {
      console.error('Failed to load result file:', err);
    }
  }

  async function initProgressiveML(){
    try {
      const info = await fetchJSON('/api/ml/progressive/status');
      const el = document.getElementById('progressive-ml-status-display');
      if(!el) return;
      if(!info || info.status === 'error'){
        el.textContent = 'Progressive ML: backend unavailable';
        return;
      }

      if(info.status === 'success'){
        const d = info.data || {};
        const parts = [];
        parts.push('Status: ' + (d.status || 'ready'));
        if(typeof d.jobs_running === 'number') parts.push(`Jobs: ${d.jobs_running}`);
        if(d.last_updated) parts.push(`Updated: ${new Date(d.last_updated).toLocaleString()}`);
        el.textContent = parts.join(' • ');
        return;
      }

      if(info.status === 'available'){
        const ready = [];
        if(typeof info.data_loader === 'boolean') ready.push(`Loader: ${info.data_loader ? '✅' : '⚠️'}`);
        if(typeof info.trainer === 'boolean') ready.push(`Trainer: ${info.trainer ? '✅' : '⚠️'}`);
        if(typeof info.predictor === 'boolean') ready.push(`Predictor: ${info.predictor ? '✅' : '⚠️'}`);
        const parts = ['Progressive ML: ready'];
        if(ready.length) parts.push(ready.join(' • '));
        if(info.timestamp) parts.push(`Updated: ${new Date(info.timestamp).toLocaleString()}`);
        el.textContent = parts.join(' • ');
        return;
      }

      if(info.status === 'unavailable'){
        el.textContent = 'Progressive ML: backend disabled';
        return;
      }

      el.textContent = `Progressive ML: ${info.status || 'unknown status'}`;
    } catch (err) {
      const el = document.getElementById('progressive-ml-status-display');
      if(el) el.textContent = 'Progressive ML: error reading status';
    }
  }

  async function getProgressivePrediction(){
    const sym = (document.getElementById('progressive-symbol')?.value || '').trim().toUpperCase();
    const mode = document.getElementById('progressive-mode')?.value || 'progressive';
    if(!sym){ alert('Please enter a stock symbol'); return; }
    const out = document.getElementById('progressive-prediction-results');
    if(out) out.textContent = 'Loading...';
    try {
      const res = await fetchJSON(`/api/ml/progressive/predict/${encodeURIComponent(sym)}?mode=${encodeURIComponent(mode)}`, { method: 'POST' });
      if(out){
        if(res && res.status === 'success'){
          const data = res.data || res.result || res;
          out.textContent = JSON.stringify(data, null, 2);
        } else {
          out.textContent = res && (res.detail || res.message) ? String(res.detail || res.message) : 'Prediction failed';
        }
      }
    } catch (err) {
      if(out) out.textContent = 'Prediction error: ' + err.message;
    }
  }

  async function startProgressiveTraining(){
    const sym = (document.getElementById('progressive-symbol')?.value || '').trim().toUpperCase();
    const mode = document.getElementById('progressive-mode')?.value || 'progressive';
    if(!sym){ alert('Please enter a stock symbol'); return; }
    const types = getSelectedModels();
    const panel = document.getElementById('progressive-training-results');
    if(panel) panel.textContent = 'Starting training...';
    try {
      const qs = new URLSearchParams({ symbol: sym, model_types: types.join(','), mode });
      const res = await fetchJSON(`/api/ml/progressive/train?${qs.toString()}`, { method: 'POST' });
      if(panel){
        if(res && res.status === 'success'){
          panel.textContent = 'Training started...';
        } else {
          panel.textContent = res && (res.detail || res.message) ? String(res.detail || res.message) : 'Failed to start training';
        }
      }
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
    } catch (err) {
      if(panel) panel.textContent = 'Training error: ' + err.message;
    }
  }

  function openBacktestExplainer(){
    const modal = document.getElementById('backtest-explainer-modal');
    if(modal) modal.style.display = 'block';
  }

  function closeBacktestExplainer(){
    const modal = document.getElementById('backtest-explainer-modal');
    if(modal) modal.style.display = 'none';
  }

  window.initProgressiveML = initProgressiveML;
  window.getProgressivePrediction = getProgressivePrediction;
  window.startProgressiveTraining = startProgressiveTraining;
  window.startBacktesting = startBacktesting;
  window.startAutoBacktesting = startAutoBacktesting;
  window.stopBacktesting = stopBacktesting;
  window.loadBacktestHistory = loadBacktestHistory;
  window.loadBacktestResultFile = loadBacktestResultFile;
  window.runChampionForwardTest = runChampionForwardTest;
  window.openBacktestExplainer = window.openBacktestExplainer || openBacktestExplainer;
  window.closeBacktestExplainer = window.closeBacktestExplainer || closeBacktestExplainer;
})();
