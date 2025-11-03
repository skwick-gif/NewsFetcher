(function(){
  const statusEl = document.getElementById('rl-status');
  const positionsEl = document.getElementById('rl-positions');
  const pnlEl = document.getElementById('rl-pnl');
  const ordersEl = document.getElementById('rl-open-orders');
  const ibkrStatusEl = document.getElementById('ibkr-connection');
  const ibkrStatusText = ibkrStatusEl ? ibkrStatusEl.querySelector('.badge-text') : null;
  const ibkrConnectBtn = document.getElementById('ibkr-connect-btn');
  const CONNECT_LABEL = 'Connect IBKR';
  const RECONNECT_LABEL = 'Reconnect IBKR';
  const RETRY_LABEL = 'Retry Connect';
  const CONNECTING_LABEL = 'Connecting…';
  const IBKR_STATUS_INTERVAL_MS = 20000;
  const toastContainer = document.getElementById('toast-container');
  const toastCooldown = new Map();
  const CAPACITY = 4;

  const showToast = (variant, message) => {
    if (!toastContainer || !message) {
      return;
    }
    const node = document.createElement('div');
    node.className = `toast toast-${variant || 'info'}`;
    node.textContent = message;
    toastContainer.appendChild(node);
    while (toastContainer.children.length > CAPACITY) {
      toastContainer.removeChild(toastContainer.firstElementChild);
    }
    const remove = () => {
      if (node.parentNode) {
        node.parentNode.removeChild(node);
      }
    };
    node.addEventListener('click', remove);
    setTimeout(remove, 5000);
  };

  const emitToast = (variant, message, key) => {
    if (!message) {
      return;
    }
    const token = key || message;
    const now = Date.now();
    const previous = toastCooldown.get(token) || 0;
    if (now - previous < 45000) {
      return;
    }
    toastCooldown.set(token, now);
    showToast(variant, message);
  };

  if (typeof window !== 'undefined') {
    window.emitRLToast = emitToast;
    window.showRLToast = showToast;
  }

  if (!statusEl) {
    return;
  }

  const REFRESH_INTERVAL_MS = 15000;

  const updateIbkrBadge = (state, message) => {
    if (!ibkrStatusEl) {
      return;
    }
    ibkrStatusEl.classList.remove('badge-online', 'badge-offline', 'badge-error', 'badge-loading');
    switch (state) {
      case 'online':
        ibkrStatusEl.classList.add('badge-online');
        break;
      case 'offline':
        ibkrStatusEl.classList.add('badge-offline');
        break;
      case 'error':
        ibkrStatusEl.classList.add('badge-error');
        break;
      default:
        ibkrStatusEl.classList.add('badge-loading');
        break;
    }
    if (ibkrStatusText) {
      ibkrStatusText.textContent = message;
    }
  };

  const refreshIbkrStatus = async () => {
    if (!ibkrStatusEl) {
      return;
    }
    try {
  const response = await fetch('/api/ibkr/status');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();
      const connected = payload.isConnected ?? payload.connected ?? false;
      if (connected) {
        const endpoint = payload.host && payload.port ? `${payload.host}:${payload.port}` : payload.port ? `Port ${payload.port}` : 'Live';
        updateIbkrBadge('online', `IBKR: Live (${endpoint})`);
        if (ibkrConnectBtn) {
          ibkrConnectBtn.disabled = false;
          ibkrConnectBtn.textContent = RECONNECT_LABEL;
        }
      } else {
        const message = payload.detail ? `IBKR: ${payload.detail}` : 'IBKR: Disconnected';
        updateIbkrBadge('offline', message);
        if (ibkrConnectBtn) {
          ibkrConnectBtn.disabled = false;
          ibkrConnectBtn.textContent = CONNECT_LABEL;
        }
      }
    } catch (error) {
      console.error('Failed to refresh IBKR status', error);
      updateIbkrBadge('error', 'IBKR: Error');
      if (ibkrConnectBtn) {
        ibkrConnectBtn.disabled = false;
        ibkrConnectBtn.textContent = RETRY_LABEL;
      }
      emitToast('error', 'Unable to reach IBKR bridge. Check bridge process.', 'ibkr-status');
    }
  };

  const connectIbkrBridge = async () => {
    if (!ibkrConnectBtn || ibkrConnectBtn.disabled) {
      return;
    }
    let statusUpdated = false;
    ibkrConnectBtn.disabled = true;
    ibkrConnectBtn.textContent = CONNECTING_LABEL;
    updateIbkrBadge('loading', 'IBKR: Connecting…');
    try {
      const response = await fetch('/api/ibkr/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        let detail = await response.text().catch(()=>'');
        try { const j = JSON.parse(detail); detail = j.detail || j.message || detail; } catch(_){}
        throw new Error(detail || `HTTP ${response.status}`);
      }
      await refreshIbkrStatus();
      statusUpdated = true;
      emitToast('success', 'IBKR bridge connected.', 'ibkr-connect-ok');
    } catch (error) {
      console.error('Failed to initiate IBKR connection', error);
      updateIbkrBadge('error', 'IBKR: Connect failed');
      if (ibkrConnectBtn) {
        ibkrConnectBtn.textContent = RETRY_LABEL;
      }
      emitToast('error', `IBKR connect failed: ${error.message || 'See bridge logs'}`, 'ibkr-connect');
    } finally {
      if (ibkrConnectBtn) {
        ibkrConnectBtn.disabled = false;
        if (!statusUpdated && ibkrConnectBtn.textContent === CONNECTING_LABEL) {
          ibkrConnectBtn.textContent = RETRY_LABEL;
        }
      }
    }
  };

  const toNumber = (value) => {
    if (value === null || value === undefined) {
      return NaN;
    }
    if (typeof value === 'number') {
      return Number.isFinite(value) ? value : NaN;
    }
    if (typeof value === 'string') {
      const cleaned = value.replace(/[^0-9.\-]/g, '');
      if (!cleaned) {
        return NaN;
      }
      const parsed = Number(cleaned);
      return Number.isFinite(parsed) ? parsed : NaN;
    }
    return NaN;
  };

  const formatCurrency = (value) => {
    const num = toNumber(value);
    if (!Number.isFinite(num)) {
      return '—';
    }
    return `$${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatSignedCurrency = (value) => {
    const num = toNumber(value);
    if (!Number.isFinite(num)) {
      return '—';
    }
    const absolute = Math.abs(num).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const prefix = num > 0 ? '+' : num < 0 ? '-' : '';
    return `${prefix}$${absolute}`;
  };

  const formatQuantity = (value) => {
    const num = toNumber(value);
    if (!Number.isFinite(num)) {
      return '—';
    }
    const fractionDigits = Math.abs(num) < 10 ? 2 : Math.abs(num) < 1000 ? 1 : 0;
    return num.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: fractionDigits });
  };

  const setErrorState = (message) => {
    statusEl.textContent = message;
    if (positionsEl) {
      positionsEl.textContent = '—';
    }
    if (pnlEl) {
      pnlEl.textContent = '—';
    }
    if (ordersEl) {
      ordersEl.textContent = '—';
    }
  };

  const renderMetrics = (metrics) => {
    const totalValue = formatCurrency(metrics.totalValue);
    const cash = formatCurrency(metrics.cashBalance);
    const positionsValue = formatCurrency(metrics.positionsValue);
    const buyingPower = formatCurrency(metrics.buyingPower);
    const updated = new Date().toLocaleTimeString();

    statusEl.innerHTML = [
      `<div style="font-size:1.1em; font-weight:600;">Total Value: ${totalValue}</div>`,
      `<div style="opacity:0.85;">Positions ${positionsValue} · Cash ${cash}</div>`,
      `<div style="opacity:0.8;">Buying Power ${buyingPower}</div>`,
      `<div style="opacity:0.6; font-size:0.85em;">Updated ${updated}</div>`,
    ].join('');
  };

  const renderBalances = (metrics) => {
    if (!pnlEl) {
      return;
    }
    const cash = formatCurrency(metrics.cashBalance);
    const positionsValue = formatCurrency(metrics.positionsValue);
    const buyingPower = formatCurrency(metrics.buyingPower);
    const available = formatCurrency(metrics.cashAvailableForTrading);
    const total = toNumber(metrics.totalValue);
    const invested = toNumber(metrics.positionsValue);
    const investedPct = Number.isFinite(total) && total > 0 && Number.isFinite(invested)
      ? ((invested / total) * 100).toFixed(1)
      : null;

    const rows = [
      `<div>Cash: ${cash}</div>`,
      `<div>Positions Value: ${positionsValue}</div>`,
      `<div>Buying Power: ${buyingPower}</div>`,
      `<div>Tradable Cash: ${available}</div>`,
    ];
    if (investedPct !== null) {
      rows.push(`<div>Invested Allocation: ${investedPct}%</div>`);
    }
    pnlEl.innerHTML = rows.join('');
  };

  const renderPositions = (portfolio) => {
    if (!positionsEl) {
      return;
    }
    if (!Array.isArray(portfolio) || portfolio.length === 0) {
      positionsEl.textContent = 'No active positions';
      return;
    }

    const rows = portfolio.map((position) => {
      const symbol = position.symbol || position.ticker || position.conidSymbol || '—';
      const quantity = formatQuantity(position.quantity ?? position.position ?? position.shares);
      const marketPrice = formatCurrency(position.currentPrice ?? position.marketPrice ?? position.lastPrice);
      const marketValueNumber = toNumber(position.marketValue);
      const fallbackMarketValue = toNumber(position.quantity ?? position.position) * toNumber(position.currentPrice ?? position.marketPrice);
      const marketValue = Number.isFinite(marketValueNumber) ? marketValueNumber : fallbackMarketValue;
      const unrealized = position.unrealizedPnl ?? position.unrealizedPNL ?? position.unrealizedPnL;
      const unrealizedValue = Number.isFinite(toNumber(unrealized)) ? toNumber(unrealized) : NaN;
      const unrealizedDisplay = formatSignedCurrency(unrealizedValue);
      const unrealizedStyle = Number.isFinite(unrealizedValue)
        ? (unrealizedValue > 0 ? 'color:#22c55e;' : unrealizedValue < 0 ? 'color:#ef4444;' : 'color:#94a3b8;')
        : 'color:#94a3b8;';

      return `
        <tr>
          <td>${symbol}</td>
          <td style="text-align:right;">${quantity}</td>
          <td style="text-align:right;">${marketPrice}</td>
          <td style="text-align:right;">${formatCurrency(marketValue)}</td>
          <td style="text-align:right;"><span style="${unrealizedStyle}">${unrealizedDisplay}</span></td>
        </tr>
      `;
    }).join('');

    positionsEl.innerHTML = `
      <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
        <thead>
          <tr style="text-align:left; border-bottom:1px solid rgba(148,163,184,0.3);">
            <th style="padding:4px 6px;">Symbol</th>
            <th style="padding:4px 6px; text-align:right;">Qty</th>
            <th style="padding:4px 6px; text-align:right;">Last</th>
            <th style="padding:4px 6px; text-align:right;">Market Value</th>
            <th style="padding:4px 6px; text-align:right;">Unrealized</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  };

  const renderOpenOrders = (openOrders) => {
    if (!ordersEl) {
      return;
    }
    let entries = [];
    if (Array.isArray(openOrders)) {
      entries = openOrders;
    } else if (openOrders && Array.isArray(openOrders.orders)) {
      entries = openOrders.orders;
    }
    if (!entries.length) {
      ordersEl.textContent = 'No open orders';
      return;
    }

    const rows = entries.map((order) => {
      const symbol = order.symbol || order.ticker || '—';
      const side = (order.action || order.side || '').toString().toUpperCase() || '—';
      const quantity = formatQuantity(order.quantity ?? order.totalQuantity ?? order.shares);
      const orderType = (order.orderType || order.type || '').toString().toUpperCase() || '—';
      const limitPrice = order.limitPrice ?? order.lmtPrice ?? order.auxPrice ?? order.stopPrice;
      const status = (order.status || order.orderStatus || '—').toString();
      const priceDisplay = limitPrice !== undefined ? formatCurrency(limitPrice) : '—';
      return `
        <tr>
          <td>${symbol}</td>
          <td style="text-align:center;">${side}</td>
          <td style="text-align:right;">${quantity}</td>
          <td style="text-align:center;">${orderType}</td>
          <td style="text-align:right;">${priceDisplay}</td>
          <td style="text-align:center;">${status}</td>
        </tr>
      `;
    }).join('');

    ordersEl.innerHTML = `
      <table style="width:100%; border-collapse:collapse; font-size:0.9em;">
        <thead>
          <tr style="text-align:left; border-bottom:1px solid rgba(148,163,184,0.3);">
            <th style="padding:4px 6px;">Symbol</th>
            <th style="padding:4px 6px; text-align:center;">Side</th>
            <th style="padding:4px 6px; text-align:right;">Qty</th>
            <th style="padding:4px 6px; text-align:center;">Type</th>
            <th style="padding:4px 6px; text-align:right;">Price</th>
            <th style="padding:4px 6px; text-align:center;">Status</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  };

  const refreshSummary = async () => {
    try {
  const response = await fetch('/api/rl/live/summary');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();
      if (payload.status !== 'ok') {
        console.warn('RL summary unavailable', payload);
        setErrorState(payload.detail || 'IBKR bridge unavailable');
        emitToast('error', payload.detail || 'Live summary unavailable.', 'rl-summary');
        return;
      }
      const metrics = payload.metrics || {};
      renderMetrics(metrics);
      renderBalances(metrics);
      renderPositions(payload.portfolio || []);
      renderOpenOrders(payload.openOrders);
    } catch (error) {
      console.error('Failed to refresh RL live summary', error);
      setErrorState('Error loading summary');
      emitToast('error', 'Failed to refresh live summary.', 'rl-summary');
    }
  };

  if (ibkrConnectBtn) {
    ibkrConnectBtn.addEventListener('click', connectIbkrBridge);
  }

  refreshSummary();
  refreshIbkrStatus();
  setInterval(refreshSummary, REFRESH_INTERVAL_MS);
  if (ibkrStatusEl) {
    setInterval(refreshIbkrStatus, IBKR_STATUS_INTERVAL_MS);
  }
})();