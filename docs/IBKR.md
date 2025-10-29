# IBKR Bridge (Stub) API

This documents the interim IBKR endpoints exposed by the FastAPI backend and proxied by the Flask UI. They provide a stable contract for the RL tab while the external C# IBKR bridge integration is being finalized.

Base URLs:
- Backend (FastAPI): http://localhost:8000
- UI proxy (Flask): http://localhost:5000

Use either base; UI calls should prefer the Flask proxy to avoid CORS.

## Endpoints

### POST /api/ibkr/connect
Connect (or mark as connected) to the IBKR bridge.

Request (JSON):
{
  "host": "127.0.0.1",   // optional
  "port": 4001,           // optional
  "client_id": 1          // optional
}

Response (JSON):
{
  "status": "success",
  "data": {
    "connected": true,
    "host": "127.0.0.1",
    "port": 4001,
    "client_id": 1
  }
}

### GET /api/ibkr/status
Returns current connection state and simple counters.

Response:
{
  "status": "success",
  "data": {
    "connected": true,
    "host": "127.0.0.1",
    "port": 4001,
    "client_id": 1,
    "orders_count": 3,
    "positions_count": 1
  }
}

### POST /api/ibkr/place_order
Place a simulated order. The stub fills immediately and updates in-memory positions.

Request:
{
  "symbol": "AAPL",
  "quantity": 10,
  "side": "BUY",          // BUY | SELL
  "order_type": "MKT",    // MKT | LMT
  "limit_price": 190.5      // required when order_type=LMT
}

Response:
{
  "status": "success",
  "data": {
    "ok": true,
    "order_id": "SIM-1700000000000",
    "status": "filled"
  }
}

Errors:
- 400 Invalid side or order_type
- 503 Not connected

### GET /api/ibkr/positions
List aggregate positions maintained by the stub.

Response:
{
  "status": "success",
  "data": {
    "positions": [
      {"symbol": "AAPL", "quantity": 10}
    ]
  }
}

### POST /api/ibkr/disconnect
Mark the client as disconnected.

Response:
{
  "status": "success",
  "data": {"connected": false}
}

## Notes
- These endpoints are backed by an in-memory stub at `app/integrations/ibkr_client.py`. They are non-persistent and reset on process restart.
- Replace the stub with real C# bridge calls by swapping the client implementation while keeping the same API shape.
- The Flask server proxies these endpoints at the same paths, forwarding to the FastAPI backend configured by `FASTAPI_BACKEND`.
