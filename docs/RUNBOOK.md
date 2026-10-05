# Runbook

## Start / stop
- Local: `uvicorn code.app.main:app --host 0.0.0.0 --port 8731` (see README for env).
- Docker: `docker compose up -d` / `docker compose down`.
- Check: `curl http://localhost:8731/healthz` → `"status": "ok"`.

## Daily schedule (IST)
- 08:30 refresh, 18:30 nightly pipeline (screen, setups, market health, predictions, settle paper trades, AI reviews).
- 1st of month 20:00: pairs cointegration rescan.

## Emergency halt
Blocks every new order immediately (existing paper positions are untouched).
```bash
curl -X POST -b "<session cookie>" -H "Origin: http://localhost:8731" -d "on=1&reason=..." http://localhost:8731/api/halt
```
Turn off with `on=0`. Both are written to the audit log. (A halt button in the UI lands with L4/L6.)

## Stale data
- Amber "Data from …" banner = last pipeline didn't finish. Run `python -m code.pipeline.nightly_pipeline` and check its log.
- `/healthz` → `snapshot_stale: true` means the same.

## Audit log integrity
```bash
python -c "from code.live import audit; print(audit.verify())"
```
`{"ok": false, "broken_at": N}` means row N or the row before it was altered — investigate before trusting history.

## NSE holidays
Fetched from NSE weekly into `code/data/nse_holidays.json`. If NSE is unreachable the cached list is used.
Force a refresh by deleting that file.

## Secrets
Only in `.env` / the host's environment. Rotate a leaked key at the provider, update `.env`, restart.
