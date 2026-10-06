// Live prices during market hours. Polls /api/quotes (the server's quote feed) and fills any element
// tagged with data-live-* for that symbol. Server-rendered values (last night's close) stay as the
// fallback whenever there's no fresh quote.
(function () {
  const POLL_MS = 30000;
  const inr = new Intl.NumberFormat("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const each = (attr, fn) => document.querySelectorAll("[" + attr + "]").forEach(el => fn(el, el.getAttribute(attr)));
  const tone = (el, v) => { el.classList.toggle("up", v > 0); el.classList.toggle("down", v < 0); };

  function apply(data) {
    const q = data.quotes || {};
    const open = data.session && data.session.phase === "OPEN";
    each("data-live-px", (el, s) => { if (q[s]) el.textContent = (el.dataset.prefix || "") + inr.format(q[s].ltp); });
    each("data-live-chg", (el, s) => {
      if (!q[s]) return;
      el.textContent = (q[s].chg_pct > 0 ? "+" : "") + q[s].chg_pct.toFixed(2) + "%";
      tone(el, q[s].chg_pct);
    });
    // Setups: where price is against the pivot right now (Minervini buy range = up to 5% past it).
    each("data-live-pivot", (el, s) => {
      const pivot = parseFloat(el.dataset.pivot);
      if (!q[s] || !pivot) return;
      const d = (q[s].ltp / pivot - 1) * 100;
      el.textContent = d > 5 ? "extended " + d.toFixed(1) + "% past pivot — don't chase"
        : d >= 0 ? "in buy range, " + d.toFixed(1) + "% past pivot"
        : Math.abs(d).toFixed(1) + "% below pivot";
      el.className = "live-pivot " + (d > 5 ? "down" : d >= 0 ? "up" : "muted");
    });
    // Paper positions: mark-to-market on the live price.
    each("data-live-pnl", (el, s) => {
      if (!q[s]) return;
      const sign = el.dataset.side === "SELL" ? -1 : 1;
      const pnl = sign * (q[s].ltp - parseFloat(el.dataset.entry)) * parseFloat(el.dataset.qty);
      el.textContent = (pnl < 0 ? "−₹" : "₹") + Math.round(Math.abs(pnl)).toLocaleString("en-IN");
      tone(el, pnl);
    });
    const t = data.last_poll ? new Date(data.last_poll).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" }) : "";
    each("data-live-status", el => {
      el.hidden = !(open && data.last_poll);
      el.textContent = "Live · " + t;
    });
  }

  async function poll() {
    try {
      const r = await fetch("/api/quotes", { credentials: "same-origin" });
      if (r.ok) apply(await r.json());
    } catch (e) { /* offline: keep what's on screen */ }
  }

  poll();
  setInterval(() => { if (!document.hidden) poll(); }, POLL_MS);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
})();
