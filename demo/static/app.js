/*
 * FL-IDS live demo frontend: "Flight Strip Board" (design A).
 *
 * README
 * ------
 * Files (all in demo/static/, served by demo/server.py at /):
 *   index.html   page skeleton: header + manila tabs, Live Detection view, Training Run view
 *   style.css    the whole look; palette and sizing tokens live on :root, media queries tune
 *                sizes for 1366x768 laptops up to 1920x1080 projectors
 *   app.js       this file: transport, state, rendering, controls (no framework, no build)
 *   mock.js      in-browser simulator, loaded ONLY when the URL has ?mock=1
 *   fonts/       vendored woff2 (Bebas Neue, Courier Prime, Roboto Slab) plus licences
 *
 * How it talks to the backend (see demo/CONTRACT.md, the binding spec):
 *   - WebSocket /ws: first a `snapshot`, then msg / incident / counters / status / config /
 *     training / toast events. Auto-reconnects with exponential backoff (0.5 s up to 8 s); every
 *     reconnect gets a fresh snapshot and the page re-renders from it.
 *   - REST: /api/inject, /api/strength, /api/stream, /api/reset, /api/training/*.
 *   - ?mock=1 swaps both for window.FLMock (mock.js), with the same event shapes.
 *
 * Code map (sections below, in order):
 *   1. constants and copy      stamp names per attack type, quips, ticker lines
 *   2. transport               request() and openSocket(), real or mock
 *   3. live state + events     incoming events go into a queue; one flush per animation frame
 *   4. racks                   building racks, strips, status treatments, bay sizing
 *   5. incident log            incremental rows, rate-limited aria-live summary
 *   6. controls                Mischief Desk, dial, stream switch, speed, reset
 *   7. ticker, clock, header
 *   8. training view           replay and live training, SVG chart and sparklines
 *   9. tabs, toasts, boot
 *
 * Performance notes: at 6 clients x 8 msgs/s the page gets about 50 events per second plus
 * incidents. Events are only queued on arrival; DOM work happens once per frame. Each bay
 * holds at most 8 strip nodes and the log at most 12 rows, so node counts stay flat.
 */
(function () {
  'use strict';

  /* ===================== 1. constants and copy ===================== */

  var MOCK = /(?:^|[?&])mock=1(?:&|$)/.test(location.search);

  var STAMP = {
    ghost_aircraft: 'GHOST', gps_spoof: 'SPOOFED', velocity_spoof: 'VELOCITY?',
    replay: 'REPLAYED', flooding: 'FLOODED'
  };
  var SHORT = {
    ghost_aircraft: 'ghost', gps_spoof: 'gps', velocity_spoof: 'speed',
    replay: 'replay', flooding: 'flood'
  };
  var ATTACKS = ['gps_spoof', 'velocity_spoof', 'replay', 'ghost_aircraft', 'flooding'];

  var QUIP_NORMAL = [
    'Roger, all benign.', 'Copy. Nothing to report, honest.', 'Tower, standing by.',
    'Skies clean. Coffee less so.', 'Squawking normal.', 'Radar says hi.'
  ];
  var QUIP_NOISY = ['Seeing things. Disregard.', 'Jumpy today. Dial it up?', 'Was that a plane? Probably.'];
  var QUIP_HOT = {
    ghost_aircraft: 'Mayday. Ghosts in the pattern.',
    gps_spoof: 'GPS says you are in a cornfield.',
    velocity_spoof: 'Say again your airspeed?',
    replay: 'Did we not clear this one already?',
    flooding: 'Frequency jammed. Everyone talking at once.'
  };
  var TICKER_QUIET = [
    'Negative contact. Suspiciously quiet.',
    'All towers report benign. The model is bored.',
    'Nothing to see here. Literally, we checked.',
    'Six towers, zero drama. Enjoy it while it lasts.',
    'Standing by. Coffee is cold, skies are clean.',
    'No ghosts detected. The tower cat remains unconvinced.',
    'In training, only weights travel. The coffee stays put.',
    'Federated, vigilant, slightly over-caffeinated.',
    'The lever is right there. Just saying.'
  ];

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (ch) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch];
    });
  }
  function fmtCount(n) {
    n = Number(n) || 0;
    if (n < 10000) return n.toLocaleString('en-US');
    if (n < 100000) return (n / 1000).toFixed(1) + 'k';
    if (n < 1e6) return Math.round(n / 1000) + 'k';
    return (n / 1e6).toFixed(2) + 'M';
  }
  function fmtBytes(b) {
    b = Number(b) || 0;
    if (b < 1000) return b + ' B';
    if (b < 1e6) return (b / 1000).toFixed(1) + ' KB';
    if (b < 1e9) return (b / 1e6).toFixed(2) + ' MB';
    return (b / 1e9).toFixed(2) + ' GB';
  }
  function fmtTime(t) {
    if (t == null) return '--:--:--';
    if (typeof t === 'number') { // unix seconds, just in case
      var d = new Date(t * 1000);
      return d.toISOString().slice(11, 19);
    }
    var m = String(t).match(/(\d{2}:\d{2}:\d{2})/);
    return m ? m[1] : String(t).slice(0, 8);
  }
  function fmtStrength(s) { return (Math.round(Number(s) * 100) / 100).toFixed(2); }
  function pick(arr, seed) { return arr[Math.abs(seed | 0) % arr.length]; }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  /* ===================== 2. transport ===================== */

  var transport = {
    request: function (method, path, body) {
      if (MOCK) return window.FLMock.request(method, path, body);
      var opts = { method: method, headers: {} };
      if (body !== undefined) {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(body);
      }
      return fetch(path, opts).then(function (res) {
        return res.text().then(function (txt) {
          var data = null;
          try { data = txt ? JSON.parse(txt) : null; } catch (e) { data = { detail: txt }; }
          if (!res.ok) {
            var detail = data && (data.detail || data.error || data.message);
            if (detail && typeof detail !== 'string') detail = JSON.stringify(detail);
            var err = new Error(detail || ('HTTP ' + res.status));
            err.status = res.status;
            throw err;
          }
          return data;
        });
      });
    },

    openSocket: function (onEvent, onState) {
      if (MOCK) {
        onState('mock');
        return window.FLMock.connect(onEvent);
      }
      var delay = 500, ws = null, everOpen = false, timer = null;
      function open() {
        onState(everOpen ? 'reconnecting' : 'connecting');
        var proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
        try { ws = new WebSocket(proto + location.host + '/ws'); }
        catch (e) { schedule(); return; }
        ws.onopen = function () { everOpen = true; delay = 500; onState('connected'); };
        ws.onmessage = function (e) {
          var ev;
          try { ev = JSON.parse(e.data); } catch (err) { console.warn('bad ws frame', err); return; }
          onEvent(ev);
        };
        ws.onclose = function () { ws = null; schedule(); };
        ws.onerror = function () { try { ws && ws.close(); } catch (e) { /* ignore */ } };
      }
      function schedule() {
        onState(everOpen ? 'reconnecting' : 'connecting');
        clearTimeout(timer);
        timer = setTimeout(open, delay + Math.random() * 250);
        delay = Math.min(delay * 2, 8000);
      }
      open();
      return { close: function () { clearTimeout(timer); if (ws) ws.close(); } };
    }
  };

  /* ===================== 3. live state + event queue ===================== */

  var S = {
    clients: new Map(),   // id -> client record (see newClient)
    order: [],
    strength: 0.5, running: true, rate: 4,
    model: null,
    incidents: [],        // newest first, capped at 50
    lastTime: null,
    snapshotSeen: false
  };
  var queue = [];
  var flushScheduled = false;
  var logDirty = false, logFresh = 0, logRebuild = false;
  var measureNeeded = true, tightNeeded = true;
  var pendingAnnounce = { n: 0, last: null };

  function enqueue(ev) {
    if (!ev || typeof ev !== 'object') return;
    queue.push(ev);
    // cap the backlog if the tab was hidden for a while; a snapshot will follow on reconnect anyway
    if (queue.length > 4000) queue.splice(0, queue.length - 2000);
    scheduleFlush();
  }
  function scheduleFlush() {
    if (flushScheduled) return;
    flushScheduled = true;
    var done = false;
    var run = function () { if (done) return; done = true; flushScheduled = false; flush(); };
    requestAnimationFrame(run);
    setTimeout(run, 200); // rAF does not fire in hidden tabs
  }

  function flush() {
    var evs = queue; queue = [];
    for (var i = 0; i < evs.length; i++) {
      try { handle(evs[i]); } catch (e) { console.error('event failed', evs[i], e); }
    }
    if (measureNeeded) { measureNeeded = false; measureAll(); }
    S.clients.forEach(function (c) {
      if (c.bayDirty) renderBay(c);
      if (c.cntDirty) renderCounters(c);
      if (c.chipDirty) renderChip(c);
    });
    if (logDirty) renderLog();
    if (tightNeeded) { tightNeeded = false; checkTight(); }
    renderClock();
  }

  // hide the quip line on placards that would otherwise overflow (short screens, two-line names)
  function checkTight() {
    S.clients.forEach(function (c) {
      if (!c.el) return;
      var pl = c.el.quip.parentNode;
      pl.classList.remove('tight');
      if (pl.scrollHeight > pl.clientHeight + 1) pl.classList.add('tight');
    });
  }

  function handle(ev) {
    switch (ev.type) {
      case 'snapshot': applySnapshot(ev); break;
      case 'msg': onMsg(ev); break;
      case 'incident': onIncident(ev); break;
      case 'counters': {
        var c = S.clients.get(ev.client);
        if (c && ev.counters) { c.counters = Object.assign({}, c.counters, ev.counters); c.cntDirty = true; }
        break;
      }
      case 'status': {
        var c2 = S.clients.get(ev.client);
        if (c2) setStatus(c2, ev.status);
        break;
      }
      case 'config': applyConfig(ev); break;
      case 'training': onTrainingEvent(unwrapTraining(ev)); break;
      case 'toast': toast(ev.text || '', /error|fail|cannot|could not|409/i.test(ev.text || '') || ev.level === 'error'); break;
      default:
        // a raw training line forwarded without an envelope
        if (ev.type === 'run_start' || ev.type === 'round' || ev.type === 'client_fit' || ev.type === 'run_end' || ev.type === 'run_error') onTrainingEvent(ev);
    }
  }

  function newClient(c) {
    return {
      id: c.id, cell: c.cell, name: c.name, lat: c.lat, lon: c.lon,
      status: 'normal',
      counters: Object.assign({ seen: 0, flagged: 0, correct: 0, false_alarm: 0, missed: 0 }, c.counters || {}),
      recent: [],          // newest first, max 8
      fresh: 0,            // strips not yet in the DOM
      cap: 6,              // strips that fit in the bay
      alerts: [],          // {at, pred} correct catches, for the chip
      lastHotType: null,
      el: null, bayDirty: true, cntDirty: true, chipDirty: true,
      quipSeed: Math.floor(Math.random() * 100)
    };
  }

  function bySeqDesc(a, b) { return (Number(b.seq) || 0) - (Number(a.seq) || 0); }

  function applySnapshot(snap) {
    var list = Array.isArray(snap.clients) ? snap.clients : [];
    var ids = list.map(function (c) { return c.id; });
    var same = ids.length === S.order.length && ids.every(function (id, i) { return id === S.order[i]; });
    if (!same) {
      S.clients.clear();
      list.forEach(function (c) { S.clients.set(c.id, newClient(c)); });
      S.order = ids;
      buildRacks();
      buildTargetSelect();
    }
    list.forEach(function (sc) {
      var c = S.clients.get(sc.id);
      c.name = sc.name; c.cell = sc.cell;
      c.counters = Object.assign({ seen: 0, flagged: 0, correct: 0, false_alarm: 0, missed: 0 }, sc.counters || {});
      var rec = (sc.recent || []).slice().sort(bySeqDesc).slice(0, 8);
      c.recent = rec;
      c.fresh = rec.length;
      if (c.el) c.el.bay.textContent = '';
      c.alerts = [];
      c.bayDirty = c.cntDirty = c.chipDirty = true;
      c.status = null; // force re-apply without entrance animation
      setStatus(c, sc.status || 'normal', true);
      if (rec[0]) S.lastTime = rec[0].time;
    });
    S.incidents = (snap.incidents || []).slice().sort(bySeqDesc).slice(0, 50);
    // after a reload mid-attack, recover each tower's latest caught attack type from the snapshot
    S.clients.forEach(function (c) {
      var hit = c.recent.concat(S.incidents).filter(function (m) {
        return m.client === c.id && m.verdict === 'correct_catch' && m.pred && m.pred !== 'benign';
      }).sort(bySeqDesc)[0];
      c.lastHotType = hit ? hit.pred : null;
      c.chipDirty = true;
    });
    logRebuild = true; logDirty = true;
    applyConfig(snap);
    if (snap.model) S.model = snap.model;
    S.snapshotSeen = true;
    renderInfo();
    tickerNow();
    measureNeeded = true;
  }

  function onMsg(m) {
    var c = S.clients.get(m.client);
    if (!c) return;
    c.recent.unshift(m);
    if (c.recent.length > 8) c.recent.length = 8;
    c.fresh = Math.min(c.fresh + 1, 8);
    c.bayDirty = true;
    if (m.pred && m.pred !== 'benign' && m.verdict === 'correct_catch') {
      c.alerts.push({ at: Date.now(), pred: m.pred });
      if (c.alerts.length > 40) c.alerts.splice(0, c.alerts.length - 40);
      c.lastHotType = m.pred;
      c.chipDirty = true;
    }
    if (m.time) S.lastTime = m.time;
  }

  function onIncident(inc) {
    S.incidents.unshift(inc);
    if (S.incidents.length > 50) S.incidents.length = 50;
    logFresh++; logDirty = true;
    pendingAnnounce.n++; pendingAnnounce.last = inc;
  }

  /* ===================== 4. racks ===================== */

  function buildRacks() {
    var host = $('racks');
    host.textContent = '';
    S.order.forEach(function (id) {
      var c = S.clients.get(id);
      var rack = document.createElement('article');
      rack.className = 'rack wood';
      rack.innerHTML =
        '<div class="hazard" aria-hidden="true"><b class="live"><span class="lamp"></span>LIVE</b><b>ATTACK IN PROGRESS</b></div>' +
        '<div class="placard">' +
          '<span class="screw l"></span><span class="screw r"></span>' +
          '<span class="p-id"><b>' + esc(c.id) + '</b><span class="p-cell">' + esc(c.cell) + '</span></span>' +
          '<span class="stamp stamp-big" aria-hidden="true">Under<br>attack</span>' +
          '<span class="p-name">' + esc(c.name) + '</span>' +
          '<span class="p-cell p-cell2">cell ' + esc(c.cell) + '</span>' +
          '<span class="chip">Normal</span>' +
          '<span class="cnt">SEEN <b data-k="seen">0</b></span>' +
          '<span class="cnt hotc">FLAGGED <b data-k="flagged">0</b></span>' +
          '<span class="cnt hotc">CORRECT <b data-k="correct">0</b></span>' +
          '<span class="cnt">FALSE ALM <b data-k="false_alarm">0</b></span>' +
          '<span class="cnt missed">MISSED <b data-k="missed">0</b></span>' +
          '<span class="quip"></span>' +
        '</div>' +
        '<div class="bay"></div>';
      host.appendChild(rack);
      var cnt = {};
      rack.querySelectorAll('b[data-k]').forEach(function (b) { cnt[b.getAttribute('data-k')] = b; });
      c.el = {
        rack: rack, bay: rack.querySelector('.bay'), chip: rack.querySelector('.chip'),
        quip: rack.querySelector('.quip'), cnt: cnt
      };
      c.bayDirty = c.cntDirty = c.chipDirty = true;
      updateRackLabel(c);
    });
    measureNeeded = true;
  }

  function updateRackLabel(c) {
    if (!c.el) return;
    var st = c.status === 'under_attack' ? 'under attack' : (c.status || 'normal');
    c.el.rack.setAttribute('aria-label', c.id + ' ' + c.name + ', cell ' + c.cell + ', status ' + st);
  }

  function setStatus(c, st, silent) {
    if (st !== 'under_attack' && st !== 'noisy') st = 'normal';
    if (c.status === st) return;
    c.status = st;
    if (!c.el) return;
    var r = c.el.rack;
    r.classList.toggle('hot', st === 'under_attack');
    r.classList.toggle('noisy', st === 'noisy');
    r.classList.remove('enter');
    if (st === 'under_attack' && !silent) {
      void r.offsetWidth; // restart the entrance animation
      r.classList.add('enter');
      setTimeout(function () { r.classList.remove('enter'); }, 1300);
    }
    c.chipDirty = true;
    updateRackLabel(c);
    measureNeeded = true; // hot racks have a taller top edge, so fewer strips fit
    scheduleFlush();
    tickerNow();
  }

  function measureAll() {
    var probe = getComputedStyle(document.documentElement);
    var sh = parseFloat(probe.getPropertyValue('--strip-h')) || 32;
    var gap = parseFloat(probe.getPropertyValue('--strip-gap')) || 4;
    S.clients.forEach(function (c) {
      if (!c.el) return;
      var h = c.el.bay.clientHeight - 10; // bay padding
      var cap = clamp(Math.floor((h + gap) / (sh + gap)), 1, 8);
      if (cap !== c.cap) {
        c.cap = cap;
        c.el.bay.textContent = '';
        c.fresh = Math.min(c.recent.length, cap);
        c.bayDirty = true;
      }
    });
    tightNeeded = true;
    var wrap = $('logWrap');
    if (wrap) {
      var rowH = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--fs-data')) > 14 ? 28 : 27;
      var lcap = clamp(Math.floor((wrap.clientHeight - 24) / rowH), 1, 12);
      if (lcap !== logCap) { logCap = lcap; logRebuild = true; logDirty = true; }
    }
  }

  function stripHTML(m) {
    var pred = m.pred || 'benign';
    var v = m.verdict || (pred === 'benign' ? (m.truth && m.truth !== 'benign' ? 'missed' : 'ok') : (m.truth === 'benign' ? 'false_alarm' : 'correct_catch'));
    var seq = Number(m.seq) || 0;
    var rot = '--r:' + (-6 - (seq % 6)) + 'deg';
    var cls = 'strip', label, mark = '';
    if (v === 'false_alarm') {
      cls += ' alert fa';
      label = '<span class="c p two">' + esc(pred) + '<small>truth: benign</small></span>';
      mark = '<span class="stamp oops" style="' + rot + '">OOPS<span class="sub">(FALSE ALARM)</span></span>';
    } else if (pred !== 'benign') {
      cls += ' alert';
      if (m.wrong_type && m.truth) label = '<span class="c p two">' + esc(pred) + '<small>truth: ' + esc(m.truth) + '</small></span>';
      else label = '<span class="c p">' + esc(pred) + '</span>';
      mark = '<span class="stamp" style="' + rot + '">' + esc(STAMP[pred] || pred.toUpperCase()) + '</span>';
    } else if (v === 'missed') {
      cls += ' missed';
      label = '<span class="c p two ben">benign<small>truth: ' + esc(m.truth) + '</small></span>';
      mark = '<span class="missed-mark">MISSED</span>';
    } else {
      label = '<span class="c p ben">benign</span>';
    }
    var inj = m.injected ? '<span class="inj-tag" title="injected by the Mischief Desk' + (m.orig_time ? '; real row time ' + esc(m.orig_time) + ' UTC' : '') + '">INJ</span>' : '';
    return { cls: cls, html: inj + '<span class="c">' + esc(fmtTime(m.time)) + '</span><span class="c">' + esc(m.icao24 || '------') + '</span>' + label + mark };
  }

  function renderBay(c) {
    c.bayDirty = false;
    if (!c.el) return;
    var bay = c.el.bay, cap = c.cap;
    var empties = bay.querySelectorAll('.empty');
    for (var e = 0; e < empties.length; e++) empties[e].remove();
    var fresh = Math.min(c.fresh, cap, c.recent.length);
    c.fresh = 0;
    for (var i = fresh - 1; i >= 0; i--) {
      var s = stripHTML(c.recent[i]);
      var d = document.createElement('div');
      d.className = s.cls + (c.recent[i].injected ? ' injd' : '') + (i < 3 ? ' enter' : '');
      d.innerHTML = s.html;
      bay.insertBefore(d, bay.firstChild);
    }
    while (bay.children.length > cap) bay.removeChild(bay.lastChild);
    var have = bay.children.length;
    for (var k = have; k < cap; k++) {
      var em = document.createElement('div');
      em.className = 'empty';
      if (k === have) em.textContent = S.running ? 'Negative contact. Suspiciously quiet.' : 'Stream paused. Tower on break.';
      bay.appendChild(em);
    }
  }

  function renderCounters(c) {
    c.cntDirty = false;
    if (!c.el) return;
    ['seen', 'flagged', 'correct', 'false_alarm', 'missed'].forEach(function (k) {
      var txt = fmtCount(c.counters[k]);
      if (c.el.cnt[k].textContent !== txt) c.el.cnt[k].textContent = txt;
    });
  }

  function renderChip(c) {
    c.chipDirty = false;
    if (!c.el) return;
    var chip = c.el.chip, now = Date.now();
    c.alerts = c.alerts.filter(function (a) { return now - a.at < 15000; });
    var text, cls = 'chip';
    if (c.status === 'noisy') { text = 'Noisy'; cls += ' noisy'; }
    else if (c.status === 'under_attack') { text = 'Under attack'; }
    else if (c.alerts.length) {
      var last = c.alerts[c.alerts.length - 1].pred;
      var n = c.alerts.length;
      text = n === 1 ? '1 alert: ' + (SHORT[last] || last) : (n < 10 ? n + ' alerts: ' + (SHORT[last] || last) : n + ' alerts');
      cls += ' alerted';
    } else { text = 'Normal'; }
    if (chip.textContent !== text) chip.textContent = text;
    if (chip.className !== cls) chip.className = cls;
    var quip;
    if (c.status === 'under_attack') quip = QUIP_HOT[c.lastHotType] || QUIP_HOT.ghost_aircraft;
    else if (c.status === 'noisy') quip = pick(QUIP_NOISY, c.quipSeed);
    else quip = pick(QUIP_NORMAL, c.quipSeed + S.order.indexOf(c.id));
    if (c.el.quip.textContent !== quip) { c.el.quip.textContent = quip; tightNeeded = true; }
  }

  /* ===================== 5. incident log ===================== */

  var logCap = 6;

  function incidentRow(inc) {
    var v = inc.verdict === 'false_alarm' ? 'fa' : (inc.wrong_type ? 'wt' : 'ok');
    var vtxt = v === 'fa' ? 'false alarm' : (v === 'wt' ? 'wrong type' : 'correct catch');
    var c = S.clients.get(inc.client);
    var name = inc.name || (c && c.name) || '';
    var conf = inc.conf == null ? '-' : Number(inc.conf).toFixed(2);
    var tr = document.createElement('tr');
    if (v === 'fa') tr.className = 'fa-row';
    if (inc.injected) tr.title = 'Injected row' + (inc.orig_time ? ', real row time ' + inc.orig_time + ' UTC' : '');
    tr.innerHTML = '<td>' + esc(fmtTime(inc.time)) + '</td><td>' + esc(inc.client + ' ' + name) + '</td><td class="atk-cell">' + esc(inc.pred) +
      '</td><td>' + conf + '</td><td>' + esc(inc.icao24 || '') + '</td><td><span class="v-' + v + '">' + vtxt + '</span></td>';
    return tr;
  }

  function renderLog() {
    logDirty = false;
    var body = $('logBody');
    var fresh = logFresh; logFresh = 0;
    if (logRebuild) {
      logRebuild = false;
      body.textContent = '';
      fresh = 0;
      S.incidents.slice(0, logCap).forEach(function (inc) { body.appendChild(incidentRow(inc)); });
    } else {
      var empty = body.querySelector('.empty-row');
      if (empty && S.incidents.length) empty.remove();
      var n = Math.min(fresh, logCap, S.incidents.length);
      for (var i = n - 1; i >= 0; i--) {
        var tr = incidentRow(S.incidents[i]);
        tr.classList.add('new');
        body.insertBefore(tr, body.firstChild);
      }
      while (body.children.length > logCap) body.removeChild(body.lastChild);
    }
    if (!S.incidents.length && !body.children.length) {
      body.innerHTML = '<tr class="empty-row"><td colspan="6">Clean sheet. Not a single incident. Yet.</td></tr>';
    }
    $('logNote').textContent = 'Newest on top, per the regs. ' + S.incidents.length + (S.incidents.length === 1 ? ' entry.' : ' entries.');
  }

  // aria-live summary at most every 6 s, so screen readers are not flooded
  setInterval(function () {
    if (!pendingAnnounce.n) return;
    var inc = pendingAnnounce.last, n = pendingAnnounce.n;
    pendingAnnounce.n = 0;
    var v = inc.verdict === 'false_alarm' ? 'false alarm' : (inc.wrong_type ? 'wrong type' : 'correct catch');
    $('logLive').textContent = n + (n === 1 ? ' new incident. ' : ' new incidents. ') + 'Latest: ' + inc.pred + ' at ' + inc.client + ' ' + (inc.name || '') + ', ' + v + '.';
  }, 6000);

  /* ===================== 6. controls ===================== */

  function buildTargetSelect() {
    var sel = $('target');
    var prev = sel.value;
    sel.textContent = '';
    S.order.forEach(function (id) {
      var c = S.clients.get(id);
      var o = document.createElement('option');
      o.value = id; o.textContent = id + ' ' + c.name;
      sel.appendChild(o);
    });
    if (prev && S.clients.has(prev)) sel.value = prev;
    else if (S.clients.has('C2')) sel.value = 'C2';
    updatePullSummary();
  }

  function selectedAttack() {
    var r = document.querySelector('input[name="atk"]:checked');
    return r ? r.value : 'ghost_aircraft';
  }
  function burstValue() {
    var v = parseInt($('burst').value, 10);
    if (!isFinite(v)) v = 25;
    return clamp(v, 1, 200);
  }
  function updatePullSummary() {
    $('pullSummary').textContent = selectedAttack() + ' x' + burstValue() + ' at ' + ($('target').value || '-');
    document.querySelectorAll('.rad').forEach(function (l) { l.classList.toggle('on', l.querySelector('input').checked); });
  }

  var noteTimer = null;
  function setInjectNote(text, ok) {
    var n = $('injectNote');
    n.textContent = text;
    n.classList.toggle('ok', !!ok);
    clearTimeout(noteTimer);
    noteTimer = setTimeout(function () {
      n.textContent = 'Mischief Desk: authorised personnel and chaos goblins only';
      n.classList.remove('ok');
    }, 6000);
  }

  function initControls() {
    $('target').addEventListener('change', updatePullSummary);
    document.querySelectorAll('input[name="atk"]').forEach(function (r) { r.addEventListener('change', updatePullSummary); });
    $('burst').addEventListener('input', updatePullSummary);
    $('burst').addEventListener('change', function () { $('burst').value = burstValue(); updatePullSummary(); });
    $('burstDec').addEventListener('click', function () { $('burst').value = clamp(burstValue() - 5, 1, 200); updatePullSummary(); });
    $('burstInc').addEventListener('click', function () { $('burst').value = clamp(burstValue() + 5, 1, 200); updatePullSummary(); });

    $('injectForm').addEventListener('submit', function (e) {
      e.preventDefault();
      var btn = $('pullBtn');
      if (btn.disabled) return;
      var client = $('target').value, attack = selectedAttack(), count = burstValue();
      if (!client) { toast('No towers on the board yet.', true); return; }
      btn.disabled = true;
      setTimeout(function () { btn.disabled = false; }, 1000);
      var lever = $('lever');
      lever.classList.remove('pulled'); void lever.offsetWidth; lever.classList.add('pulled');
      transport.request('POST', '/api/inject', { client: client, attack: attack, count: count }).then(function (res) {
        var id = (res && res.burst_id) || '?';
        var q = (res && res.queued) || count;
        setInjectNote('Burst ' + id + ' away: ' + q + ' x ' + attack + ' into ' + client + '. Watch the rack.', true);
        tickerFlash('Mischief logged: ' + q + ' ' + attack + ' to ' + client + ' (' + id + ').');
      }).catch(function (err) {
        setInjectNote('Lever jammed: ' + err.message, false);
        toast('Inject failed: ' + err.message, true);
      });
    });

    // speed, pause, reset
    document.querySelectorAll('input[name="rate"]').forEach(function (r) {
      r.addEventListener('change', function () { postStream(S.running, Number(r.value)); });
    });
    $('runBtn').addEventListener('click', function () { postStream(!S.running, S.rate); });
    $('resetBtn').addEventListener('click', function () {
      var b = $('resetBtn');
      b.disabled = true;
      transport.request('POST', '/api/reset').then(function () {
        S.clients.forEach(function (c) {
          c.counters = { seen: 0, flagged: 0, correct: 0, false_alarm: 0, missed: 0 };
          c.alerts = []; c.cntDirty = c.chipDirty = true;
        });
        S.incidents = []; logRebuild = true; logDirty = true;
        scheduleFlush();
        tickerFlash('Counters zeroed. Fresh sheet, fresh start.');
      }).catch(function (err) { toast('Reset failed: ' + err.message, true); })
        .then(function () { b.disabled = false; });
    });

    initDial();
  }

  function postStream(running, rate) {
    var prev = { running: S.running, rate: S.rate };
    applyConfig({ running: running, rate: rate });
    transport.request('POST', '/api/stream', { running: running, rate: rate }).catch(function (err) {
      applyConfig(prev);
      toast('Stream control failed: ' + err.message, true);
    });
  }

  function applyConfig(cfg) {
    if (cfg.running != null) {
      var wasRunning = S.running;
      S.running = !!cfg.running;
      $('runBtn').setAttribute('aria-pressed', S.running ? 'true' : 'false');
      $('runLabel').textContent = S.running ? 'Streaming' : 'Paused';
      if (wasRunning !== S.running) { S.clients.forEach(function (c) { c.bayDirty = true; }); tickerNow(); scheduleFlush(); }
    }
    if (cfg.rate != null) {
      S.rate = Number(cfg.rate);
      var r = document.querySelector('input[name="rate"][value="' + S.rate + '"]');
      document.querySelectorAll('input[name="rate"]').forEach(function (x) { x.checked = (x === r); });
    }
    if (cfg.strength != null && !dial.dragging && !dial.pending) {
      S.strength = Number(cfg.strength);
      setDialVisual(S.strength);
      $('corr').value = S.strength;
    }
    renderInfo();
  }

  // the bakelite dial: a real range input drives it; the knob and marks are mouse sugar
  var dial = { dragging: false, pending: false, timer: null };
  function setDialVisual(v) {
    v = clamp(Number(v) || 0, 0, 1);
    $('knob').style.setProperty('--v', v);
    $('corrVal').textContent = fmtStrength(v) + ' set';
    document.querySelectorAll('.dial-mark').forEach(function (m) {
      m.classList.toggle('on', Math.abs(Number(m.getAttribute('data-v')) - v) < 0.001);
    });
  }
  function dialInput(v, immediate) {
    v = clamp(Math.round(v * 20) / 20, 0, 1);
    $('corr').value = v;
    S.strength = v;
    setDialVisual(v);
    renderInfo();
    dial.pending = true;
    clearTimeout(dial.timer);
    dial.timer = setTimeout(function () {
      transport.request('POST', '/api/strength', { strength: v })
        .catch(function (err) { toast('Could not set strength: ' + err.message, true); })
        .then(function () { dial.pending = false; });
    }, immediate ? 0 : 250);
  }
  function initDial() {
    var input = $('corr');
    input.addEventListener('input', function () { dialInput(Number(input.value)); });
    input.addEventListener('change', function () { dialInput(Number(input.value), true); });
    document.querySelectorAll('.dial-mark').forEach(function (m) {
      m.addEventListener('click', function () { dialInput(Number(m.getAttribute('data-v')), true); });
    });
    var knob = $('knob');
    function angleToValue(ev) {
      var r = knob.getBoundingClientRect();
      var cx = r.left + r.width / 2, cy = r.top + r.height / 2;
      var a = Math.atan2(ev.clientX - cx, cy - ev.clientY) * 180 / Math.PI;
      return clamp((a + 60) / 120, 0, 1);
    }
    knob.addEventListener('pointerdown', function (ev) {
      dial.dragging = true; knob.classList.add('drag');
      knob.setPointerCapture(ev.pointerId);
      dialInput(angleToValue(ev));
    });
    knob.addEventListener('pointermove', function (ev) { if (dial.dragging) dialInput(angleToValue(ev)); });
    function end() { if (!dial.dragging) return; dial.dragging = false; knob.classList.remove('drag'); }
    knob.addEventListener('pointerup', end);
    knob.addEventListener('pointercancel', end);
    setDialVisual(0.5);
  }

  /* ===================== 7. ticker, clock, header ===================== */

  var tick = { i: 0, flash: null, flashUntil: 0 };
  function tickerText() {
    if (tick.flash && Date.now() < tick.flashUntil) return { t: tick.flash, alarm: false };
    var hot = [], noisy = [];
    S.order.forEach(function (id) {
      var c = S.clients.get(id);
      if (c.status === 'under_attack') hot.push(c);
      else if (c.status === 'noisy') noisy.push(c);
    });
    if (hot.length) {
      var h = hot[tick.i % hot.length];
      return { t: 'ATTACK IN PROGRESS at ' + h.id + ' ' + h.name + (h.lastHotType ? ': ' + h.lastHotType : '') + ', ' + fmtCount(h.counters.correct) + ' caught.', alarm: true };
    }
    if (!S.running) return { t: 'Stream paused. Tower on coffee break.', alarm: false };
    if (noisy.length) {
      var n = noisy[0];
      return { t: n.id + ' ' + n.name + ' is jumpy. Try turning the dial up.', alarm: false };
    }
    return { t: TICKER_QUIET[tick.i % TICKER_QUIET.length], alarm: false };
  }
  function tickerNow() {
    var el = $('ticker'), span = $('tickerText'), r = tickerText();
    if (span.textContent === r.t) return;
    el.classList.add('fade');
    setTimeout(function () {
      span.textContent = r.t;
      el.classList.toggle('alarm', r.alarm);
      el.classList.remove('fade');
      // too long for the bar: slide it slowly instead of cutting it off
      var over = span.scrollWidth - el.clientWidth;
      el.classList.toggle('scroll', over > 2);
      if (over > 2) {
        el.style.setProperty('--dx', -(over + 8) + 'px');
        el.style.setProperty('--dur', Math.max(4, over / 25 + 3) + 's');
      }
    }, 180);
  }
  function tickerFlash(text) { tick.flash = text; tick.flashUntil = Date.now() + 5000; tickerNow(); }
  setInterval(function () { tick.i++; tickerNow(); }, 6000);
  // chips decay ("2 alerts: gps" fades back to Normal after 15 s)
  setInterval(function () { S.clients.forEach(function (c) { if (c.alerts.length) c.chipDirty = true; }); scheduleFlush(); }, 1000);

  var lastClock = '';
  function renderClock() {
    var t = fmtTime(S.lastTime);
    if (t === lastClock) return;
    lastClock = t;
    $('clock').innerHTML = esc(t) + '<span>Z</span>';
  }

  function renderInfo() {
    var m = S.model || {};
    var type = (m.model_type || 'cnn').toUpperCase();
    var round = m.best_round != null ? ', round ' + m.best_round : '';
    var narrow = window.innerWidth < 1560;
    var txt = narrow
      ? 'Model: federated ' + type + round + ' | Strength ' + fmtStrength(S.strength)
      : 'Global model: federated ' + type + round + ' | Replay: 2019-02-04 | Correction strength ' + fmtStrength(S.strength);
    var box = $('infobox');
    if (box.textContent !== txt) box.textContent = txt;
    box.title = m.path ? 'Model file: ' + m.path : '';
  }

  function setConn(state) {
    var el = $('conn');
    el.setAttribute('data-state', state);
    $('connText').textContent = {
      connected: 'Link: connected', reconnecting: 'Link: reconnecting',
      connecting: 'Link: connecting', mock: 'Link: mock sim'
    }[state] || state;
  }

  /* ===================== 8. training view ===================== */

  var T = {
    run: null, rounds: 0, clients: [], payload: null,
    globals: [],        // [{round, acc}]
    per: {},            // id -> [acc...]
    rows: {},           // id -> n
    bytesTotal: 0, best: null, ended: false,
    mode: 'idle',       // idle | replay | live
    starting: false, startedAt: 0, startTimer: null,   // live run launched, no round yet
    replayTimer: null, replaySteps: null, replayIdx: 0,
    cache: {}, loaded: false
  };

  function unwrapTraining(ev) {
    if (ev.data && typeof ev.data === 'object') return ev.data;
    if (ev.line && typeof ev.line === 'object') return ev.line;
    if (typeof ev.event === 'string') { var o = Object.assign({}, ev); o.type = ev.event; return o; }
    if (typeof ev.kind === 'string') { var k = Object.assign({}, ev); k.type = ev.kind; return k; }
    return ev;
  }

  function resetTraining() {
    T.run = null; T.rounds = 0; T.clients = []; T.payload = null; T.globals = []; T.per = {}; T.rows = {};
    T.bytesTotal = 0; T.best = null; T.ended = false; T.yHint = null;
  }

  function applyTrain(line) {
    if (!line || !line.type) return;
    if (line.type === 'run_start') {
      resetTraining();
      T.run = line.run || 'run';
      T.rounds = Number(line.rounds) || 0;
      T.payload = line.payload_bytes || null;
      T.clients = (line.clients || []).map(function (c) {
        var known = S.clients.get(c.id);
        return { id: c.id, cell: c.cell || (known && known.cell) || '', name: c.name || (known && known.name) || c.id, train_rows: c.train_rows };
      });
      T.clients.forEach(function (c) { T.per[c.id] = []; T.rows[c.id] = c.train_rows; });
      buildCards();
      if (T.payload) $('keyFoot').textContent = T.clients.length + ' clients | ' + fmtBytes(T.payload) + ' per model update | Federated with Flower';
    } else if (line.type === 'round') {
      var r = Number(line.round);
      T.globals = T.globals.filter(function (g) { return g.round !== r; });
      T.globals.push({ round: r, acc: Number(line.global_acc) });
      T.globals.sort(function (a, b) { return a.round - b.round; });
      var pc = line.per_client || {};
      Object.keys(pc).forEach(function (id) {
        if (!T.per[id]) { T.per[id] = []; T.clients.push({ id: id, name: id, cell: '' }); buildCards(); }
        T.per[id][r - 1] = Number(pc[id].acc);
        if (T.rows[id] == null && pc[id].n != null) T.rows[id] = pc[id].n; // train_rows from run_start wins
      });
      if (line.bytes_total != null) T.bytesTotal = line.bytes_total;
      else if (line.bytes_this_round != null) T.bytesTotal += line.bytes_this_round;
      if (!T.best || Number(line.global_acc) > T.best.acc) T.best = { round: r, acc: Number(line.global_acc) };
      if (r > T.rounds) T.rounds = r;
    } else if (line.type === 'client_fit') {
      flashCard(line.client);
    } else if (line.type === 'run_end') {
      T.ended = true;
      if (line.best_round != null) T.best = { round: Number(line.best_round), acc: Number(line.best_acc) };
    }
  }

  function renderTraining() {
    $('tsRun').textContent = T.run || 'none';
    var cur = T.globals.length ? T.globals[T.globals.length - 1] : null;
    $('tsRound').textContent = (cur ? cur.round : 0) + ' / ' + (T.rounds || 0);
    $('tsAcc').textContent = cur ? cur.acc.toFixed(3) : '-';
    $('tsBest').textContent = T.best ? 'r' + T.best.round + ' at ' + T.best.acc.toFixed(3) : '-';
    $('tsBytes').textContent = fmtBytes(T.bytesTotal);
    $('tsPayload').textContent = T.payload ? fmtBytes(T.payload) : '-';
    $('chartNote').textContent = T.run
      ? (T.mode === 'replay' ? 'Replaying ' + T.run + ', one round per second.' : T.starting ? 'Live run ' + T.run + ' starting.' : T.mode === 'live' ? 'Live run. Ink still wet.' : T.run + ', ' + T.globals.length + ' rounds on file.')
      : (T.starting ? 'Live run starting.' : 'No run loaded yet.');
    renderChart();
    renderCards();
  }

  function buildCards() {
    var host = $('tcards');
    host.textContent = '';
    T.clients.forEach(function (c) {
      var d = document.createElement('div');
      d.className = 'tcard wood';
      d.setAttribute('data-id', c.id);
      d.innerHTML = '<div class="placard"><span class="screw l"></span><span class="screw r"></span>' +
        '<span class="fitting">FITTING</span>' +
        '<span class="p-id"><b>' + esc(c.id) + '</b><span class="p-cell">' + esc(c.cell) + '</span></span>' +
        '<span class="p-name">' + esc(c.name) + '</span>' +
        '<span class="t-acc">LOCAL ACC <b>-</b></span>' +
        '<span class="t-rows">rows -</span>' +
        '<svg viewBox="0 0 160 46" preserveAspectRatio="none" aria-hidden="true"></svg></div>';
      host.appendChild(d);
    });
  }

  function flashCard(id) {
    var el = document.querySelector('.tcard[data-id="' + id + '"]');
    if (!el) return;
    el.classList.add('fit');
    setTimeout(function () { el.classList.remove('fit'); }, 700);
  }

  // deterministic pencil wobble so lines look hand-drawn but do not jitter between renders
  function wobble(i, k) { return Math.sin(i * 12.9898 + k * 78.233) * 0.5 + Math.sin(i * 3.1 + k) * 0.35; }
  function inkPath(pts, amp) {
    if (!pts.length) return '';
    var d = 'M' + pts[0][0].toFixed(1) + ' ' + pts[0][1].toFixed(1);
    for (var i = 1; i < pts.length; i++) {
      var a = pts[i - 1], b = pts[i];
      var steps = 4;
      for (var s = 1; s <= steps; s++) {
        var t = s / steps;
        var x = a[0] + (b[0] - a[0]) * t, y = a[1] + (b[1] - a[1]) * t;
        if (s < steps) { x += wobble(i * 7 + s, 1) * amp; y += wobble(i * 7 + s, 2) * amp; }
        d += ' L' + x.toFixed(1) + ' ' + y.toFixed(1);
      }
    }
    return d;
  }
  function wobblyLine(x1, y1, x2, y2, seed) {
    var pts = [[x1, y1]];
    var n = Math.max(2, Math.round(Math.hypot(x2 - x1, y2 - y1) / 60));
    for (var i = 1; i < n; i++) {
      var t = i / n;
      pts.push([x1 + (x2 - x1) * t + wobble(seed + i, 3) * 0.9, y1 + (y2 - y1) * t + wobble(seed + i, 4) * 0.9]);
    }
    pts.push([x2, y2]);
    return 'M' + pts.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' L');
  }

  function renderChart() {
    var svg = $('chart'), box = $('chartBox');
    var W = Math.max(200, box.clientWidth), H = Math.max(120, box.clientHeight);
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var desc = '<desc id="chartDesc">Global model accuracy after each federated round.</desc>';
    var ml = 62, mr = 24, mt = 14, mb = 44;
    var pw = W - ml - mr, ph = H - mt - mb;
    var g = T.globals;
    var R = Math.max(T.rounds || 5, g.length ? g[g.length - 1].round : 0, 2);
    var lo = 0.5, hi = 1.0;
    if (g.length || T.yHint != null) {
      var mn = Math.min.apply(null, g.map(function (p) { return p.acc; }).concat(T.yHint != null ? [T.yHint] : []));
      lo = Math.min(0.9, Math.floor((mn - 0.03) * 20) / 20);
      lo = Math.max(0, lo);
    }
    var x = function (r) { return ml + (R === 1 ? pw / 2 : (r - 1) / (R - 1) * pw); };
    var y = function (v) { return mt + (1 - (v - lo) / (hi - lo)) * ph; };
    var out = [desc];
    // grid + y ticks
    var stepY = (hi - lo) > 0.3 ? 0.1 : 0.05;
    for (var v = lo; v <= hi + 1e-9; v += stepY) {
      var yy = y(v);
      out.push('<line class="grid-l" x1="' + ml + '" x2="' + (ml + pw) + '" y1="' + yy.toFixed(1) + '" y2="' + yy.toFixed(1) + '"/>');
      out.push('<text class="tick-t" x="' + (ml - 8) + '" y="' + (yy + 4).toFixed(1) + '" text-anchor="end">' + v.toFixed(2) + '</text>');
    }
    // x ticks
    var stepX = R > 30 ? 5 : R > 15 ? 2 : 1;
    for (var r = 1; r <= R; r += stepX) {
      out.push('<text class="tick-t" x="' + x(r).toFixed(1) + '" y="' + (mt + ph + 18) + '" text-anchor="middle">' + r + '</text>');
      out.push('<path class="ax" d="' + wobblyLine(x(r), mt + ph, x(r), mt + ph + 5, r) + '"/>');
    }
    // axes
    out.push('<path class="ax" d="' + wobblyLine(ml, mt, ml, mt + ph, 11) + '"/>');
    out.push('<path class="ax" d="' + wobblyLine(ml, mt + ph, ml + pw, mt + ph, 23) + '"/>');
    out.push('<text class="ax-t" x="' + (ml + pw / 2) + '" y="' + (H - 6) + '" text-anchor="middle">Round</text>');
    out.push('<text class="ax-t" transform="translate(14 ' + (mt + ph / 2) + ') rotate(-90)" text-anchor="middle">Global accuracy</text>');
    if (!g.length && T.starting) {
      var secs = Math.floor((Date.now() - T.startedAt) / 1000);
      out.push('<text class="empty-t starting" x="' + (ml + pw / 2) + '" y="' + (mt + ph / 2 - 12) + '" text-anchor="middle">' +
        'Starting Flower and Ray, about 15 seconds...</text>');
      out.push('<text class="empty-t" x="' + (ml + pw / 2) + '" y="' + (mt + ph / 2 + 16) + '" text-anchor="middle">' +
        secs + ' s so far. Six clients load their data, then round 1 lands here.</text>');
    } else if (!g.length) {
      out.push('<text class="empty-t" x="' + (ml + pw / 2) + '" y="' + (mt + ph / 2) + '" text-anchor="middle">' +
        (T.mode === 'live' ? 'Waiting for round 1. Flower is warming up.' : 'No rounds yet. Press Replay run or Train live.') + '</text>');
    } else {
      var pts = g.map(function (p) { return [x(p.round), y(p.acc)]; });
      out.push('<path class="ink2" d="' + inkPath(pts.map(function (p) { return [p[0] + 0.9, p[1] - 0.8]; }), 1.2) + '"/>');
      out.push('<path class="ink" d="' + inkPath(pts, 0.9) + '"/>');
      g.forEach(function (p, i) {
        out.push('<circle class="pt" cx="' + pts[i][0].toFixed(1) + '" cy="' + pts[i][1].toFixed(1) + '" r="4.5"/>');
        out.push('<circle class="pt-hit" cx="' + pts[i][0].toFixed(1) + '" cy="' + pts[i][1].toFixed(1) + '" r="13"><title>Round ' + p.round + ': global accuracy ' + p.acc.toFixed(3) + '</title></circle>');
      });
      if (T.best) {
        var bp = g.filter(function (p) { return p.round === T.best.round; })[0];
        if (bp) {
          var bx = x(bp.round), by = y(bp.acc);
          out.push('<ellipse class="best-ring" cx="' + bx.toFixed(1) + '" cy="' + by.toFixed(1) + '" rx="12" ry="10" transform="rotate(-12 ' + bx.toFixed(1) + ' ' + by.toFixed(1) + ')"/>');
          var anchor = bx > ml + pw - 80 ? 'end' : (bx < ml + 40 ? 'start' : 'middle');
          out.push('<text class="best-t" x="' + bx.toFixed(1) + '" y="' + (by + 30).toFixed(1) + '" text-anchor="' + anchor + '">BEST r' + bp.round + '</text>');
        }
      }
      var last = g[g.length - 1], lp = pts[pts.length - 1];
      var la = lp[0] > ml + pw - 60 ? 'end' : 'start';
      out.push('<text class="val-t" x="' + (lp[0] + (la === 'end' ? -10 : 10)).toFixed(1) + '" y="' + (lp[1] - 10).toFixed(1) + '" text-anchor="' + la + '">' + last.acc.toFixed(3) + '</text>');
    }
    svg.innerHTML = out.join('');
  }

  function renderCards() {
    var all = [];
    Object.keys(T.per).forEach(function (id) { T.per[id].forEach(function (v) { if (v != null && !isNaN(v)) all.push(v); }); });
    var lo = all.length ? Math.min.apply(null, all) : 0, hi = all.length ? Math.max.apply(null, all) : 1;
    if (hi - lo < 0.05) { lo -= 0.025; hi += 0.025; }
    var R = Math.max(T.rounds || 1, 2);
    document.querySelectorAll('.tcard').forEach(function (card) {
      var id = card.getAttribute('data-id');
      var arr = T.per[id] || [];
      var lastV = null;
      for (var i = arr.length - 1; i >= 0; i--) { if (arr[i] != null && !isNaN(arr[i])) { lastV = arr[i]; break; } }
      card.querySelector('.t-acc b').textContent = lastV == null ? '-' : lastV.toFixed(3);
      card.querySelector('.t-rows').textContent = 'rows ' + (T.rows[id] != null ? Number(T.rows[id]).toLocaleString('en-US') : '-');
      var pts = [];
      arr.forEach(function (v, i2) {
        if (v == null || isNaN(v)) return;
        pts.push([4 + i2 / (R - 1) * 152, 42 - (v - lo) / (hi - lo) * 38]);
      });
      var svg = card.querySelector('svg');
      if (!pts.length) { svg.innerHTML = ''; return; }
      var lp = pts[pts.length - 1];
      svg.innerHTML = '<path class="spark" vector-effect="non-scaling-stroke" d="' + inkPath(pts, 0.6) + '"/>' +
        '<path d="M' + lp[0].toFixed(1) + ' ' + lp[1].toFixed(1) + ' l0.01 0" stroke="#b3261e" stroke-width="7" stroke-linecap="round" vector-effect="non-scaling-stroke"/>';
    });
  }

  function setTrainStatus(t) { $('trainStatus').textContent = t; }

  function loadRuns() {
    return transport.request('GET', '/api/training/runs').then(function (res) {
      var runs = Array.isArray(res) ? res : (res && res.runs) || [];
      var sel = $('runSel'), prev = sel.value;
      sel.textContent = '';
      if (runs.indexOf('latest') < 0) runs = ['latest'].concat(runs);
      runs.forEach(function (r) { var o = document.createElement('option'); o.value = r; o.textContent = r; sel.appendChild(o); });
      if (runs.indexOf(prev) >= 0) sel.value = prev;
    }).catch(function (err) { console.warn('runs list failed', err); });
  }

  function fetchReplay(run) {
    if (T.cache[run]) return Promise.resolve(T.cache[run]);
    return transport.request('GET', '/api/training/replay?run=' + encodeURIComponent(run)).then(function (lines) {
      lines = Array.isArray(lines) ? lines : (lines && lines.lines) || [];
      T.cache[run] = lines;
      return lines;
    });
  }

  function showStatic(run) {
    return fetchReplay(run).then(function (lines) {
      if (T.mode !== 'idle') return;
      resetTraining();
      lines.forEach(applyTrain);
      renderTraining();
      setTrainStatus(lines.length ? 'Run on file. Press Replay run to watch it again.' : 'That run file is empty.');
    }).catch(function (err) {
      setTrainStatus('No run on file yet (' + err.message + '). Train live to make one.');
      renderTraining();
    });
  }

  function stopReplay() {
    clearTimeout(T.replayTimer);
    T.replayTimer = null;
    if (T.mode === 'replay') T.mode = 'idle';
    $('replayBtn').textContent = 'Replay run';
  }

  function startReplay() {
    if (T.mode === 'replay') { // second press finishes it instantly
      var steps = T.replaySteps || [];
      for (var i = T.replayIdx; i < steps.length; i++) steps[i].forEach(applyTrain);
      stopReplay(); renderTraining();
      setTrainStatus('Replay skipped to the end.');
      return;
    }
    if (T.mode === 'live') { toast('A live run is in progress. Replay after it finishes.', false); return; }
    var run = $('runSel').value || 'latest';
    fetchReplay(run).then(function (lines) {
      if (!lines.length) { toast('That run has no rounds to replay.', true); return; }
      // group: run_start | each round with the client_fit lines before it | run_end
      var steps = [], pending = [];
      lines.forEach(function (l) {
        if (l.type === 'run_start') steps.push([l]);
        else if (l.type === 'client_fit') pending.push(l);
        else if (l.type === 'round') { steps.push(pending.concat([l])); pending = []; }
        else if (l.type === 'run_end') steps.push(pending.concat([l]));
      });
      var hint = Math.min.apply(null, lines.filter(function (l) { return l.type === 'round'; }).map(function (l) { return Number(l.global_acc); }).concat([1]));
      T.mode = 'replay';
      T.replaySteps = steps; T.replayIdx = 0;
      $('replayBtn').textContent = 'Skip to end';
      setTrainStatus('Replaying ' + run + '. Pretend it is live.');
      var stepFn = function () {
        if (T.mode !== 'replay') return;
        var st = T.replaySteps[T.replayIdx++];
        st.forEach(applyTrain);
        T.yHint = hint;
        renderTraining();
        if (T.replayIdx >= T.replaySteps.length) {
          stopReplay();
          setTrainStatus('Replay done. Best round ' + (T.best ? T.best.round : '-') + '.');
          renderTraining();
          return;
        }
        T.replayTimer = setTimeout(stepFn, T.replayIdx === 1 ? 600 : 1000);
      };
      stepFn();
    }).catch(function (err) { toast('Replay failed: ' + err.message, true); });
  }

  function startLive() {
    var rounds = clamp(parseInt($('rounds').value, 10) || 10, 1, 100);
    $('rounds').value = rounds;
    var btn = $('trainBtn');
    btn.disabled = true;
    transport.request('POST', '/api/training/start', { rounds: rounds }).then(function (res) {
      stopReplay();
      T.mode = 'live';
      resetTraining();
      T.rounds = rounds;
      if (res && res.run) T.run = res.run;
      setStarting(true);
      renderTraining();
    }).catch(function (err) {
      toast(err.status === 409 ? 'A training run is already in progress.' : 'Could not start training: ' + err.message, true);
    }).then(function () { setTimeout(function () { btn.disabled = false; }, 1500); });
  }

  // between "Train live" and the first round event: Ray and Flower take about 15 s to start
  function setStarting(on) {
    clearInterval(T.startTimer);
    T.startTimer = null;
    T.starting = !!on;
    if (!on) return;
    T.startedAt = Date.now();
    var tickStart = function () {
      var secs = Math.floor((Date.now() - T.startedAt) / 1000);
      setTrainStatus('Starting Flower and Ray, about 15 seconds (' + secs + ' s so far). Round 1 follows.');
      if (activeView === 'train') renderTraining();
    };
    tickStart();
    T.startTimer = setInterval(tickStart, 1000);
  }

  function onTrainingEvent(line) {
    if (!line || !line.type) return;
    if (T.mode === 'replay') stopReplay();
    var wasStarting = T.starting;
    T.mode = 'live';
    applyTrain(line); // run_start resets the run, so keep the starting flag across it
    if (line.type === 'run_start') {
      if (!wasStarting) setStarting(true); // a run launched from another tab
      T.starting = true;
    }
    if (line.type === 'round') {
      setStarting(false);
      setTrainStatus('Round ' + line.round + ' of ' + (T.rounds || '?') + ' aggregated. FedAvg did its thing.');
    }
    if (line.type === 'run_end') {
      setStarting(false);
      T.mode = 'idle';
      setTrainStatus('Run finished. Best round ' + line.best_round + ' at ' + Number(line.best_acc).toFixed(3) + '.');
      loadRuns();
    }
    if (line.type === 'run_error') {
      setStarting(false);
      T.mode = 'idle';
      setTrainStatus('Run failed: ' + (line.error || 'unknown error') + '. See the server terminal.');
      loadRuns();
    }
    if (activeView === 'train') renderTraining();
  }

  function initTraining() {
    $('replayBtn').addEventListener('click', startReplay);
    $('trainBtn').addEventListener('click', startLive);
    $('roundsDec').addEventListener('click', function () { $('rounds').value = clamp((parseInt($('rounds').value, 10) || 10) - 1, 1, 100); });
    $('roundsInc').addEventListener('click', function () { $('rounds').value = clamp((parseInt($('rounds').value, 10) || 10) + 1, 1, 100); });
    $('runSel').addEventListener('change', function () { if (T.mode === 'idle') showStatic($('runSel').value); });
  }

  /* ===================== 9. tabs, toasts, boot ===================== */

  var activeView = 'live';
  function showView(which) {
    activeView = which;
    var live = which === 'live';
    $('tab-live').setAttribute('aria-selected', live ? 'true' : 'false');
    $('tab-train').setAttribute('aria-selected', live ? 'false' : 'true');
    $('tab-live').tabIndex = live ? 0 : -1;
    $('tab-train').tabIndex = live ? -1 : 0;
    $('view-live').hidden = !live;
    $('view-train').hidden = live;
    $('viewTitle').textContent = live ? 'Live Detection' : 'Training Run';
    $('viewKicker').textContent = live ? 'Strip board | all towers' : 'Flight log | FedAvg rounds';
    if (live) { measureNeeded = true; scheduleFlush(); }
    else {
      if (!T.loaded) { T.loaded = true; loadRuns(); if (T.mode === 'idle') showStatic('latest'); }
      renderTraining();
    }
  }
  function initTabs() {
    $('tab-live').addEventListener('click', function () { showView('live'); });
    $('tab-train').addEventListener('click', function () { showView('train'); });
    document.querySelector('.tabs').addEventListener('keydown', function (e) {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      var next = activeView === 'live' ? 'train' : 'live';
      showView(next);
      $(next === 'live' ? 'tab-live' : 'tab-train').focus();
      e.preventDefault();
    });
  }

  function toast(text, isErr) {
    if (!text) return;
    var host = $('toasts');
    var d = document.createElement('div');
    d.className = 'toast' + (isErr ? ' err' : '');
    d.textContent = text;
    host.appendChild(d);
    while (host.children.length > 4) host.removeChild(host.firstChild);
    setTimeout(function () { d.remove(); }, isErr ? 7000 : 4500);
  }

  function boot() {
    initTabs();
    initControls();
    initTraining();
    var resizeTimer = null;
    window.addEventListener('resize', function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        measureNeeded = true; scheduleFlush(); renderInfo();
        if (activeView === 'train') renderTraining();
      }, 120);
    });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { measureNeeded = true; scheduleFlush(); });
    renderLog();
    transport.openSocket(enqueue, setConn);
    if (/(?:^|[?&])view=train(?:&|$)/.test(location.search)) showView('train');
  }

  // expose a tiny handle for debugging and automated tests
  window.FLApp = { state: S, training: T, showView: function (v) { showView(v); } };

  if (MOCK) {
    var s = document.createElement('script');
    s.src = 'mock.js';
    s.onload = boot;
    s.onerror = function () { document.body.insertAdjacentHTML('afterbegin', '<p style="background:#f8d9cf;padding:8px;margin:0">mock.js failed to load.</p>'); };
    document.head.appendChild(s);
  } else {
    boot();
  }
})();
