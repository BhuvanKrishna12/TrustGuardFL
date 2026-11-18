/*
 * mock.js: in-browser stand-in for demo/server.py, loaded only with index.html?mock=1.
 * Emits the same WebSocket events and answers the same REST calls as CONTRACT.md, so the page
 * can be developed and demoed with no backend. Behaviour, roughly:
 *   - six towers stream benign traffic at `rate` msgs/s each
 *   - C6 (Gulf of Alaska) is the weak client: it raises occasional false alarms, more of them at
 *     low correction strength, enough to go `noisy` near strength 0
 *   - an inject queues rows of the chosen attack, drained over a few seconds on top of normal
 *     traffic; most are caught (status flips to under_attack), some are missed, a few get the
 *     wrong attack type
 *   - training: a seeded fake 15-round run for replay, and /api/training/start emits
 *     `training` events round by round
 */
(function () {
  'use strict';

  var CLIENTS = [
    { id: 'C1', cell: '4_-10', name: 'US Midwest', lat: 45, lon: -95, hex: 'a', rows: 412880 },
    { id: 'C2', cell: '5_1', name: 'Central Europe', lat: 55, lon: 15, hex: '3c', rows: 389412 },
    { id: 'C3', cell: '3_-9', name: 'US Southeast', lat: 35, lon: -85, hex: 'a', rows: 356204 },
    { id: 'C4', cell: '0_10', name: 'Malay Peninsula', lat: 5, lon: 105, hex: '75', rows: 118733 },
    { id: 'C5', cell: '-4_14', name: 'SE Australia', lat: -35, lon: 145, hex: '7c', rows: 97508 },
    { id: 'C6', cell: '5_-14', name: 'Gulf of Alaska', lat: 55, lon: -135, hex: 'a', rows: 41260 }
  ];
  var ATTACKS = ['flooding', 'ghost_aircraft', 'gps_spoof', 'replay', 'velocity_spoof'];

  var listeners = [];
  var st = {
    strength: 0.5, running: true, rate: 4, seq: 0, burst: 0,
    simMs: 0, startClock: 14 * 3600 + 30 * 60, emit: false,
    incidents: [], training: null
  };
  var per = {};
  CLIENTS.forEach(function (c) {
    per[c.id] = {
      acc: Math.random(), queue: [], injAcc: 0, recent: [],
      counters: { seen: 0, flagged: 0, correct: 0, false_alarm: 0, missed: 0 },
      catches: [], fas: [], status: 'normal', cntDirty: false, lastCnt: 0
    };
  });

  function send(ev) {
    if (!st.emit) return;
    var s = JSON.stringify(ev); // round-trip like a real socket would
    listeners.forEach(function (fn) { try { fn(JSON.parse(s)); } catch (e) { console.error(e); } });
  }
  function hex(prefix) {
    var s = prefix;
    while (s.length < 6) s += '0123456789abcdef'[Math.floor(Math.random() * 16)];
    return s;
  }
  function clockStr() {
    var t = st.startClock + Math.floor(st.simMs / 1000);
    var h = Math.floor(t / 3600) % 24, m = Math.floor(t / 60) % 60, s = t % 60;
    return [h, m, s].map(function (x) { return (x < 10 ? '0' : '') + x; }).join(':');
  }
  function r2(x) { return Math.round(x * 100) / 100; }

  function faProb(id) {
    var s = st.strength;
    if (id === 'C6') return Math.max(0.002, 0.04 * (1 - 1.4 * s));
    return Math.max(0.0004, 0.0018 * (1 - 0.8 * s));
  }

  function emitRow(id, truth, injected, burstId) {
    var p = per[id], c = CLIENTS.filter(function (x) { return x.id === id; })[0];
    var pred, conf, verdict, wrong = false;
    if (truth === 'benign') {
      var forced = st.forceFA && st.forceFA[id] > 0;
      if (forced) st.forceFA[id]--;
      if (forced || Math.random() < faProb(id)) {
        pred = ['replay', 'flooding', 'gps_spoof', 'velocity_spoof'][Math.floor(Math.random() * 4)];
        conf = 0.5 + Math.random() * 0.2; verdict = 'false_alarm';
      } else { pred = 'benign'; conf = 0.9 + Math.random() * 0.1; verdict = 'ok'; }
    } else {
      var pDetect = 0.97 - 0.22 * st.strength;
      if (Math.random() < pDetect) {
        verdict = 'correct_catch';
        if (Math.random() < 0.04) {
          var others = ATTACKS.filter(function (a) { return a !== truth; });
          pred = others[Math.floor(Math.random() * others.length)]; wrong = true; conf = 0.55 + Math.random() * 0.2;
        } else { pred = truth; conf = 0.85 + Math.random() * 0.14; }
      } else { pred = 'benign'; conf = 0.5 + Math.random() * 0.3; verdict = 'missed'; }
    }
    var m = {
      type: 'msg', client: id, seq: ++st.seq, time: clockStr(), icao24: hex(c.hex),
      pred: pred, conf: r2(conf), truth: truth, injected: !!injected, verdict: verdict
    };
    if (burstId) m.burst_id = burstId;
    if (wrong) m.wrong_type = true;
    p.counters.seen++;
    if (pred !== 'benign') p.counters.flagged++;
    if (verdict === 'correct_catch') { p.counters.correct++; p.catches.push(st.simMs); }
    if (verdict === 'false_alarm') { p.counters.false_alarm++; p.fas.push(st.simMs); }
    if (verdict === 'missed') p.counters.missed++;
    p.cntDirty = true;
    p.recent.push(m); if (p.recent.length > 12) p.recent.shift();
    send(m);
    if (pred !== 'benign') {
      var inc = Object.assign({}, m, { type: 'incident', name: c.name });
      st.incidents.push(inc); if (st.incidents.length > 50) st.incidents.shift();
      send(inc);
    }
  }

  function updateStatus(id) {
    var p = per[id], now = st.simMs;
    p.catches = p.catches.filter(function (t) { return now - t < 10000; });
    p.fas = p.fas.filter(function (t) { return now - t < 30000; });
    var s = p.catches.length >= 3 ? 'under_attack' : (p.fas.length >= 2 ? 'noisy' : 'normal');
    if (s !== p.status) { p.status = s; send({ type: 'status', client: id, status: s }); }
  }

  function step(dtMs) {
    st.simMs += dtMs;
    CLIENTS.forEach(function (c) {
      var p = per[c.id];
      if (st.running) {
        p.acc += st.rate * dtMs / 1000;
        while (p.acc >= 1) { p.acc -= 1; emitRow(c.id, 'benign'); }
        if (p.queue.length) {
          var drain = Math.max(4, p.queue.burstRate || 5);
          p.injAcc += drain * dtMs / 1000;
          while (p.injAcc >= 1 && p.queue.length) { p.injAcc -= 1; var q = p.queue.shift(); emitRow(c.id, q.attack, true, q.burst); }
        } else p.injAcc = 0;
      }
      updateStatus(c.id);
      if (p.cntDirty && st.simMs - p.lastCnt >= 500) {
        p.cntDirty = false; p.lastCnt = st.simMs;
        send({ type: 'counters', client: c.id, counters: Object.assign({}, p.counters) });
      }
    });
  }

  function snapshot() {
    return {
      type: 'snapshot',
      clients: CLIENTS.map(function (c) {
        var p = per[c.id];
        return { id: c.id, cell: c.cell, name: c.name, lat: c.lat, lon: c.lon, status: p.status,
          counters: Object.assign({}, p.counters), recent: p.recent.slice() };
      }),
      strength: st.strength, running: st.running, rate: st.rate,
      model: { path: 'models/fixed4_cnn.pt', model_type: 'cnn', best_round: 46 },
      incidents: st.incidents.slice(),
      attack_types: ATTACKS.slice()
    };
  }
  function config() { return { type: 'config', strength: st.strength, running: st.running, rate: st.rate }; }

  // warm up 25 s of history silently so the board is not empty on load
  for (var w = 0; w < 500; w++) step(50);
  st.emit = true;
  var last = performance.now();
  setInterval(function () {
    var now = performance.now();
    var dt = Math.min(500, now - last); last = now;
    step(dt);
  }, 50);

  /* ---------- fake training ---------- */
  function rng(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  var OFFS = { C1: 0.012, C2: 0.02, C3: 0.004, C4: -0.018, C5: -0.03, C6: -0.11 };
  function fakeRun(name, rounds, seed) {
    var rand = rng(seed), payload = 44312, t0 = 1762784520, lines = [], total = 0;
    lines.push({ type: 'run_start', run: name, clients: CLIENTS.map(function (c) { return { id: c.id, cell: c.cell, name: c.name, train_rows: c.rows }; }),
      rounds: rounds, model: 'cnn', cnn_channels: [32, 64], payload_bytes: payload, t: t0 });
    var best = { r: 0, a: 0 };
    for (var r = 1; r <= rounds; r++) {
      var g = 0.56 + 0.35 * (1 - Math.exp(-r / 3.4)) + (rand() - 0.5) * 0.018;
      var pc = {};
      CLIENTS.forEach(function (c) {
        lines.push({ type: 'client_fit', round: r, client: c.id, train_acc: r2(g + OFFS[c.id] + 0.02), n: c.rows, t: t0 + r * 40 });
        pc[c.id] = { acc: Math.round((g + OFFS[c.id] + (rand() - 0.5) * 0.03) * 1000) / 1000, n: Math.round(c.rows * 0.2) };
      });
      var bytes = payload * CLIENTS.length * 2;
      total += bytes;
      g = Math.round(g * 10000) / 10000;
      if (g > best.a) best = { r: r, a: g };
      lines.push({ type: 'round', round: r, global_acc: g, per_client: pc, bytes_this_round: bytes, bytes_total: total, t: t0 + r * 42 });
    }
    lines.push({ type: 'run_end', best_round: best.r, best_acc: best.a, model_path: 'demo/models/' + name + '.pt', t: t0 + rounds * 42 + 5 });
    return lines;
  }
  var RUNS = { run_20261110_1402: fakeRun('run_20261110_1402', 15, 7), run_20261109_2215: fakeRun('run_20261109_2215', 12, 3) };
  RUNS.latest = RUNS.run_20261110_1402;

  function startTraining(rounds) {
    var name = 'run_mock_' + Date.now().toString(36);
    var lines = fakeRun(name, rounds, Date.now() & 0xffff);
    st.training = { name: name };
    var i = 0;
    send({ type: 'toast', text: 'Mock mode: training is simulated in the browser.' });
    (function next() {
      if (i >= lines.length) { RUNS[name] = lines; RUNS.latest = lines; st.training = null; return; }
      var l = lines[i++];
      send({ type: 'training', data: l });
      setTimeout(next, l.type === 'client_fit' ? 150 : (l.type === 'run_start' ? 1500 : 500));
    })();
  }

  /* ---------- REST ---------- */
  function ok(data) { return new Promise(function (res) { setTimeout(function () { res(data); }, 40); }); }
  function fail(status, detail) {
    return new Promise(function (_, rej) { setTimeout(function () { var e = new Error(detail); e.status = status; rej(e); }, 40); });
  }

  function request(method, path, body) {
    var url = new URL(path, location.href), p = url.pathname;
    body = body || {};
    if (method === 'GET' && p === '/api/state') return ok(snapshot());
    if (method === 'POST' && p === '/api/inject') {
      if (!per[body.client]) return fail(400, 'unknown client ' + body.client);
      if (ATTACKS.indexOf(body.attack) < 0) return fail(400, 'unknown attack ' + body.attack);
      var n = Math.max(1, Math.min(200, body.count | 0)), id = 'b' + (++st.burst);
      var q = per[body.client].queue;
      for (var i = 0; i < n; i++) q.push({ attack: body.attack, burst: id });
      q.burstRate = Math.max(5, q.length / 5); // spread over about five seconds
      return ok({ ok: true, queued: n, burst_id: id });
    }
    if (method === 'POST' && p === '/api/strength') {
      st.strength = Math.max(0, Math.min(1, Number(body.strength)));
      send(config()); return ok({ ok: true });
    }
    if (method === 'POST' && p === '/api/stream') {
      if (body.running != null) st.running = !!body.running;
      if (body.rate != null) st.rate = Number(body.rate);
      send(config()); return ok({ ok: true });
    }
    if (method === 'POST' && p === '/api/reset') {
      st.incidents = [];
      CLIENTS.forEach(function (c) {
        var pp = per[c.id];
        pp.counters = { seen: 0, flagged: 0, correct: 0, false_alarm: 0, missed: 0 };
        pp.catches = []; pp.fas = [];
        send({ type: 'counters', client: c.id, counters: Object.assign({}, pp.counters) });
      });
      return ok({ ok: true });
    }
    if (method === 'GET' && p === '/api/training/runs') return ok({ runs: Object.keys(RUNS) });
    if (method === 'GET' && p === '/api/training/replay') {
      var run = url.searchParams.get('run') || 'latest';
      if (!RUNS[run]) return fail(404, 'no such run');
      return ok(JSON.parse(JSON.stringify(RUNS[run])));
    }
    if (method === 'POST' && p === '/api/training/start') {
      if (st.training) return fail(409, 'A training run is already in progress.');
      startTraining(Math.max(1, Math.min(100, body.rounds | 0 || 10)));
      return ok({ ok: true, run: 'mock' });
    }
    return fail(404, 'mock has no route ' + method + ' ' + p);
  }

  function connect(onEvent) {
    listeners.push(onEvent);
    setTimeout(function () { onEvent(JSON.parse(JSON.stringify(snapshot()))); }, 30);
    return { close: function () { listeners = listeners.filter(function (f) { return f !== onEvent; }); } };
  }

  // test helper: force the weak client noisy now (used by screenshots)
  function forceNoisy(id) {
    for (var i = 0; i < 3; i++) emitRowForced(id); // next three benign rows become false alarms
  }
  function emitRowForced(id) {
    st.forceFA = st.forceFA || {};
    st.forceFA[id] = (st.forceFA[id] || 0) + 1;
  }

  window.FLMock = { request: request, connect: connect, forceNoisy: forceNoisy, state: st };
})();
