(function () {
  'use strict';
  var cv = document.getElementById('game'), ctx = cv.getContext('2d');
  var W = cv.width, H = cv.height;
  var $ = function (i) { return document.getElementById(i); };
  var overlay = $('overlay'), status = $('game-status');
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var keys = { left: false, right: false, up: false };
  var best = 0, total = 0, streak = 0, state = 'idle', s, terrain, pad, stars = [], last = 0;
  try { best = +localStorage.getItem('moonhop-best') || 0; } catch (e) {}
  $('h-best').textContent = best;
  for (var i = 0; i < 70; i++) stars.push([(i * 137.5) % W, (i * 71.3) % (H * 0.7), (i % 3) + 0.5]);

  function makeLevel() {
    var padW = Math.max(90, 150 - streak * 12);
    var px = 100 + Math.random() * (W - 200 - padW);
    var py = H - 70 - Math.random() * 60;
    terrain = []; var y = py, n = 40;
    for (var k = 0; k <= n; k++) {
      var x = k * W / n;
      if (x >= px && x <= px + padW) y = py;
      else y = Math.min(H - 20, Math.max(H - 180, y + (Math.random() - 0.5) * 36));
      if (Math.abs(x - px) < W / n || Math.abs(x - px - padW) < W / n) y = py;
      terrain.push([x, y]);
    }
    pad = { x: px, w: padW, y: py };
    // flatten terrain under pad
    terrain = terrain.filter(function (t) { return t[0] < px || t[0] > px + padW; });
    terrain.push([px, py], [px + padW, py]);
    terrain.sort(function (a, b) { return a[0] - b[0]; });
    s = { x: W / 2 + (Math.random() - 0.5) * 300, y: 60, vx: (Math.random() - 0.5) * 50, vy: 0, a: (Math.random() - 0.5) * 0.4, fuel: 100, thrust: false };
  }
  function groundY(x) {
    for (var k = 1; k < terrain.length; k++) {
      if (x <= terrain[k][0]) { var a = terrain[k - 1], b = terrain[k]; return a[1] + (b[1] - a[1]) * (x - a[0]) / ((b[0] - a[0]) || 1); }
    }
    return terrain[terrain.length - 1][1];
  }
  function show(title, text, btn) { $('ov-title').textContent = title; $('ov-text').textContent = text; $('ov-btn').textContent = btn; overlay.hidden = false; $('ov-btn').focus({ preventScroll: true }); status.textContent = title + '. ' + text; }
  function start(newRun) {
    if (newRun) { total = 0; streak = 0; }
    makeLevel(); state = 'play'; overlay.hidden = true; last = performance.now(); $('h-score').textContent = total;
    cv.focus({ preventScroll: true });
  }
  function end(landed) {
    var sp = Math.hypot(s.vx, s.vy);
    if (landed) {
      var gain = 100 + Math.round(s.fuel) + Math.round(Math.max(0, 40 - sp) * 2) + streak * 50;
      total += gain; streak++; state = 'won';
      if (total > best) { best = total; try { localStorage.setItem('moonhop-best', best); } catch (e) {} }
      $('h-best').textContent = best; $('h-score').textContent = total;
      show('Nice landing! +' + gain, 'Run score ' + total + '. Next pad is smaller.', 'Next landing');
    } else {
      state = 'lost';
      show('Crashed', 'Final score ' + total + '. Too fast, too tilted, or off the pad.', 'Try again');
    }
  }
  function step(dt) {
    var g = 38, thr = 95, rot = 2.2;
    if (keys.left) s.a -= rot * dt;
    if (keys.right) s.a += rot * dt;
    s.thrust = keys.up && s.fuel > 0;
    if (s.thrust) { s.vx += Math.sin(s.a) * thr * dt; s.vy -= Math.cos(s.a) * thr * dt; s.fuel = Math.max(0, s.fuel - 14 * dt); }
    s.vy += g * dt; s.x += s.vx * dt; s.y += s.vy * dt;
    if (s.x < 0 || s.x > W || s.y < -200) { end(false); return; }
    var gy = groundY(s.x);
    if (s.y + 14 >= gy) {
      var onPad = s.x > pad.x && s.x < pad.x + pad.w;
      var ok = onPad && Math.hypot(s.vx, s.vy) < 40 && Math.abs(s.a) < 12 * Math.PI / 180;
      s.y = gy - 14; end(ok);
    }
  }
  function draw() {
    ctx.fillStyle = '#05070f'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#cfd8ff';
    stars.forEach(function (t) { ctx.fillRect(t[0], t[1], t[2], t[2]); });
    ctx.beginPath(); ctx.moveTo(0, H);
    terrain.forEach(function (t) { ctx.lineTo(t[0], t[1]); });
    ctx.lineTo(W, H); ctx.closePath(); ctx.fillStyle = '#2a3150'; ctx.fill();
    ctx.fillStyle = '#5eea9a'; ctx.fillRect(pad.x, pad.y - 3, pad.w, 5);
    ctx.save(); ctx.translate(s.x, s.y); ctx.rotate(s.a);
    if (s.thrust && state === 'play') { ctx.fillStyle = '#ffc857'; ctx.beginPath(); ctx.moveTo(-6, 12); ctx.lineTo(0, 12 + 14 + Math.random() * 10); ctx.lineTo(6, 12); ctx.fill(); }
    ctx.fillStyle = state === 'lost' ? '#ff6b7a' : '#e8ecf8';
    ctx.beginPath(); ctx.moveTo(0, -16); ctx.lineTo(11, 8); ctx.lineTo(-11, 8); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = '#7cc4ff'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(-8, 8); ctx.lineTo(-14, 14); ctx.moveTo(8, 8); ctx.lineTo(14, 14); ctx.stroke();
    ctx.restore();
    var sp = Math.hypot(s.vx, s.vy);
    $('h-fuel').textContent = Math.round(s.fuel); $('h-speed').textContent = Math.round(sp);
  }
  function loop(now) {
    if (state === 'play') { var dt = Math.min(0.033, (now - last) / 1000); last = now; step(dt); }
    if (terrain) draw();
    requestAnimationFrame(loop);
  }
  var map = { ArrowLeft: 'left', a: 'left', A: 'left', ArrowRight: 'right', d: 'right', D: 'right', ArrowUp: 'up', w: 'up', W: 'up' };
  window.addEventListener('keydown', function (e) {
    if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
    if (e.key === 'r' || e.key === 'R') { if (state !== 'idle') start(true); return; }
    var k = map[e.key]; if (!k) return;
    if (document.activeElement === cv || state === 'play') e.preventDefault();
    keys[k] = true;
  });
  window.addEventListener('keyup', function (e) { var k = map[e.key]; if (k) keys[k] = false; });
  window.addEventListener('blur', function () { keys.left = keys.right = keys.up = false; });
  Array.prototype.forEach.call(document.querySelectorAll('.pad button'), function (b) {
    var k = b.getAttribute('data-k');
    function on(e) { e.preventDefault(); keys[k] = true; b.classList.add('on'); }
    function off() { keys[k] = false; b.classList.remove('on'); }
    b.addEventListener('pointerdown', on);
    ['pointerup', 'pointercancel', 'pointerleave'].forEach(function (ev) { b.addEventListener(ev, off); });
    b.addEventListener('keydown', function (e) { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); on(e); } });
    b.addEventListener('keyup', function (e) { if (e.key === ' ' || e.key === 'Enter') off(); });
  });
  $('ov-btn').addEventListener('click', function () { start(state === 'lost' || state === 'idle'); });
  makeLevel(); draw();
  requestAnimationFrame(loop);
})();
