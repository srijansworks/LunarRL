(function () {
  'use strict';
  var R = window.LUNARRL_RESULTS;
  var video = document.getElementById('replay');
  var btns = { ppo: document.getElementById('btn-ppo'), random: document.getElementById('btn-random') };
  var cap = document.getElementById('replay-caption');
  var captions = {
    ppo: 'Trained PPO agent: slows down, steadies itself, and settles between the flags.',
    random: 'Random agent: fires engines with no plan and crashes fast.'
  };
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function setMode(m) {
    Object.keys(btns).forEach(function (k) { btns[k].setAttribute('aria-pressed', String(k === m)); });
    var base = 'assets/media/lunar-rl-' + m;
    video.poster = base + '.jpg';
    video.src = base + '.mp4';
    video.load();
    if (!reduce) { var p = video.play(); if (p && p.catch) p.catch(function () {}); }
    cap.textContent = captions[m];
  }
  btns.ppo.addEventListener('click', function () { setMode('ppo'); });
  btns.random.addEventListener('click', function () { setMode('random'); });
  cap.textContent = captions.ppo;
  if (!reduce) { var pp = video.play(); if (pp && pp.catch) pp.catch(function () {}); }

  var grid = document.getElementById('stat-grid');
  var bars = document.getElementById('bars');
  if (!R || !R.eval) { grid.textContent = 'Results data did not load.'; return; }
  var p = R.eval.ppo, r = R.eval.random;
  function stat(v, l) { var d = document.createElement('div'); d.className = 'stat'; var b = document.createElement('b'); b.textContent = v; var s = document.createElement('span'); s.textContent = l; d.appendChild(b); d.appendChild(s); grid.appendChild(d); }
  stat(p.landed_on_pad + '/' + p.episodes, 'Trained: landed on pad');
  stat(String(p.crashed), 'Trained: crashes');
  stat(p.mean_return.toFixed(1), 'Trained: mean return');
  stat(r.crashed + '/' + r.episodes, 'Random: crashes');
  stat(r.mean_return.toFixed(1), 'Random: mean return');
  stat(String(r.solved_episodes), 'Random: solved (200+)');
  var colors = { pad: '#5eea9a', other: '#ffc857', crash: '#ff6b7a', oob: '#9a6bff' };
  function row(name, d) {
    var n = d.episodes;
    var parts = [['pad', d.landed_on_pad, 'on pad'], ['other', d.landed - d.landed_on_pad + d.timeout, 'landed off pad or timeout'], ['crash', d.crashed, 'crashed'], ['oob', d.out_of_bounds, 'out of bounds']];
    var wrap = document.createElement('div'); wrap.className = 'bar-row';
    var lab = document.createElement('span'); lab.textContent = name; wrap.appendChild(lab);
    var bar = document.createElement('div'); bar.className = 'bar';
    var desc = name + ': ';
    parts.forEach(function (x) {
      if (!x[1]) return;
      var i = document.createElement('i'); i.style.width = (100 * x[1] / n) + '%'; i.style.background = colors[x[0]]; i.title = x[2] + ': ' + x[1];
      bar.appendChild(i); desc += x[1] + ' ' + x[2] + ', ';
    });
    bar.setAttribute('role', 'img'); bar.setAttribute('aria-label', desc.replace(/, $/, ''));
    wrap.appendChild(bar); bars.appendChild(wrap);
  }
  row('Trained', p); row('Random', r);
  var lg = document.createElement('div'); lg.className = 'legend';
  lg.textContent = 'Green: on pad. Yellow: landed off pad or timed out. Red: crashed. Purple: flew out of bounds.';
  bars.appendChild(lg);
  document.getElementById('stats-sub').textContent = R.eval.ppo.episodes + ' fresh episodes per agent, seeds ' + R.eval.seeds + '.';
  document.getElementById('stats-note').textContent = R.config.algorithm + ' on ' + R.config.env_id + ', ' + R.config.total_timesteps.toLocaleString() + ' training steps, CPU only (' + Math.round(R.config.wall_clock_seconds / 60) + ' min). "Solved" means a return of 200 or more.';
})();
