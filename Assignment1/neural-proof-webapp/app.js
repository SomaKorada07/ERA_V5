/* Neural Network Proof Lab
   Static browser app: deterministic data, tiny neural nets, canvas plots.
*/

class RNG {
  constructor(seed = 123456) {
    this.seed = seed >>> 0;
    this._spare = null;
  }
  next() {
    this.seed = (1664525 * this.seed + 1013904223) >>> 0;
    return this.seed / 4294967296;
  }
  randn() {
    if (this._spare !== null) {
      const v = this._spare;
      this._spare = null;
      return v;
    }
    let u = this.next();
    let v = this.next();
    if (u < 1e-12) u = 1e-12;
    const mag = Math.sqrt(-2 * Math.log(u));
    const z0 = mag * Math.cos(2 * Math.PI * v);
    const z1 = mag * Math.sin(2 * Math.PI * v);
    this._spare = z1;
    return z0;
  }
  int(max) {
    return Math.floor(this.next() * max);
  }
  choice(arr) {
    return arr[this.int(arr.length)];
  }
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const sigmoid = (z) => 1 / (1 + Math.exp(-clamp(z, -40, 40)));
const fmt = (x, d = 3) => Number.isFinite(x) ? x.toFixed(d) : 'n/a';
const pct = (x) => `${(100 * x).toFixed(1)}%`;
const sleep = (ms = 20) => new Promise(resolve => setTimeout(resolve, ms));

const state = {
  ringData: null,
  s11: null,
  s12: null,
  s13: null,
  s14: null,
  running: false
};

function setStatus(text) {
  document.getElementById('overallStatus').textContent = text;
}

function disableButtons(disabled) {
  document.querySelectorAll('button').forEach(btn => btn.disabled = disabled);
}

function generateRings(n = 320, seed = 7) {
  const rng = new RNG(seed);
  const data = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const a = rng.next() * Math.PI * 2;
    const r = 0.72 + 0.08 * rng.randn();
    data.push({
      x: r * Math.cos(a) + 0.025 * rng.randn(),
      y: r * Math.sin(a) + 0.025 * rng.randn(),
      label: 0
    });
  }
  for (let i = 0; i < n - half; i++) {
    const a = rng.next() * Math.PI * 2;
    const r = 1.55 + 0.10 * rng.randn();
    data.push({
      x: r * Math.cos(a) + 0.025 * rng.randn(),
      y: r * Math.sin(a) + 0.025 * rng.randn(),
      label: 1
    });
  }
  for (let i = data.length - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    [data[i], data[j]] = [data[j], data[i]];
  }
  return data;
}

function boundsFor(data) {
  const xs = data.map(d => d.x);
  const ys = data.map(d => d.y);
  const margin = 0.35;
  return {
    xmin: Math.min(...xs) - margin,
    xmax: Math.max(...xs) + margin,
    ymin: Math.min(...ys) - margin,
    ymax: Math.max(...ys) + margin
  };
}

function drawDecision(canvas, data, predict, opts = {}) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const b = boundsFor(data);
  const sx = (x) => (x - b.xmin) / (b.xmax - b.xmin) * W;
  const sy = (y) => H - (y - b.ymin) / (b.ymax - b.ymin) * H;
  const ux = (px) => b.xmin + px / W * (b.xmax - b.xmin);
  const uy = (py) => b.ymax - py / H * (b.ymax - b.ymin);

  ctx.clearRect(0, 0, W, H);
  const step = opts.step || 5;
  for (let py = 0; py < H; py += step) {
    for (let px = 0; px < W; px += step) {
      const p = predict(ux(px + step / 2), uy(py + step / 2));
      const t = clamp(p, 0, 1);
      const r = Math.round(251 * (1 - t) + 56 * t);
      const g = Math.round(113 * (1 - t) + 189 * t);
      const bl = Math.round(133 * (1 - t) + 248 * t);
      ctx.fillStyle = `rgba(${r},${g},${bl},0.32)`;
      ctx.fillRect(px, py, step, step);
    }
  }

  // Draw approximate 0.5 contour.
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.92)';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 5]);
  const contourStep = opts.contourStep || 10;
  for (let py = 0; py < H; py += contourStep) {
    let last = predict(ux(0), uy(py)) - 0.5;
    for (let px = contourStep; px < W; px += contourStep) {
      const cur = predict(ux(px), uy(py)) - 0.5;
      if (last === 0 || cur === 0 || last * cur < 0) {
        ctx.beginPath();
        ctx.moveTo(px - contourStep, py);
        ctx.lineTo(px, py);
        ctx.stroke();
      }
      last = cur;
    }
  }
  for (let px = 0; px < W; px += contourStep) {
    let last = predict(ux(px), uy(0)) - 0.5;
    for (let py = contourStep; py < H; py += contourStep) {
      const cur = predict(ux(px), uy(py)) - 0.5;
      if (last === 0 || cur === 0 || last * cur < 0) {
        ctx.beginPath();
        ctx.moveTo(px, py - contourStep);
        ctx.lineTo(px, py);
        ctx.stroke();
      }
      last = cur;
    }
  }
  ctx.restore();

  // Axes.
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.13)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(sx(0), 0);
  ctx.lineTo(sx(0), H);
  ctx.moveTo(0, sy(0));
  ctx.lineTo(W, sy(0));
  ctx.stroke();
  ctx.restore();

  // Points.
  for (const d of data) {
    ctx.beginPath();
    ctx.arc(sx(d.x), sy(d.y), opts.pointRadius || 4, 0, Math.PI * 2);
    ctx.fillStyle = d.label === 0 ? '#fb7185' : '#38bdf8';
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(255,255,255,0.72)';
    ctx.stroke();
  }
}

function evaluateBinary(data, predict) {
  let correct = 0;
  let loss = 0;
  for (const d of data) {
    const p = clamp(predict(d.x, d.y), 1e-6, 1 - 1e-6);
    const y = d.label;
    loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
    if ((p >= 0.5 ? 1 : 0) === y) correct++;
  }
  return { accuracy: correct / data.length, loss: loss / data.length };
}

function trainLogistic(data, seed = 1, epochs = 1200, lr = 0.1) {
  const rng = new RNG(seed);
  let w0 = 0.1 * rng.randn();
  let w1 = 0.1 * rng.randn();
  let b = 0;
  const n = data.length;
  for (let e = 0; e < epochs; e++) {
    let dw0 = 0, dw1 = 0, db = 0;
    for (const d of data) {
      const p = sigmoid(w0 * d.x + w1 * d.y + b);
      const dz = p - d.label;
      dw0 += dz * d.x;
      dw1 += dz * d.y;
      db += dz;
    }
    w0 -= lr * dw0 / n;
    w1 -= lr * dw1 / n;
    b -= lr * db / n;
  }
  const predict = (x, y) => sigmoid(w0 * x + w1 * y + b);
  return { kind: 'logistic', w: [w0, w1], b, predict, metrics: evaluateBinary(data, predict) };
}

function makeLayer(fanIn, fanOut, rng, relu, identity = false) {
  const W = Array.from({ length: fanIn }, (_, i) => Array.from({ length: fanOut }, (_, j) => {
    if (identity && fanIn === fanOut) return (i === j ? 1 : 0) + 0.02 * rng.randn();
    const scale = relu ? Math.sqrt(2 / fanIn) : 0.16;
    return scale * rng.randn();
  }));
  const b = Array.from({ length: fanOut }, () => 0);
  return { W, b };
}

function trainDense(data, sizes, config = {}) {
  const rng = new RNG(config.seed || 1);
  const epochs = config.epochs || 1200;
  const lr = config.lr || 0.04;
  const activation = config.activation || 'relu';
  const linearOnly = activation === 'linear';
  const layers = [];
  for (let i = 0; i < sizes.length - 1; i++) {
    const isHidden = i < sizes.length - 2;
    const identity = config.identityLinear && isHidden && sizes[i] === sizes[i + 1];
    layers.push(makeLayer(sizes[i], sizes[i + 1], rng, !linearOnly && isHidden, identity));
  }
  const n = data.length;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const grad = layers.map(layer => ({
      W: layer.W.map(row => row.map(() => 0)),
      b: layer.b.map(() => 0)
    }));

    for (const sample of data) {
      const A = [[sample.x, sample.y]];
      const Z = [];
      for (let l = 0; l < layers.length; l++) {
        const layer = layers[l];
        const prev = A[A.length - 1];
        const z = Array.from({ length: layer.b.length }, (_, j) => {
          let sum = layer.b[j];
          for (let i = 0; i < prev.length; i++) sum += prev[i] * layer.W[i][j];
          return sum;
        });
        Z.push(z);
        if (l === layers.length - 1) {
          A.push([sigmoid(z[0])]);
        } else if (linearOnly) {
          A.push(z.slice());
        } else {
          A.push(z.map(v => Math.max(0, v)));
        }
      }

      let dz = [A[A.length - 1][0] - sample.label];
      for (let l = layers.length - 1; l >= 0; l--) {
        const prevA = A[l];
        for (let i = 0; i < prevA.length; i++) {
          for (let j = 0; j < dz.length; j++) {
            grad[l].W[i][j] += prevA[i] * dz[j];
          }
        }
        for (let j = 0; j < dz.length; j++) grad[l].b[j] += dz[j];
        if (l > 0) {
          const nextDz = Array.from({ length: layers[l].W.length }, () => 0);
          for (let i = 0; i < layers[l].W.length; i++) {
            for (let j = 0; j < dz.length; j++) nextDz[i] += dz[j] * layers[l].W[i][j];
          }
          if (!linearOnly) {
            for (let i = 0; i < nextDz.length; i++) {
              nextDz[i] *= Z[l - 1][i] > 0 ? 1 : 0;
            }
          }
          dz = nextDz;
        }
      }
    }

    for (let l = 0; l < layers.length; l++) {
      for (let i = 0; i < layers[l].W.length; i++) {
        for (let j = 0; j < layers[l].W[i].length; j++) {
          layers[l].W[i][j] -= lr * grad[l].W[i][j] / n;
        }
      }
      for (let j = 0; j < layers[l].b.length; j++) layers[l].b[j] -= lr * grad[l].b[j] / n;
    }
  }

  const predict = (x, y) => {
    let a = [x, y];
    for (let l = 0; l < layers.length; l++) {
      const layer = layers[l];
      const z = Array.from({ length: layer.b.length }, (_, j) => {
        let sum = layer.b[j];
        for (let i = 0; i < a.length; i++) sum += a[i] * layer.W[i][j];
        return sum;
      });
      if (l === layers.length - 1) a = [sigmoid(z[0])];
      else a = linearOnly ? z : z.map(v => Math.max(0, v));
    }
    return a[0];
  };
  return { layers, predict, metrics: evaluateBinary(data, predict) };
}

function collapseLinearLayers(layers) {
  // Row-vector convention: output = x W_eff + b_eff.
  let W = layers[0].W.map(row => row.slice());
  let b = layers[0].b.slice();
  for (let l = 1; l < layers.length; l++) {
    const next = layers[l];
    const newW = Array.from({ length: W.length }, () => Array.from({ length: next.W[0].length }, () => 0));
    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < next.W[0].length; j++) {
        for (let k = 0; k < W[0].length; k++) newW[i][j] += W[i][k] * next.W[k][j];
      }
    }
    const newB = Array.from({ length: next.W[0].length }, (_, j) => {
      let sum = next.b[j];
      for (let k = 0; k < b.length; k++) sum += b[k] * next.W[k][j];
      return sum;
    });
    W = newW;
    b = newB;
  }
  return { W, b };
}

async function runS11() {
  setStatus('S1-1');
  const data = state.ringData || generateRings();
  state.ringData = data;
  document.getElementById('linearStats').textContent = 'training...';
  document.getElementById('reluStats').textContent = 'training...';
  await sleep(10);
  const linear = trainLogistic(data, 2, 1200, 0.1);
  drawDecision(document.getElementById('linearCanvas'), data, linear.predict);
  document.getElementById('linearStats').textContent = `${pct(linear.metrics.accuracy)} acc · loss ${fmt(linear.metrics.loss)}`;
  await sleep(10);
  const relu = trainDense(data, [2, 16, 1], { seed: 4, epochs: 1200, lr: 0.05, activation: 'relu' });
  drawDecision(document.getElementById('reluCanvas'), data, relu.predict);
  document.getElementById('reluStats').textContent = `${pct(relu.metrics.accuracy)} acc · loss ${fmt(relu.metrics.loss)}`;
  document.getElementById('s11Takeaway').innerHTML = `<strong>Proof:</strong> the linear model is stuck at ${pct(linear.metrics.accuracy)} with one straight split. The only architecture change is a ReLU hidden layer, and it reaches ${pct(relu.metrics.accuracy)} by wrapping the ring.`;
  state.s11 = { linear, relu };
}

async function runS12() {
  setStatus('S1-2');
  const data = state.ringData || generateRings();
  state.ringData = data;
  document.getElementById('oneLinearStats').textContent = 'training...';
  document.getElementById('fiveLinearStats').textContent = 'training...';
  document.getElementById('fiveReluStats').textContent = 'training...';
  await sleep(10);

  const one = trainLogistic(data, 12, 1200, 0.1);
  drawDecision(document.getElementById('oneLinearCanvas'), data, one.predict, { pointRadius: 3.2 });
  document.getElementById('oneLinearStats').textContent = `${pct(one.metrics.accuracy)} acc`;

  await sleep(10);
  const fiveLinear = trainDense(data, [2, 2, 2, 2, 2, 1], {
    seed: 5,
    epochs: 1200,
    lr: 0.08,
    activation: 'linear',
    identityLinear: true
  });
  drawDecision(document.getElementById('fiveLinearCanvas'), data, fiveLinear.predict, { pointRadius: 3.2 });
  document.getElementById('fiveLinearStats').textContent = `${pct(fiveLinear.metrics.accuracy)} acc`;

  const collapsed = collapseLinearLayers(fiveLinear.layers);
  const wText = `W_eff = [ ${fmt(collapsed.W[0][0], 5)}\n          ${fmt(collapsed.W[1][0], 5)} ]\n\nb_eff = ${fmt(collapsed.b[0], 5)}\n\nSo five linear layers compute sigmoid(x · W_eff + b_eff): one linear boundary.`;
  document.getElementById('productMatrix').textContent = wText;

  await sleep(10);
  const fiveRelu = trainDense(data, [2, 8, 8, 8, 8, 1], {
    seed: 6,
    epochs: 800,
    lr: 0.04,
    activation: 'relu'
  });
  drawDecision(document.getElementById('fiveReluCanvas'), data, fiveRelu.predict, { pointRadius: 3.2 });
  document.getElementById('fiveReluStats').textContent = `${pct(fiveRelu.metrics.accuracy)} acc`;
  state.s12 = { one, fiveLinear, fiveRelu, collapsed };
}

function softmax(logits) {
  const m = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / s);
}

function trainToyEmbeddings(seed = 21) {
  const rng = new RNG(seed);
  const animals = ['cat', 'dog', 'cow'];
  const fruits = ['apple', 'mango'];
  const verbs = ['eat', 'chase', 'see'];
  const animalNext = ['runs', 'sleeps', 'moves'];
  const fruitNext = ['sweet', 'ripe', 'fresh'];
  const verbNext = ['cat', 'dog', 'cow', 'apple', 'mango'];
  const other = ['runs', 'sleeps', 'moves', 'sweet', 'ripe', 'fresh'];
  const vocab = [...animals, ...fruits, ...verbs, ...other];
  const index = new Map(vocab.map((t, i) => [t, i]));
  const categories = {};
  for (const t of animals) categories[t] = 'animal';
  for (const t of fruits) categories[t] = 'fruit';
  for (const t of verbs) categories[t] = 'verb';
  for (const t of other) categories[t] = 'next-token';

  const pairs = [];
  for (let r = 0; r < 900; r++) {
    const a = rng.choice(animals);
    pairs.push([index.get(a), index.get(rng.choice(animalNext))]);
    const f = rng.choice(fruits);
    pairs.push([index.get(f), index.get(rng.choice(fruitNext))]);
    const v = rng.choice(verbs);
    pairs.push([index.get(v), index.get(rng.choice(verbNext))]);
  }

  const V = vocab.length;
  const D = 6;
  const E = Array.from({ length: V }, () => Array.from({ length: D }, () => 0.03 * rng.randn()));
  const W = Array.from({ length: D }, () => Array.from({ length: V }, () => 0.03 * rng.randn()));
  const b = Array.from({ length: V }, () => 0);
  const lr = 0.055;
  const decay = 0.0009;

  for (let epoch = 0; epoch < 44; epoch++) {
    for (let s = pairs.length - 1; s > 0; s--) {
      const j = rng.int(s + 1);
      [pairs[s], pairs[j]] = [pairs[j], pairs[s]];
    }
    for (const [input, target] of pairs) {
      const h = E[input];
      const logits = Array.from({ length: V }, (_, j) => {
        let z = b[j];
        for (let d = 0; d < D; d++) z += h[d] * W[d][j];
        return z;
      });
      const probs = softmax(logits);
      probs[target] -= 1;
      const gradE = Array.from({ length: D }, () => 0);
      for (let d = 0; d < D; d++) {
        for (let j = 0; j < V; j++) gradE[d] += probs[j] * W[d][j];
      }
      for (let d = 0; d < D; d++) {
        for (let j = 0; j < V; j++) W[d][j] -= lr * (h[d] * probs[j] + decay * W[d][j]);
      }
      for (let j = 0; j < V; j++) b[j] -= lr * probs[j];
      for (let d = 0; d < D; d++) E[input][d] -= lr * (gradE[d] + decay * E[input][d]);
    }
  }
  return { vocab, categories, E, focus: [...animals, ...fruits, ...verbs] };
}

function pca2(points) {
  const n = points.length;
  const d = points[0].length;
  const mean = Array.from({ length: d }, (_, j) => points.reduce((s, p) => s + p[j], 0) / n);
  const X = points.map(p => p.map((v, j) => v - mean[j]));
  const C = Array.from({ length: d }, () => Array.from({ length: d }, () => 0));
  for (const row of X) {
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += row[i] * row[j] / Math.max(1, n - 1);
  }
  const mv = (M, v) => M.map(row => row.reduce((s, m, i) => s + m * v[i], 0));
  const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  let v1 = Array.from({ length: d }, (_, i) => i === 0 ? 1 : 0.13);
  for (let k = 0; k < 80; k++) {
    v1 = mv(C, v1);
    const z = norm(v1);
    v1 = v1.map(x => x / z);
  }
  const lambda1 = dot(v1, mv(C, v1));
  const C2 = C.map((row, i) => row.map((val, j) => val - lambda1 * v1[i] * v1[j]));
  let v2 = Array.from({ length: d }, (_, i) => i === 1 ? 1 : -0.07);
  for (let k = 0; k < 80; k++) {
    v2 = mv(C2, v2);
    const proj = dot(v2, v1);
    v2 = v2.map((x, i) => x - proj * v1[i]);
    const z = norm(v2);
    v2 = v2.map(x => x / z);
  }
  return X.map(row => [dot(row, v1), dot(row, v2)]);
}

function cosine(a, b) {
  let num = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) {
    num += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return num / ((Math.sqrt(aa) * Math.sqrt(bb)) || 1);
}

function drawEmbeddings(canvas, model) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0b1020';
  ctx.fillRect(0, 0, W, H);
  const coords = pca2(model.E);
  const xs = coords.map(c => c[0]);
  const ys = coords.map(c => c[1]);
  const xmin = Math.min(...xs) - 0.25, xmax = Math.max(...xs) + 0.25;
  const ymin = Math.min(...ys) - 0.25, ymax = Math.max(...ys) + 0.25;
  const sx = x => 54 + (x - xmin) / (xmax - xmin) * (W - 108);
  const sy = y => H - 54 - (y - ymin) / (ymax - ymin) * (H - 108);
  const colors = {
    animal: '#fb7185',
    fruit: '#fbbf24',
    verb: '#38bdf8',
    'next-token': '#8b93aa'
  };
  ctx.strokeStyle = 'rgba(255,255,255,0.10)';
  ctx.lineWidth = 1;
  for (let i = 1; i < 5; i++) {
    const x = 54 + i * (W - 108) / 5;
    const y = 54 + i * (H - 108) / 5;
    ctx.beginPath(); ctx.moveTo(x, 34); ctx.lineTo(x, H - 34); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(34, y); ctx.lineTo(W - 34, y); ctx.stroke();
  }
  for (let i = 0; i < model.vocab.length; i++) {
    const token = model.vocab[i];
    const cat = model.categories[token];
    const x = sx(coords[i][0]);
    const y = sy(coords[i][1]);
    ctx.beginPath();
    ctx.arc(x, y, model.focus.includes(token) ? 8 : 5, 0, Math.PI * 2);
    ctx.fillStyle = colors[cat];
    ctx.globalAlpha = model.focus.includes(token) ? 0.96 : 0.45;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = 'rgba(255,255,255,0.74)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.font = model.focus.includes(token) ? '800 15px Inter, sans-serif' : '600 12px Inter, sans-serif';
    ctx.fillStyle = model.focus.includes(token) ? '#ffffff' : '#aeb8d6';
    ctx.fillText(token, x + 11, y + 5);
  }
  ctx.font = '700 13px Inter, sans-serif';
  const labels = [['animal', colors.animal], ['fruit', colors.fruit], ['verb', colors.verb], ['next-token target', colors['next-token']]];
  labels.forEach((l, i) => {
    const x = 24 + i * 150;
    ctx.fillStyle = l[1];
    ctx.beginPath(); ctx.arc(x, 24, 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#dbeafe';
    ctx.fillText(l[0], x + 12, 29);
  });
}

function nearestTable(model) {
  const rows = [];
  const focusIdx = model.focus.map(t => model.vocab.indexOf(t));
  for (const t of model.focus) {
    const i = model.vocab.indexOf(t);
    const sims = focusIdx
      .filter(j => j !== i)
      .map(j => ({ token: model.vocab[j], sim: cosine(model.E[i], model.E[j]), cat: model.categories[model.vocab[j]] }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, 2);
    rows.push({ token: t, cat: model.categories[t], n: sims.map(s => `${s.token} (${fmt(s.sim, 2)})`).join(', ') });
  }
  return `<table class="table"><thead><tr><th>Token</th><th>Category</th><th>Nearest learned neighbors</th></tr></thead><tbody>${rows.map(r => `<tr><td><strong>${r.token}</strong></td><td><span class="badge">${r.cat}</span></td><td>${r.n}</td></tr>`).join('')}</tbody></table>`;
}

async function runS13() {
  setStatus('S1-3');
  document.getElementById('embedStats').textContent = 'training...';
  await sleep(10);
  const model = trainToyEmbeddings();
  drawEmbeddings(document.getElementById('embeddingCanvas'), model);
  document.getElementById('neighborTable').innerHTML = nearestTable(model);
  document.getElementById('embedStats').textContent = 'trained on next-token pairs only';
  state.s13 = model;
}

function generateNoisyClassification(n, seed = 100) {
  const rng = new RNG(seed);
  const data = [];
  for (let i = 0; i < n; i++) {
    const x = -2 + 4 * rng.next();
    const y = -2 + 4 * rng.next();
    const score = Math.sin(2.8 * x) + Math.cos(2.2 * y) + 0.35 * Math.sin(3 * x * y);
    let label = score > 0 ? 1 : 0;
    if (rng.next() < 0.08) label = 1 - label;
    data.push({ x, y, label });
  }
  return data;
}

function initNet(sizes, rng) {
  return sizes.slice(0, -1).map((fanIn, l) => {
    const fanOut = sizes[l + 1];
    const hidden = l < sizes.length - 2;
    const scale = hidden ? Math.sqrt(2 / fanIn) : 0.1;
    return {
      W: Array.from({ length: fanIn }, () => Array.from({ length: fanOut }, () => scale * rng.randn())),
      b: Array.from({ length: fanOut }, () => 0)
    };
  });
}

function densePredict(layers, x, y) {
  let a = [x, y];
  for (let l = 0; l < layers.length; l++) {
    const layer = layers[l];
    const z = Array.from({ length: layer.b.length }, (_, j) => {
      let sum = layer.b[j];
      for (let i = 0; i < a.length; i++) sum += a[i] * layer.W[i][j];
      return sum;
    });
    a = l === layers.length - 1 ? [sigmoid(z[0])] : z.map(v => Math.max(0, v));
  }
  return a[0];
}

function evalLayers(data, layers) {
  return evaluateBinary(data, (x, y) => densePredict(layers, x, y));
}

function trainOverParam(train, test, cfg) {
  const rng = new RNG(cfg.seed || 1);
  const sizes = [2, cfg.hidden || 48, cfg.hidden || 48, 1];
  const layers = initNet(sizes, rng);
  const steps = cfg.steps || 1200;
  const batch = Math.min(cfg.batch || 64, train.length);
  const lr = cfg.lr || 0.045;
  const decay = cfg.decay || 0.00002;

  for (let step = 0; step < steps; step++) {
    const grad = layers.map(layer => ({
      W: layer.W.map(row => row.map(() => 0)),
      b: layer.b.map(() => 0)
    }));
    for (let s = 0; s < batch; s++) {
      const sample = train[batch >= train.length ? s % train.length : rng.int(train.length)];
      const A = [[sample.x, sample.y]];
      const Z = [];
      for (let l = 0; l < layers.length; l++) {
        const prev = A[A.length - 1];
        const layer = layers[l];
        const z = Array.from({ length: layer.b.length }, (_, j) => {
          let sum = layer.b[j];
          for (let i = 0; i < prev.length; i++) sum += prev[i] * layer.W[i][j];
          return sum;
        });
        Z.push(z);
        A.push(l === layers.length - 1 ? [sigmoid(z[0])] : z.map(v => Math.max(0, v)));
      }
      let dz = [A[A.length - 1][0] - sample.label];
      for (let l = layers.length - 1; l >= 0; l--) {
        const prevA = A[l];
        for (let i = 0; i < prevA.length; i++) {
          for (let j = 0; j < dz.length; j++) grad[l].W[i][j] += prevA[i] * dz[j];
        }
        for (let j = 0; j < dz.length; j++) grad[l].b[j] += dz[j];
        if (l > 0) {
          const nextDz = Array.from({ length: layers[l].W.length }, () => 0);
          for (let i = 0; i < layers[l].W.length; i++) {
            for (let j = 0; j < dz.length; j++) nextDz[i] += dz[j] * layers[l].W[i][j];
            nextDz[i] *= Z[l - 1][i] > 0 ? 1 : 0;
          }
          dz = nextDz;
        }
      }
    }
    for (let l = 0; l < layers.length; l++) {
      for (let i = 0; i < layers[l].W.length; i++) {
        for (let j = 0; j < layers[l].W[i].length; j++) {
          layers[l].W[i][j] -= lr * (grad[l].W[i][j] / batch + decay * layers[l].W[i][j]);
        }
      }
      for (let j = 0; j < layers[l].b.length; j++) layers[l].b[j] -= lr * grad[l].b[j] / batch;
    }
  }
  return { train: evalLayers(train, layers), test: evalLayers(test, layers), layers };
}

function drawGap(canvas, results) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0b1020';
  ctx.fillRect(0, 0, W, H);

  const left = 70, right = W - 30, top = 36, bottom = H - 66;
  const maxLoss = Math.max(...results.flatMap(r => [r.train.loss, r.test.loss]), 0.75);
  const xs = results.map((_, i) => left + i * (right - left) / (results.length - 1));
  const yScale = v => bottom - v / maxLoss * (bottom - top);

  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;
  ctx.font = '700 12px Inter, sans-serif';
  ctx.fillStyle = '#aeb8d6';
  for (let i = 0; i <= 5; i++) {
    const val = maxLoss * i / 5;
    const y = yScale(val);
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    ctx.fillText(fmt(val, 2), 18, y + 4);
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.28)';
  ctx.beginPath(); ctx.moveTo(left, top); ctx.lineTo(left, bottom); ctx.lineTo(right, bottom); ctx.stroke();

  const line = (key, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    results.forEach((r, i) => {
      const y = yScale(r[key].loss);
      if (i === 0) ctx.moveTo(xs[i], y); else ctx.lineTo(xs[i], y);
    });
    ctx.stroke();
    results.forEach((r, i) => {
      const y = yScale(r[key].loss);
      ctx.beginPath(); ctx.arc(xs[i], y, 7, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 1; ctx.stroke();
    });
  };
  line('train', '#34d399');
  line('test', '#38bdf8');

  ctx.fillStyle = '#dbeafe';
  ctx.font = '800 14px Inter, sans-serif';
  results.forEach((r, i) => ctx.fillText(String(r.n), xs[i] - 14, H - 34));
  ctx.fillText('training examples', (left + right) / 2 - 58, H - 12);
  ctx.save();
  ctx.translate(18, (top + bottom) / 2 + 55);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('cross-entropy loss', 0, 0);
  ctx.restore();

  ctx.font = '800 13px Inter, sans-serif';
  ctx.fillStyle = '#34d399'; ctx.fillRect(right - 172, top + 7, 13, 13); ctx.fillText('train loss', right - 150, top + 19);
  ctx.fillStyle = '#38bdf8'; ctx.fillRect(right - 172, top + 31, 13, 13); ctx.fillText('test loss', right - 150, top + 43);
}

function gapTable(results) {
  return `<table class="table"><thead><tr><th>N</th><th>Train loss</th><th>Test loss</th><th>Gap</th><th>Train/Test acc</th></tr></thead><tbody>${results.map(r => `<tr><td><strong>${r.n}</strong></td><td>${fmt(r.train.loss, 3)}</td><td>${fmt(r.test.loss, 3)}</td><td><span class="badge">${fmt(r.test.loss - r.train.loss, 3)}</span></td><td>${pct(r.train.accuracy)} / ${pct(r.test.accuracy)}</td></tr>`).join('')}</tbody></table>`;
}

async function runS14() {
  setStatus('S1-4');
  document.getElementById('gapStats').textContent = 'training n=20...';
  await sleep(10);
  const test = generateNoisyClassification(800, 99);
  const configs = [
    { n: 20, steps: 1800, batch: 20, seed: 51 },
    { n: 200, steps: 700, batch: 64, seed: 52 },
    { n: 2000, steps: 900, batch: 96, seed: 53 }
  ];
  const results = [];
  for (const cfg of configs) {
    document.getElementById('gapStats').textContent = `training n=${cfg.n}...`;
    await sleep(10);
    const train = generateNoisyClassification(cfg.n, 10 + cfg.n);
    const fit = trainOverParam(train, test, { ...cfg, hidden: 24, lr: 0.045 });
    results.push({ n: cfg.n, train: fit.train, test: fit.test });
  }
  drawGap(document.getElementById('gapCanvas'), results);
  document.getElementById('gapTable').innerHTML = gapTable(results);
  const first = results[0].test.loss - results[0].train.loss;
  const last = results[results.length - 1].test.loss - results[results.length - 1].train.loss;
  document.getElementById('gapStats').textContent = `gap shrinks ${fmt(first, 2)} → ${fmt(last, 2)}`;
  state.s14 = results;
}

function drawPlaceholders() {
  const data = state.ringData || generateRings();
  state.ringData = data;
  const neutral = () => 0.5;
  ['linearCanvas', 'reluCanvas', 'oneLinearCanvas', 'fiveLinearCanvas', 'fiveReluCanvas'].forEach(id => {
    const canvas = document.getElementById(id);
    if (canvas) drawDecision(canvas, data, neutral, { pointRadius: id.includes('Canvas') && canvas.width < 500 ? 3.2 : 4 });
  });
  const eCanvas = document.getElementById('embeddingCanvas');
  const eCtx = eCanvas.getContext('2d');
  eCtx.fillStyle = '#0b1020';
  eCtx.fillRect(0, 0, eCanvas.width, eCanvas.height);
  eCtx.fillStyle = '#aeb8d6';
  eCtx.font = '800 22px Inter, sans-serif';
  eCtx.fillText('Run S1-3 to train embeddings and project them here.', 48, eCanvas.height / 2);
  const gCanvas = document.getElementById('gapCanvas');
  const gCtx = gCanvas.getContext('2d');
  gCtx.fillStyle = '#0b1020';
  gCtx.fillRect(0, 0, gCanvas.width, gCanvas.height);
  gCtx.fillStyle = '#aeb8d6';
  gCtx.font = '800 22px Inter, sans-serif';
  gCtx.fillText('Run S1-4 to plot train loss, test loss, and gap.', 48, gCanvas.height / 2);
}

async function guardedRun(fn) {
  if (state.running) return;
  state.running = true;
  disableButtons(true);
  try {
    await fn();
    setStatus('Done');
  } catch (err) {
    console.error(err);
    setStatus('Error');
    alert(`Something went wrong: ${err.message}`);
  } finally {
    state.running = false;
    disableButtons(false);
  }
}

async function runAll() {
  await runS11();
  await sleep(20);
  await runS12();
  await sleep(20);
  await runS13();
  await sleep(20);
  await runS14();
}

function resetApp() {
  state.ringData = generateRings();
  state.s11 = state.s12 = state.s13 = state.s14 = null;
  setStatus('Ready');
  const ids = {
    linearStats: 'waiting', reluStats: 'waiting', oneLinearStats: 'waiting',
    fiveLinearStats: 'waiting', fiveReluStats: 'waiting', embedStats: 'waiting', gapStats: 'waiting'
  };
  for (const [id, text] of Object.entries(ids)) document.getElementById(id).textContent = text;
  document.getElementById('s11Takeaway').textContent = 'The money shot appears after training: the linear model remains a line; ReLU wraps the ring.';
  document.getElementById('productMatrix').textContent = 'Run S1-2 to compute W_eff and b_eff.';
  document.getElementById('neighborTable').className = 'table-placeholder';
  document.getElementById('neighborTable').textContent = 'Run S1-3 to reveal learned neighbors.';
  document.getElementById('gapTable').className = 'table-placeholder';
  document.getElementById('gapTable').textContent = 'Run S1-4 to train three dataset sizes.';
  drawPlaceholders();
}

window.addEventListener('DOMContentLoaded', () => {
  resetApp();
  document.getElementById('runAllBtn').addEventListener('click', () => guardedRun(runAll));
  document.getElementById('resetBtn').addEventListener('click', resetApp);
  document.getElementById('runS11').addEventListener('click', () => guardedRun(runS11));
  document.getElementById('runS12').addEventListener('click', () => guardedRun(runS12));
  document.getElementById('runS13').addEventListener('click', () => guardedRun(runS13));
  document.getElementById('runS14').addEventListener('click', () => guardedRun(runS14));
});
