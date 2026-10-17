/* NEXUS neural Brain, browser/Node inference port.
 *
 * A line-by-line port of research/.../neural/transformer.py `forward` and
 * model.py `generate` (pre-LayerNorm decoder-only Transformer, GELU, tied
 * embeddings, causal mask, character tokenizer). Inference only: no training.
 *
 * Weights are float32 here (the NumPy checkpoints are float64), so tiny numeric
 * differences are expected; tests/verify_js.mjs checks greedy output matches.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.NexusModel = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var PAD_ID = 0, BOS_ID = 1, UNK_ID = 2;
  var CHARS = [];
  for (var c = 32; c < 127; c++) CHARS.push(String.fromCharCode(c));
  CHARS.push("\n", "\t");
  var CHAR_TO_ID = {}, ID_TO_CHAR = {};
  CHARS.forEach(function (ch, i) { CHAR_TO_ID[ch] = i + 3; ID_TO_CHAR[i + 3] = ch; });
  var VOCAB_SIZE = 3 + CHARS.length; // 100

  var LN_EPS = 1e-5;
  var GELU_C = Math.sqrt(2 / Math.PI);

  function encode(text) {
    var out = [];
    for (var ch of text) out.push(CHAR_TO_ID[ch] !== undefined ? CHAR_TO_ID[ch] : UNK_ID);
    return out;
  }
  function decode(ids) {
    var s = "";
    for (var i = 0; i < ids.length; i++) s += ID_TO_CHAR[ids[i]] || "";
    return s;
  }

  function b64ToFloat32(b64) {
    var bin = atob(b64);
    var buf = new ArrayBuffer(bin.length);
    var u8 = new Uint8Array(buf);
    for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    // The blob is little-endian float32; DataView keeps this correct on any host.
    var dv = new DataView(buf);
    var f = new Float32Array(bin.length / 4);
    for (var j = 0; j < f.length; j++) f[j] = dv.getFloat32(j * 4, true);
    return f;
  }

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function Model(payload) {
    this.meta = payload;
    var cfg = payload.config;
    this.D = cfg.d_model; this.L = cfg.n_layers; this.H = cfg.n_heads;
    this.T = cfg.context_length; this.F = payload.d_ff; this.V = payload.vocab_size;
    if (this.V !== VOCAB_SIZE) throw new Error("vocab mismatch: " + this.V + " vs " + VOCAB_SIZE);
    var flat = b64ToFloat32(payload.weights_f32_b64);
    var p = {}, off = 0;
    payload.keys.forEach(function (k) {
      var n = payload.shapes[k].reduce(function (a, b) { return a * b; }, 1);
      p[k] = flat.subarray(off, off + n);
      off += n;
    });
    if (off !== flat.length) throw new Error("weight blob size mismatch");
    this.p = p;
  }

  // y[t][o] = sum_i x[t][i] * w[i][o] + b[o]; w is row-major (in, out).
  function linear(x, T, nIn, w, b, nOut) {
    var y = new Float32Array(T * nOut);
    for (var t = 0; t < T; t++) {
      var xo = t * nIn, yo = t * nOut;
      for (var o = 0; o < nOut; o++) y[yo + o] = b[o];
      for (var i = 0; i < nIn; i++) {
        var xv = x[xo + i], wo = i * nOut;
        for (var o2 = 0; o2 < nOut; o2++) y[yo + o2] += xv * w[wo + o2];
      }
    }
    return y;
  }

  function layerNorm(x, T, D, g, b) {
    var y = new Float32Array(T * D);
    for (var t = 0; t < T; t++) {
      var o = t * D, mu = 0, v = 0, i;
      for (i = 0; i < D; i++) mu += x[o + i];
      mu /= D;
      for (i = 0; i < D; i++) { var d = x[o + i] - mu; v += d * d; }
      v /= D;
      var inv = 1 / Math.sqrt(v + LN_EPS);
      for (i = 0; i < D; i++) y[o + i] = (x[o + i] - mu) * inv * g[i] + b[i];
    }
    return y;
  }

  function gelu(u) {
    return 0.5 * u * (1 + Math.tanh(GELU_C * (u + 0.044715 * u * u * u)));
  }

  /* tokens: array of ids (length 1..T). Returns logits for the LAST position only
   * (that is all generation needs) as Float32Array(V). */
  Model.prototype.lastLogits = function (tokens) {
    var D = this.D, H = this.H, dh = D / H, T = tokens.length, p = this.p, V = this.V;
    if (T > this.T) throw new Error("sequence too long: " + T + " > " + this.T);
    var scale = 1 / Math.sqrt(dh);
    var x = new Float32Array(T * D), t, i;
    for (t = 0; t < T; t++)
      for (i = 0; i < D; i++)
        x[t * D + i] = p.tok_emb[tokens[t] * D + i] + p.pos_emb[t * D + i];

    for (var l = 0; l < this.L; l++) {
      var pre = "l" + l + ".";
      var h1 = layerNorm(x, T, D, p[pre + "ln1_g"], p[pre + "ln1_b"]);
      var q = linear(h1, T, D, p[pre + "wq"], p[pre + "bq"], D);
      var k = linear(h1, T, D, p[pre + "wk"], p[pre + "bk"], D);
      var v = linear(h1, T, D, p[pre + "wv"], p[pre + "bv"], D);
      var ctx = new Float32Array(T * D);
      var sc = new Float32Array(T);
      for (var h = 0; h < H; h++) {
        var ho = h * dh;
        for (t = 0; t < T; t++) {
          var mx = -Infinity, j, d;
          for (j = 0; j <= t; j++) { // causal: keys 0..t only
            var s = 0;
            for (d = 0; d < dh; d++) s += q[t * D + ho + d] * k[j * D + ho + d];
            s *= scale; sc[j] = s; if (s > mx) mx = s;
          }
          var sum = 0;
          for (j = 0; j <= t; j++) { sc[j] = Math.exp(sc[j] - mx); sum += sc[j]; }
          for (d = 0; d < dh; d++) {
            var acc = 0;
            for (j = 0; j <= t; j++) acc += sc[j] * v[j * D + ho + d];
            ctx[t * D + ho + d] = acc / sum;
          }
        }
      }
      var a = linear(ctx, T, D, p[pre + "wo"], p[pre + "bo"], D);
      for (i = 0; i < T * D; i++) x[i] += a[i];
      var h2 = layerNorm(x, T, D, p[pre + "ln2_g"], p[pre + "ln2_b"]);
      var u = linear(h2, T, D, p[pre + "w1"], p[pre + "b1"], this.F);
      for (i = 0; i < u.length; i++) u[i] = gelu(u[i]);
      var f = linear(u, T, this.F, p[pre + "w2"], p[pre + "b2"], D);
      for (i = 0; i < T * D; i++) x[i] += f[i];
    }
    var xf = layerNorm(x, T, D, p.lnf_g, p.lnf_b);
    var logits = new Float32Array(V), last = (T - 1) * D;
    for (var tok = 0; tok < V; tok++) {
      var z = 0;
      for (i = 0; i < D; i++) z += xf[last + i] * p.tok_emb[tok * D + i]; // tied embeddings
      logits[tok] = z;
    }
    return logits;
  };

  function softmax(logits, temperature) {
    var n = logits.length, out = new Float64Array(n), mx = -Infinity, i;
    var tt = temperature > 0 ? temperature : 1;
    for (i = 0; i < n; i++) { var z = logits[i] / tt; if (z > mx) mx = z; }
    var s = 0;
    for (i = 0; i < n; i++) { out[i] = Math.exp(logits[i] / tt - mx); s += out[i]; }
    for (i = 0; i < n; i++) out[i] /= s;
    return out;
  }

  /* opts: {maxTokens, temperature, seed, stop:[...], onStep(info)}.
   * Mirrors NeuralCodeModel.generate (greedy when temperature <= 0). The special
   * ids (PAD/BOS/UNK) can be sampled by an untrained-ish model; like the Python
   * version they simply decode to "" and are kept in the id stream. */
  Model.prototype.generate = function (prompt, opts) {
    opts = opts || {};
    var maxTokens = opts.maxTokens === undefined ? 40 : opts.maxTokens;
    var temperature = opts.temperature === undefined ? 0 : opts.temperature;
    var stop = opts.stop || [];
    var rng = mulberry32(opts.seed === undefined ? 0 : opts.seed);
    var ids = [BOS_ID].concat(encode(prompt));
    var promptLen = ids.length, reason = "length", steps = [];
    for (var n = 0; n < maxTokens; n++) {
      var window = ids.slice(-this.T);
      var logits = this.lastLogits(window);
      var probs = softmax(logits, temperature);
      var next;
      if (temperature <= 0) {
        next = 0;
        for (var i = 1; i < logits.length; i++) if (logits[i] > logits[next]) next = i;
      } else {
        var r = rng(), acc = 0; next = probs.length - 1;
        for (var j = 0; j < probs.length; j++) { acc += probs[j]; if (r < acc) { next = j; break; } }
      }
      // top-3 under the temperature-1 distribution, for the UI
      var p1 = softmax(logits, 1);
      var order = [];
      for (var q2 = 0; q2 < p1.length; q2++) order.push(q2);
      order.sort(function (a, b) { return p1[b] - p1[a]; });
      var top = order.slice(0, 3).map(function (id) { return { ch: ID_TO_CHAR[id] !== undefined ? ID_TO_CHAR[id] : "<" + id + ">", p: p1[id] }; });
      ids.push(next);
      var info = { index: n, ch: ID_TO_CHAR[next] || "", id: next, p: p1[next], top: top };
      steps.push(info);
      if (opts.onStep) opts.onStep(info);
      var soFar = decode(ids);
      if (stop.some(function (m) { return soFar.endsWith(m); })) { reason = "stop"; break; }
    }
    return { text: decode(ids.slice(promptLen)), ids: ids.slice(promptLen), steps: steps, finish_reason: reason };
  };

  Model.fromPayload = function (payload) { return new Model(payload); };

  return { Model: Model, encode: encode, decode: decode, VOCAB_SIZE: VOCAB_SIZE, BOS_ID: BOS_ID };
});
