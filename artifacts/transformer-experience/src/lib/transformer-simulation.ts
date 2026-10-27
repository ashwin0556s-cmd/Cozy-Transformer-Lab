export type Matrix = number[][];
export type TensorName = "E" | "P" | "X" | "Q" | "K" | "V" | "Scores" | "Scaled" | "Attention" | "AV" | "Concat" | "Projected" | "Residual" | "Norm" | "FFN" | "Encoder" | "Decoder" | "Logits" | "Probabilities";

export type Simulation = {
  tokens: string[];
  ids: number[];
  embedding: Matrix;
  position: Matrix;
  x: Matrix;
  wq: Matrix;
  wk: Matrix;
  wv: Matrix;
  q: Matrix;
  k: Matrix;
  v: Matrix;
  scores: Matrix;
  scaled: Matrix;
  attention: Matrix;
  av: Matrix;
  heads: Matrix[];
  concat: Matrix;
  projected: Matrix;
  residual: Matrix;
  norm: Matrix;
  ffn: Matrix;
  encoder: Matrix;
  decoderMasked: Matrix;
  crossAttention: Matrix;
  decoder: Matrix;
  logits: Matrix;
  probabilities: Matrix;
  vocabulary: string[];
};

const D = 8;
const PROJECTION = 4;
const HEADS = 4;
const EPS = 1e-5;
const round = (n: number) => Math.round(n * 10000) / 10000;
const hash = (text: string) => [...text.toLowerCase()].reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 17);
const seeded = (seed: number, scale = 0.42) => {
  let n = seed >>> 0;
  n = (n * 1664525 + 1013904223) >>> 0;
  return round((((n / 4294967296) * 2) - 1) * scale);
};
const transpose = (a: Matrix): Matrix => a[0].map((_, j) => a.map((row) => row[j]));
const matmul = (a: Matrix, b: Matrix): Matrix => a.map((row) => b[0].map((_, j) => round(row.reduce((sum, value, k) => sum + value * b[k][j], 0))));
const add = (a: Matrix, b: Matrix): Matrix => a.map((row, i) => row.map((x, j) => round(x + b[i][j])));
const softmax = (a: Matrix, maskFuture = false): Matrix => a.map((row, i) => {
  const allowed = row.map((v, j) => maskFuture && j > i ? Number.NEGATIVE_INFINITY : v);
  const max = Math.max(...allowed);
  const exps = allowed.map((v) => Number.isFinite(v) ? Math.exp(v - max) : 0);
  const sum = exps.reduce((s, x) => s + x, 0);
  return exps.map((v) => round(v / sum));
});
const layerNorm = (a: Matrix): Matrix => a.map((row) => {
  const mean = row.reduce((s, x) => s + x, 0) / row.length;
  const variance = row.reduce((s, x) => s + (x - mean) ** 2, 0) / row.length;
  return row.map((x) => round((x - mean) / Math.sqrt(variance + EPS)));
});
const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
const matrix = (rows: number, cols: number, seed: number, scale = 0.42): Matrix => Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => seeded(seed + i * 97 + j * 31, scale)));
const linear = (a: Matrix, b: Matrix) => matmul(a, b);
const cleanTokens = (value: string) => (value.trim().match(/\w+|[^\w\s]/g) ?? ["The"]).slice(0, 6);

export function simulateTransformer(input: string): Simulation {
  const tokens = cleanTokens(input);
  const n = tokens.length;
  const ids = tokens.map((token) => 100 + hash(token) % 800);
  const embedding = tokens.map((token) => Array.from({ length: D }, (_, j) => seeded(hash(token) + j * 71, 0.75)));
  const position = tokens.map((_, i) => Array.from({ length: D }, (_, j) => round(j % 2 === 0 ? Math.sin(i / 10000 ** (j / D)) : Math.cos(i / 10000 ** ((j - 1) / D)))));
  const x = add(embedding, position);
  const wq = matrix(D, PROJECTION, 3, 0.34);
  const wk = matrix(D, PROJECTION, 13, 0.34);
  const wv = matrix(D, PROJECTION, 23, 0.34);
  const q = linear(x, wq);
  const k = linear(x, wk);
  const v = linear(x, wv);
  const scores = matmul(q, transpose(k)).map((row) => row.map(round));
  const scaled = scores.map((row) => row.map((xv) => round(xv / Math.sqrt(PROJECTION))));
  const attention = softmax(scaled);
  const av = matmul(attention, v);
  const heads: Matrix[] = Array.from({ length: HEADS }, (_, h) => {
    const start = h * (PROJECTION / HEADS);
    const qh = q.map((r) => r.slice(start, start + PROJECTION / HEADS));
    const kh = k.map((r) => r.slice(start, start + PROJECTION / HEADS));
    const vh = v.map((r) => r.slice(start, start + PROJECTION / HEADS));
    const ah = softmax(matmul(qh, transpose(kh)).map((row) => row.map((z) => round(z / Math.sqrt(PROJECTION / HEADS)))));
    return matmul(ah, vh);
  });
  const concat = tokens.map((_, i) => heads.flatMap((head) => head[i]));
  const projected = matmul(concat, matrix(PROJECTION, D, 41, 0.28));
  const residual = add(x, projected);
  const norm = layerNorm(residual);
  const w1 = matrix(D, 12, 59, 0.3);
  const w2 = matrix(12, D, 73, 0.28);
  const ffn = matmul(norm, w1).map((row) => row.map(gelu)).map((row) => matmul([row], w2)[0]);
  let encoder = layerNorm(add(norm, ffn));
  for (let layer = 1; layer < 3; layer += 1) {
    const contextual = matmul(softmax(matmul(encoder, transpose(encoder)).map((row) => row.map((z) => round(z / Math.sqrt(D)))), false), encoder);
    const afterAttention = layerNorm(add(encoder, contextual));
    const layerHidden = matmul(afterAttention, matrix(D, 12, 155 + layer * 11, 0.26)).map((row) => row.map(gelu));
    const layerFfn = matmul(layerHidden, matrix(12, D, 175 + layer * 13, 0.24));
    encoder = layerNorm(add(afterAttention, layerFfn));
  }
  const decoderMasked = softmax(matmul(encoder, transpose(encoder)).map((row) => row.map((z) => round(z / Math.sqrt(D)))), true);
  const maskedContext = matmul(decoderMasked, encoder);
  const crossAttention = matmul(softmax(matmul(maskedContext, transpose(encoder)).map((row) => row.map((z) => round(z / Math.sqrt(D)))), false), encoder);
  const decoderAfterCross = layerNorm(add(maskedContext, crossAttention));
  const decoderFfn = matmul(decoderAfterCross, matrix(D, 12, 211, 0.25)).map((row) => row.map(gelu)).map((row) => matmul([row], matrix(12, D, 223, 0.24))[0]);
  const decoder = layerNorm(add(decoderAfterCross, decoderFfn));
  const vocabulary = ["the", "animal", "was", "tired", "is", "very", "and", "can"];
  const logits = matmul([decoder[n - 1]], matrix(D, vocabulary.length, 101, 0.55));
  const probabilities = softmax(logits);
  return { tokens, ids, embedding, position, x, wq, wk, wv, q, k, v, scores, scaled, attention, av, heads, concat, projected, residual, norm, ffn, encoder, decoderMasked, crossAttention, decoder, logits, probabilities, vocabulary };
}

export const tensorMatrix = (simulation: Simulation, name: TensorName): Matrix => {
  if (name === "E") return simulation.embedding;
  if (name === "P") return simulation.position;
  if (name === "X") return simulation.x;
  if (name === "Q") return simulation.q;
  if (name === "K") return simulation.k;
  if (name === "V") return simulation.v;
  if (name === "Scores") return simulation.scores;
  if (name === "Scaled") return simulation.scaled;
  if (name === "Attention") return simulation.attention;
  if (name === "AV") return simulation.av;
  if (name === "Concat") return simulation.concat;
  if (name === "Projected") return simulation.projected;
  if (name === "Residual") return simulation.residual;
  if (name === "Norm") return simulation.norm;
  if (name === "FFN") return simulation.ffn;
  if (name === "Encoder") return simulation.encoder;
  if (name === "Decoder") return simulation.decoder;
  if (name === "Logits") return simulation.logits;
  return simulation.probabilities;
};

export const stageData = [
  ["Tokenization", "Text is split into token pieces; IDs are deterministic illustrative stand-ins, not tokenizer output."],
  ["Embedding + position", "Embedding lookup E and positional signal P are added: X = E + P."],
  ["Input tensor X", "X has n token rows and d_model = 8 illustrative channels."],
  ["Q / K / V projections", "Three learned-space analogues are computed as Q = XWQ, K = XWK, V = XWV; each projection maps 8 channels to 4."],
  ["Raw scores QKᵀ", "Each query row is dotted with every key row. Transpose aligns key features for pairwise comparison."],
  ["Scale by √dₖ", "Every score is divided by √4 before normalization."],
  ["Row-wise softmax", "Softmax is applied across keys independently for each query; each row sums to one."],
  ["Weighted values AV", "Attention weights multiply V, producing one context vector per query token."],
  ["Heads + concatenation", "Four independent 1-channel head computations are concatenated to 4 channels, then projected back to 8."],
  ["Output projection", "The concatenated heads are projected back to d_model."],
  ["Residual + LayerNorm", "The projected attention output is added to X; LayerNorm then normalizes each token row."],
  ["FFN + second norm", "A position-wise 8→12→8 MLP uses GELU, then a second residual add and LayerNorm."],
  ["Encoder stack", "The example repeats the encoder block as a compact stack; all displayed tensors remain deterministic."],
  ["Decoder masking", "Causal self-attention masks future positions; this prevents access to later target tokens."],
  ["Cross-attention", "Decoder-side queries retrieve encoder-side representations as keys and values."],
  ["Vocabulary logits", "The final decoder state is linearly projected to one score per illustrative vocabulary item."],
  ["Softmax + next token", "Softmax converts logits to probabilities; deterministic argmax selects the next illustrative token."],
] as const;
