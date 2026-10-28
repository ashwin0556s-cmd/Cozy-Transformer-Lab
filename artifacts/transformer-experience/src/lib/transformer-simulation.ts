export type Matrix = number[][];

export type TensorName =
  | "Tokens"
  | "E"
  | "P"
  | "X"
  | "Q"
  | "K"
  | "V"
  | "Scores"
  | "Scaled"
  | "Attention"
  | "Heatmap"
  | "AV"
  | "Heads"
  | "Concat"
  | "Projected"
  | "Residual1"
  | "Norm1"
  | "FFN"
  | "Residual2"
  | "Norm2"
  | "EncoderStack"
  | "DecoderMasked"
  | "CrossAttention"
  | "Decoder"
  | "Logits"
  | "Probabilities"
  | "Autoregressive";

export type LayerNormStats = {
  mean: number;
  variance: number;
  std: number;
  gamma: number[];
  beta: number[];
};

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
  headDescriptions: string[];
  concat: Matrix;
  wo: Matrix;
  projected: Matrix;
  residual1: Matrix;
  norm1: Matrix;
  norm1Stats: LayerNormStats[];
  w1: Matrix;
  b1: number[];
  ffnHidden: Matrix;
  w2: Matrix;
  b2: number[];
  ffn: Matrix;
  residual2: Matrix;
  norm2: Matrix;
  norm2Stats: LayerNormStats[];
  encoderStack: Matrix[];
  decoderMasked: Matrix;
  crossAttention: Matrix;
  decoder: Matrix;
  wVocab: Matrix;
  logits: Matrix;
  probabilities: Matrix;
  vocabulary: string[];
  coreferenceInfo?: {
    queryIndex: number;
    keyIndex: number;
    queryToken: string;
    keyToken: string;
    weight: number;
    description: string;
  };
};

export const D = 8;
export const PROJECTION = 4;
export const HEADS = 4;
export const FFN_EXPANSION = 16;
export const EPS = 1e-5;

export const round = (n: number) => (Object.is(n, -0) ? 0 : Math.round(n * 10000) / 10000);

export const hash = (text: string) =>
  [...text.toLowerCase()].reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, 17);

export const seeded = (seed: number, scale = 0.42) => {
  let n = seed >>> 0;
  n = (n * 1664525 + 1013904223) >>> 0;
  return round((((n / 4294967296) * 2) - 1) * scale);
};

export const transpose = (a: Matrix): Matrix => {
  if (!a || a.length === 0 || !a[0]) return [];
  return a[0].map((_, j) => a.map((row) => row[j]));
};

export const matmul = (a: Matrix, b: Matrix): Matrix => {
  if (!a || !b || a.length === 0 || b.length === 0 || !a[0] || !b[0]) return [];
  return a.map((row) =>
    b[0].map((_, j) => round(row.reduce((sum, value, k) => sum + value * (b[k]?.[j] ?? 0), 0)))
  );
};

export const add = (a: Matrix, b: Matrix): Matrix =>
  a.map((row, i) => row.map((x, j) => round(x + (b[i]?.[j] ?? 0))));

export const softmax = (a: Matrix, maskFuture = false): Matrix =>
  a.map((row, i) => {
    const allowed = row.map((v, j) => (maskFuture && j > i ? Number.NEGATIVE_INFINITY : v));
    const finiteVals = allowed.filter(Number.isFinite);
    const max = finiteVals.length > 0 ? Math.max(...finiteVals) : 0;
    const exps = allowed.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
    const sum = exps.reduce((s, x) => s + x, 0);
    return exps.map((v) => (sum > 0 ? round(v / sum) : 0));
  });

export const computeLayerNormWithStats = (a: Matrix): { normalized: Matrix; stats: LayerNormStats[] } => {
  const stats: LayerNormStats[] = [];
  const normalized = a.map((row) => {
    const mean = row.reduce((s, x) => s + x, 0) / row.length;
    const variance = row.reduce((s, x) => s + (x - mean) ** 2, 0) / row.length;
    const std = Math.sqrt(variance + EPS);
    const gamma = Array.from({ length: row.length }, () => 1.0);
    const beta = Array.from({ length: row.length }, () => 0.0);
    stats.push({
      mean: round(mean),
      variance: round(variance),
      std: round(std),
      gamma,
      beta,
    });
    return row.map((x, j) => round(((x - mean) / std) * gamma[j] + beta[j]));
  });
  return { normalized, stats };
};

export const gelu = (x: number) =>
  round(0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))));

export const matrix = (rows: number, cols: number, seed: number, scale = 0.42): Matrix =>
  Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => seeded(seed + i * 97 + j * 31, scale))
  );

export const cleanTokens = (value: string): string[] => {
  const matches = value.trim().match(/[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[.,!?;]/g);
  if (!matches || matches.length === 0) return ["The", "animal", "was", "tired"];
  return matches.slice(0, 10);
};

export function simulateTransformer(input: string): Simulation {
  const tokens = cleanTokens(input);
  const n = tokens.length;

  // 1. Token IDs
  const ids = tokens.map((token) => 100 + (hash(token) % 800));

  // 2. Embedding Lookup (n x D)
  const embedding = tokens.map((token) =>
    Array.from({ length: D }, (_, j) => seeded(hash(token) + j * 71, 0.75))
  );

  // 3. Positional Encoding (n x D)
  const position = tokens.map((_, i) =>
    Array.from({ length: D }, (_, j) =>
      round(
        j % 2 === 0
          ? Math.sin(i / 10000 ** (j / D))
          : Math.cos(i / 10000 ** ((j - 1) / D))
      )
    )
  );

  // 4. Input Matrix X = E + P (n x D)
  const x = add(embedding, position);

  // 5. Weight matrices WQ, WK, WV (D x PROJECTION = 8 x 4)
  const wq = matrix(D, PROJECTION, 7, 0.38);
  const wk = matrix(D, PROJECTION, 17, 0.38);
  const wv = matrix(D, PROJECTION, 27, 0.38);

  // 6. Projections Q, K, V (n x 4)
  let q = matmul(x, wq);
  let k = matmul(x, wk);
  const v = matmul(x, wv);

  // Coreference enhancement for educational clarity:
  // If "it" and "animal" exist, align query "it" and key "animal" to produce a pronounced attention link.
  const itIndex = tokens.findIndex((t) => t.toLowerCase() === "it");
  const animalIndex = tokens.findIndex((t) => t.toLowerCase() === "animal");
  let coreferenceInfo: Simulation["coreferenceInfo"] = undefined;

  if (itIndex >= 0 && animalIndex >= 0) {
    // Boost matching feature dimensions between query "it" and key "animal"
    q = q.map((row, i) =>
      i === itIndex ? [0.85, 0.45, -0.22, 0.68] : row
    );
    k = k.map((row, i) =>
      i === animalIndex ? [0.92, 0.48, -0.20, 0.74] : row
    );
  }

  // 7. Attention Scores QKᵀ (n x n)
  const scores = matmul(q, transpose(k));

  // 8. Scale by √d_k (√4 = 2)
  const sqrtDk = Math.sqrt(PROJECTION);
  const scaled = scores.map((row) => row.map((val) => round(val / sqrtDk)));

  // 9. Softmax Attention Weights (n x n)
  const attention = softmax(scaled);

  if (itIndex >= 0 && animalIndex >= 0) {
    coreferenceInfo = {
      queryIndex: itIndex,
      keyIndex: animalIndex,
      queryToken: tokens[itIndex],
      keyToken: tokens[animalIndex],
      weight: attention[itIndex][animalIndex],
      description: `Coreference resolution: Query "${tokens[itIndex]}" attends strongly to Key "${tokens[animalIndex]}" (${(attention[itIndex][animalIndex] * 100).toFixed(1)}%), identifying what was tired.`,
    };
  }

  // 10. Weighted Value Aggregation AV (n x 4)
  const av = matmul(attention, v);

  // 11. Multi-Head Attention (4 heads, each n x 1, concatenated to n x 4)
  const headDescriptions = [
    "Head 1 · Local syntax & adjacent tokens",
    "Head 2 · Coreference & long-range pronoun links",
    "Head 3 · Verbs, actions & direct objects",
    "Head 4 · Global sentence context & theme",
  ];

  const heads: Matrix[] = Array.from({ length: HEADS }, (_, h) => {
    const qh = q.map((r, i) => [seeded(hash(tokens[i]) + h * 43, 0.5) + (h === 1 && i === itIndex ? 0.6 : 0)]);
    const kh = k.map((r, i) => [seeded(hash(tokens[i]) + h * 47, 0.5) + (h === 1 && i === animalIndex ? 0.6 : 0)]);
    const vh = v.map((r, i) => [seeded(hash(tokens[i]) + h * 53, 0.5)]);
    const rawHeadScores = matmul(qh, transpose(kh));
    const headAttn = softmax(rawHeadScores);
    return matmul(headAttn, vh);
  });

  // Concatenate heads: each row gets 1 feature per head => n x 4
  const concat = tokens.map((_, i) => heads.flatMap((head) => head[i]));

  // Output Projection WO (4 x 8)
  const wo = matrix(PROJECTION, D, 41, 0.32);
  const projected = matmul(concat, wo);

  // 12. Residual Connection 1: X + Attention(X) (n x 8)
  const residual1 = add(x, projected);

  // 13. Layer Normalization 1 (n x 8)
  const { normalized: norm1, stats: norm1Stats } = computeLayerNormWithStats(residual1);

  // 14. Feed-Forward Network (FFN): W1 (8 x 16), b1, GELU, W2 (16 x 8), b2
  const w1 = matrix(D, FFN_EXPANSION, 59, 0.35);
  const b1 = Array.from({ length: FFN_EXPANSION }, (_, j) => seeded(111 + j, 0.05));
  const ffnHidden = matmul(norm1, w1).map((row) =>
    row.map((val, j) => gelu(val + b1[j]))
  );

  const w2 = matrix(FFN_EXPANSION, D, 73, 0.32);
  const b2 = Array.from({ length: D }, (_, j) => seeded(222 + j, 0.05));
  const ffn = matmul(ffnHidden, w2).map((row) =>
    row.map((val, j) => round(val + b2[j]))
  );

  // 15. Residual Connection 2: norm1 + FFN (n x 8)
  const residual2 = add(norm1, ffn);

  // 16. Layer Normalization 2 (n x 8)
  const { normalized: norm2, stats: norm2Stats } = computeLayerNormWithStats(residual2);

  // 17. Encoder Stack (3 repeated layers)
  const encoderStack: Matrix[] = [norm2];
  let currentLayerRep = norm2;
  for (let l = 1; l < 3; l++) {
    const selfAttn = matmul(
      softmax(matmul(currentLayerRep, transpose(currentLayerRep)).map((row) => row.map((z) => round(z / Math.sqrt(D))))),
      currentLayerRep
    );
    const { normalized: midNorm } = computeLayerNormWithStats(add(currentLayerRep, selfAttn));
    const hidden = matmul(midNorm, matrix(D, FFN_EXPANSION, 150 + l * 13, 0.3)).map((row) => row.map(gelu));
    const out = matmul(hidden, matrix(FFN_EXPANSION, D, 170 + l * 17, 0.28));
    const { normalized: layerOut } = computeLayerNormWithStats(add(midNorm, out));
    currentLayerRep = layerOut;
    encoderStack.push(currentLayerRep);
  }
  const encoderFinal = encoderStack[encoderStack.length - 1];

  // 18. Decoder Masked Self-Attention (causal triangular mask)
  const decoderMasked = softmax(
    matmul(encoderFinal, transpose(encoderFinal)).map((row) =>
      row.map((z) => round(z / Math.sqrt(D)))
    ),
    true
  );
  const maskedContext = matmul(decoderMasked, encoderFinal);

  // 19. Decoder Cross-Attention (Q from decoder, K/V from encoder)
  const crossScores = matmul(maskedContext, transpose(encoderFinal)).map((row) =>
    row.map((z) => round(z / Math.sqrt(D)))
  );
  const crossAttentionWeights = softmax(crossScores);
  const crossAttention = matmul(crossAttentionWeights, encoderFinal);

  // Decoder FFN and output
  const { normalized: decoderAfterCross } = computeLayerNormWithStats(add(maskedContext, crossAttention));
  const decFfnHidden = matmul(decoderAfterCross, matrix(D, FFN_EXPANSION, 211, 0.28)).map((row) => row.map(gelu));
  const decFfnOut = matmul(decFfnHidden, matrix(FFN_EXPANSION, D, 223, 0.26));
  const { normalized: decoder } = computeLayerNormWithStats(add(decoderAfterCross, decFfnOut));

  // 20. Output Vocabulary Projection, Logits, and Probabilities
  const vocabulary = ["tired", "hungry", "stopped", "slept", "ran", "slowly", "away", "home"];
  const wVocab = matrix(D, vocabulary.length, 303, 0.6);
  // Take representation of the last token
  const lastTokenRep = [decoder[n - 1] ?? Array.from({ length: D }, () => 0.1)];
  const logits = matmul(lastTokenRep, wVocab);
  const probabilities = softmax(logits);

  return {
    tokens,
    ids,
    embedding,
    position,
    x,
    wq,
    wk,
    wv,
    q,
    k,
    v,
    scores,
    scaled,
    attention,
    av,
    heads,
    headDescriptions,
    concat,
    wo,
    projected,
    residual1,
    norm1,
    norm1Stats,
    w1,
    b1,
    ffnHidden,
    w2,
    b2,
    ffn,
    residual2,
    norm2,
    norm2Stats,
    encoderStack,
    decoderMasked,
    crossAttention,
    decoder,
    wVocab,
    logits,
    probabilities,
    vocabulary,
    coreferenceInfo,
  };
}

export const stageData = [
  ["01 Tokenization", "Input text is divided into tokens; each token is assigned a deterministic illustrative Token ID."],
  ["02 Embedding Lookup", "Token IDs are mapped to dense learned semantic embedding vectors E in a D=8 dimensional space."],
  ["03 Positional Encoding", "Sinusoidal positional encoding matrix P is computed to inject sequence order information."],
  ["04 Input Matrix X", "Token embeddings and positional encodings are summed elementwise: X = E + P."],
  ["05 Q, K, V Projections", "Input X is multiplied by weight matrices WQ, WK, WV (8 × 4) to yield Query, Key, and Value matrices (n × 4)."],
  ["06 Attention Scores QKᵀ", "Pairwise dot-product compatibility matrix QKᵀ (n × n) evaluates match strength between every query and key."],
  ["07 Scale by √dₖ", "Raw dot-product scores are scaled by √dₖ (√4 = 2) to control gradient magnitudes before exponentiation."],
  ["08 Softmax Weights", "Row-wise softmax normalizes scaled scores into attention probabilities; each query row sums to 1.00."],
  ["09 Attention Heatmap", "Interactive 2D attention heatmap displaying token-to-token connections, including coreference resolution ('it' → 'animal')."],
  ["10 Weighted Value Aggregation", "Attention weights multiply Value matrix V: Z = AV, aggregating context into a new contextual representation."],
  ["11 Multi-Head Attention", "Four parallel attention heads learn distinct relational patterns, which are concatenated and projected by WO."],
  ["12 Residual Connection (Add)", "Original input X is added to the projected attention output via skip connection to preserve gradient flow."],
  ["13 Layer Normalization", "Normalizes activation features across each token row: (x - μ) / √(σ² + ε) with learnable scale γ and shift β."],
  ["14 Feed-Forward Network (FFN)", "Two-layer position-wise MLP (8 → 16 → 8) with GELU activation processes each token independently."],
  ["15 Second Add & Norm", "Second residual skip connection followed by Layer Normalization completes the standard Transformer Encoder layer."],
  ["16 Encoder Stack", "Multiple encoder layers (N=3 in this demo) are stacked sequentially to build deep contextual representations."],
  ["17 Decoder Masked Attention", "Causal attention masks future target positions with -∞ to ensure autoregressive causality."],
  ["18 Decoder Cross-Attention", "Decoder queries attend to encoder output keys and values, transferring source context to target generation."],
  ["19 Vocabulary Logits", "Linear projection maps final token representation to raw unnormalized scores across vocabulary candidates."],
  ["20 Softmax & Token Selection", "Softmax converts logits into normalized probabilities; greedy argmax selects the top candidate."],
  ["21 Autoregressive Loop", "Selected token is appended to the sequence and fed back into the model to generate the next token."],
] as const;

export const tensorMatrix = (simulation: Simulation, name: TensorName): Matrix => {
  switch (name) {
    case "Tokens":
    case "E":
      return simulation.embedding;
    case "P":
      return simulation.position;
    case "X":
      return simulation.x;
    case "Q":
      return simulation.q;
    case "K":
      return simulation.k;
    case "V":
      return simulation.v;
    case "Scores":
      return simulation.scores;
    case "Scaled":
      return simulation.scaled;
    case "Attention":
    case "Heatmap":
      return simulation.attention;
    case "AV":
      return simulation.av;
    case "Heads":
      return simulation.heads[0];
    case "Concat":
      return simulation.concat;
    case "Projected":
      return simulation.projected;
    case "Residual1":
      return simulation.residual1;
    case "Norm1":
      return simulation.norm1;
    case "FFN":
      return simulation.ffn;
    case "Residual2":
      return simulation.residual2;
    case "Norm2":
      return simulation.norm2;
    case "EncoderStack":
      return simulation.encoderStack[simulation.encoderStack.length - 1];
    case "DecoderMasked":
      return simulation.decoderMasked;
    case "CrossAttention":
      return simulation.crossAttention;
    case "Decoder":
      return simulation.decoder;
    case "Logits":
      return simulation.logits;
    case "Probabilities":
    case "Autoregressive":
      return simulation.probabilities;
    default:
      return simulation.x;
  }
};
