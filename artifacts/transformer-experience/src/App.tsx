import { type CSSProperties, useEffect, useState } from "react";
import TransformerLab from "./components/TransformerLab";

const concepts = [
  [
    "Why Transformers were introduced",
    "Overcoming the fundamental bottlenecks of sequential NLP models.",
    "Introduced in the seminal 2017 paper “Attention Is All You Need” by Vaswani et al., the Transformer discarded recurrence entirely in favor of self-attention. This unlocked massive parallelization across GPUs and TPUs, allowing models to train on unprecedented scales of data.",
    "intro",
    "Traditional models could only train sequentially token-by-token. Transformers process all input tokens concurrently in matrix operations, dramatically accelerating training efficiency.",
  ],
  [
    "Problems with RNN, LSTM & GRU",
    "Sequential bottlenecks, catastrophic forgetting, and vanishing gradients.",
    "Recurrent architectures (RNN, LSTM, GRU) pass a hidden state vector sequentially from left to right. This creates an $O(N)$ sequential dependency that cannot be parallelized. Furthermore, information from earlier tokens decays or vanishes over long distances, making long-range dependencies difficult to capture.",
    "rnn",
    "Backpropagation through time (BPTT) over long sequences causes gradients to either vanish to zero or explode. Information had to squeeze through a fixed-size vector bottleneck at every time step.",
  ],
  [
    "The Attention Mechanism",
    "Content-based dynamic retrieval replaces static information bottlenecks.",
    "Instead of compressing an entire sequence into a single static hidden vector, attention allows a model to look back at all prior hidden states and assign higher weights to the most relevant tokens dynamically.",
    "attn-concept",
    "Attention acts like a soft, differentiable database lookup where queries retrieve content from keys and aggregate corresponding values.",
  ],
  [
    "Tokenization & Token IDs",
    "Text is segmented into distinct tokens and mapped to numerical IDs.",
    "Computers cannot process raw strings directly. Tokenizers (such as Byte-Pair Encoding or WordPiece) split text into subword units, punctuation, and words, converting each into a unique integer vocabulary ID.",
    "token",
    "Token boundaries depend on frequency statistics learned during pre-training. Common words form single tokens, while rare words are broken into subwords.",
  ],
  [
    "Token Embeddings",
    "Each discrete token ID becomes a dense learned continuous vector.",
    "An embedding matrix $E \\in \\mathbb{R}^{|V| \\times d_{model}}$ acts as a lookup table mapping each integer token ID to a continuous vector representing learned semantic and syntactic features.",
    "embed",
    "Tokens appearing in similar contextual distributions develop similar vector geometries in the high-dimensional embedding space.",
  ],
  [
    "Positional Information",
    "Injecting sequence order into permutation-equivariant operations.",
    "Self-attention treats input tokens as an unordered set (bag of tokens). To distinguish “dog bites man” from “man bites dog”, fixed sinusoidal encodings or learned positional vectors $P$ are added elementwise to token embeddings: $X = E + P$.",
    "position",
    "The 2017 paper uses sinusoidal waves of varying frequencies: $PE_{(pos, 2i)} = \\sin(pos / 10000^{2i/d})$ and $PE_{(pos, 2i+1)} = \\cos(pos / 10000^{2i/d})$, allowing the model to easily learn relative positions.",
  ],
  [
    "Query, Key & Value (Q, K, V)",
    "Three distinct learned projections for every token position.",
    "Each token vector is projected into three subspaces: Query (what this token searches for), Key (what this token offers to match against queries), and Value (the contextual payload that will be gathered).",
    "qkv",
    "Projections are computed as linear matrix multiplications: $Q = X W_Q$, $K = X W_K$, and $V = X W_V$, where $W_Q, W_K, W_V \\in \\mathbb{R}^{d_{model} \\times d_k}$.",
  ],
  [
    "Scaled Dot-Product Self-Attention",
    "Pairwise token compatibility scoring with gradient stabilization.",
    "Every query row is dotted with every key row: $S = Q K^T$. The scores are divided by $\\sqrt{d_k}$ to prevent dot products from growing excessively large, and softmax produces attention weights summing to 1.00: $A = \\text{softmax}(Q K^T / \\sqrt{d_k}) V$.",
    "attention",
    "When $d_k$ is large, dot products grow large in magnitude, pushing softmax into regions with extremely small gradients. Scaling by $1/\\sqrt{d_k}$ preserves stable gradient flow.",
  ],
  [
    "Multi-Head Attention",
    "Parallel attention heads specialize in distinct linguistic relationships.",
    "Rather than computing a single attention distribution, Multi-Head Attention projects $Q, K, V$ into $h$ distinct lower-dimensional subspaces in parallel. One head may focus on immediate syntax, another on coreference pronouns, and another on subject-verb agreements.",
    "heads",
    "The $h$ independent head outputs are concatenated and multiplied by an output projection matrix $W_O \\in \\mathbb{R}^{h \\cdot d_v \\times d_{model}}$ to synthesize the multi-view representations.",
  ],
  [
    "Add & Norm (Residual Skip Connections)",
    "Gradient highways and activation stabilization across deep layers.",
    "The output of each sublayer is added back to its input via a residual skip connection: $x + \\text{Sublayer}(x)$, followed by Layer Normalization: $\\text{LayerNorm}(x + \\text{Sublayer}(x))$.",
    "norm",
    "Residual connections allow gradients to backpropagate directly through dozens of layers without attenuating, preventing vanishing gradients and enabling deep architectures.",
  ],
  [
    "Position-Wise Feed-Forward Network (FFN)",
    "Non-linear per-token transformations applied identically in parallel.",
    "While self-attention mixes representations across different token positions, the FFN transforms each token vector individually: $\\text{FFN}(x) = W_2 \\sigma(W_1 x + b_1) + b_2$. The inner hidden layer expands the dimension (typically $4 \\times d_{model}$).",
    "ffn",
    "In modern variants, non-linear activations like GELU or SwiGLU have largely superseded ReLU for smoother gradient propagation.",
  ],
  [
    "The Encoder Stack",
    "Multiple repeated blocks synthesize rich bidirectional representations.",
    "The Transformer Encoder is composed of a stack of $N$ identical layers (e.g. $N=6$ in original Transformer, 32+ in modern models). Each layer contains Multi-Head Self-Attention and an FFN, each preceded or followed by Add & Norm.",
    "encoder",
    "Because encoder attention is unmasked (bidirectional), every token representation in the upper layers incorporates rich contextual awareness of all preceding and succeeding tokens.",
  ],
  [
    "The Decoder Stack",
    "Autoregressive target generation with masked self-attention.",
    "The Decoder generates output tokens from left to right. In addition to the encoder's sublayers, each decoder layer contains Masked Multi-Head Self-Attention and Cross-Attention to the encoder outputs.",
    "decoder",
    "In decoder-only models (like GPT and modern generative LLMs), the cross-attention sublayer is omitted, leaving causally masked self-attention and FFN sublayers.",
  ],
  [
    "Masked (Causal) Attention",
    "Enforcing the autoregressive arrow of time during training and inference.",
    "During text generation, position $i$ must not attend to future tokens $j > i$. Causal attention sets future attention logits to $-\\infty$ prior to softmax: $\\text{softmax}((Q K^T / \\sqrt{d_k}) + M)$, which zeros out future weights.",
    "masked",
    "This causal mask allows training an entire sequence in parallel using teacher forcing while strictly ensuring the model cannot cheat by looking ahead.",
  ],
  [
    "Cross-Attention",
    "Connecting the decoder's queries to the encoder's keys and values.",
    "In encoder-decoder models (like translation models and T5), cross-attention takes Queries $Q$ from the previous decoder layer, while Keys $K$ and Values $V$ come from the output of the encoder stack.",
    "cross",
    "This allows the decoder to dynamically query the full source sequence context at every step of generating the target translation or summary.",
  ],
  [
    "Output Prediction & Autoregressive Generation",
    "From contextual vectors to vocabulary probabilities and sequential generation.",
    "The final decoder representation is mapped by a linear projection $W_{vocab}$ to unnormalized logits across the vocabulary. Softmax produces a probability distribution: $p(w) = \\text{softmax}(\\text{logits})$. The chosen token is appended to the prompt, and the loop repeats.",
    "output",
    "Sampling strategies such as greedy argmax, temperature scaling, top-k, and nucleus (top-p) sampling control the diversity and coherence of generated text.",
  ],
  [
    "Modern Transformer Variants",
    "Encoder-only, Decoder-only, and Encoder-Decoder architectures.",
    "The original 2017 Transformer was an Encoder-Decoder designed for machine translation. Today: BERT is Encoder-only (ideal for classification and embeddings); GPT, LLaMA, and Claude are Decoder-only (ideal for generative text); and T5/BART are Encoder-Decoder.",
    "variants",
    "Decoder-only architectures have become the dominant paradigm for frontier Large Language Models due to computational simplicity and unified next-token prediction objectives.",
  ],
];

function ChapterVisual({ kind }: { kind: string }) {
  if (kind === "intro") {
    return (
      <div style={{ display: "grid", gap: "10px", textAlign: "center" }}>
        <div style={{ display: "flex", gap: "8px", justifyContent: "center" }}>
          {["All", "Tokens", "Processed", "In Parallel"].map((t) => (
            <span
              key={t}
              style={{
                padding: "8px 12px",
                borderRadius: "8px",
                background: "#fdf4e8",
                border: "1px solid #dcb58e",
                color: "#7e522c",
                fontSize: "11px",
                fontFamily: "var(--app-font-mono)",
              }}
            >
              {t}
            </span>
          ))}
        </div>
        <div className="diagram-label">Full sequence parallel matrix operations (GPU accelerated)</div>
      </div>
    );
  }

  if (kind === "rnn") {
    return (
      <div style={{ textAlign: "center" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "6px" }}>
          {["h₀", "h₁", "h₂", "h₃"].map((h, i) => (
            <div key={h} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <span
                style={{
                  width: "36px",
                  height: "36px",
                  borderRadius: "50%",
                  border: "1px solid #c97f48",
                  background: "#fff9f0",
                  display: "grid",
                  placeItems: "center",
                  fontSize: "11px",
                  color: "#834f2b",
                  fontFamily: "var(--app-font-mono)",
                  opacity: 0.4 + i * 0.2,
                }}
              >
                {h}
              </span>
              {i < 3 && <b style={{ color: "#bd7042", fontSize: "16px" }}>→</b>}
            </div>
          ))}
        </div>
        <div className="diagram-label">Sequential dependency: hₜ requires hₜ₋₁ (sequential bottleneck)</div>
      </div>
    );
  }

  if (kind === "attn-concept") {
    return (
      <div className="mini-network">
        <svg viewBox="0 0 320 200" aria-hidden="true">
          <path d="M30 60 Q160 10 290 60" fill="none" stroke="#bb8056" strokeWidth="2.5" opacity=".8" />
          <path d="M30 60 Q160 190 290 140" fill="none" stroke="#bb8056" strokeWidth="1.5" opacity=".4" />
          <path d="M30 140 Q160 20 290 60" fill="none" stroke="#bb8056" strokeWidth="1.5" opacity=".4" />
          <path d="M30 140 Q160 180 290 140" fill="none" stroke="#bb8056" strokeWidth="2.5" opacity=".8" />
        </svg>
        <div className="network-nodes">
          <span>Query</span>
          <span>Key</span>
        </div>
        <div className="diagram-label" style={{ marginTop: "170px" }}>Dynamic similarity weighting</div>
      </div>
    );
  }

  if (kind === "token") {
    return (
      <div>
        <div className="diagram-token">
          <span>Trans</span>
          <span>form</span>
          <span>er</span>
          <span>!</span>
        </div>
        <div className="diagram-label">Subword token decomposition</div>
      </div>
    );
  }

  if (kind === "embed") {
    return (
      <div className="diagram-vector">
        <span className="diagram-token">
          <span>token ID 142</span>
        </span>
        <b className="vector-arrow">→</b>
        <span className="dot-cloud">
          {Array.from({ length: 20 }, (_, i) => (
            <i key={i} style={{ opacity: 0.35 + (i % 5) * 0.13 }} />
          ))}
        </span>
        <span className="diagram-label">d_model continuous vector</span>
      </div>
    );
  }

  if (kind === "position") {
    return (
      <div>
        <div className="pos-list">
          {["01", "02", "03", "04", "05"].map((x) => (
            <span key={x}>{x}</span>
          ))}
        </div>
        <div className="diagram-label">Sinusoidal position signal + Token embedding = Input X</div>
      </div>
    );
  }

  if (kind === "qkv") {
    return (
      <div className="branch">
        <span>Query (Q) · What this token searches for</span>
        <span>Key (K) · What this token offers to match</span>
        <span>Value (V) · Information payload to gather</span>
      </div>
    );
  }

  if (kind === "attention") {
    return (
      <div className="mini-network">
        <svg viewBox="0 0 320 200" aria-hidden="true">
          <path d="M25 55 Q155 5 294 65 M25 55 Q150 180 294 145 M25 145 Q150 30 294 65 M25 145 Q155 190 294 145" fill="none" stroke="#bb8056" strokeWidth="2" opacity=".7" />
        </svg>
        <div className="network-nodes">
          <span>it</span>
          <span>animal</span>
        </div>
        <div className="diagram-label" style={{ marginTop: "170px" }}>softmax(QKᵀ / √dₖ) V</div>
      </div>
    );
  }

  if (kind === "heads") {
    return (
      <div className="mini-heads">
        {["syntax", "coreference", "verb-obj", "theme"].map((x, i) => (
          <span className="mini-head" style={{ "--tilt": `${(i - 1.5) * 6}deg` } as CSSProperties} key={x}>
            {x}
          </span>
        ))}
      </div>
    );
  }

  if (kind === "norm") {
    return (
      <svg className="residual-svg" viewBox="0 0 300 160" role="img" aria-label="Residual path adds the original input to a transformed output">
        <path d="M30 80 H116 Q150 80 150 45 H263" fill="none" stroke="#b96f42" strokeWidth="3" />
        <path d="M30 80 Q30 138 150 138 Q263 138 263 80" fill="none" stroke="#859e8b" strokeWidth="3" />
        <circle cx="30" cy="80" r="8" fill="#c77c4c" />
        <circle cx="263" cy="80" r="11" fill="#829a86" />
        <text x="88" y="67" fill="#80694f" fontSize="11">Sublayer</text>
        <text x="117" y="154" fill="#6a806f" fontSize="11">Residual skip shortcut</text>
      </svg>
    );
  }

  if (kind === "ffn") {
    return (
      <div className="ffn">
        <i />
        <i />
        <i />
        <i />
        <i />
      </div>
    );
  }

  if (kind === "encoder") {
    return (
      <div className="encoder-blocks">
        <span>Self-Attention</span>
        <span>Add & Norm</span>
        <span>Feed-Forward</span>
        <span>Add & Norm</span>
      </div>
    );
  }

  if (kind === "decoder") {
    return (
      <div className="mask-grid">
        {Array.from({ length: 25 }, (_, i) => (
          <i className={i % 5 > Math.floor(i / 5) ? "masked" : ""} key={i} />
        ))}
      </div>
    );
  }

  if (kind === "masked") {
    return (
      <div>
        <div className="mask-grid">
          {Array.from({ length: 16 }, (_, i) => (
            <i className={i % 4 > Math.floor(i / 4) ? "masked" : ""} key={i} />
          ))}
        </div>
        <div className="diagram-label">Upper triangle masked with -∞ (0% attention)</div>
      </div>
    );
  }

  if (kind === "cross") {
    return (
      <div className="cross-net">
        <span>
          <i>Q (dec)</i>
        </span>
        <svg viewBox="0 0 290 160">
          <path d="M50 80 C 140 30, 160 30, 240 45" stroke="#bd7042" strokeWidth="2" fill="none" />
          <path d="M50 80 C 140 130, 160 130, 240 115" stroke="#bd7042" strokeWidth="2" fill="none" />
        </svg>
        <span>
          <i>K (enc)</i>
          <i>V (enc)</i>
        </span>
      </div>
    );
  }

  if (kind === "output") {
    return (
      <div className="output-bars">
        <i style={{ "--h": "78%" } as CSSProperties} data-token="tired" />
        <i style={{ "--h": "45%" } as CSSProperties} data-token="hungry" />
        <i style={{ "--h": "25%" } as CSSProperties} data-token="stopped" />
        <i style={{ "--h": "12%" } as CSSProperties} data-token="slept" />
      </div>
    );
  }

  if (kind === "variants") {
    return (
      <div style={{ display: "grid", gap: "8px", width: "160px" }}>
        <div style={{ padding: "8px", background: "#fbf6ec", border: "1px solid #d9c4ad", borderRadius: "6px", textAlign: "center", fontSize: "11px", color: "#6e4b2d" }}>
          <strong>BERT</strong> · Encoder-only
        </div>
        <div style={{ padding: "8px", background: "#fbf6ec", border: "1px solid #d9c4ad", borderRadius: "6px", textAlign: "center", fontSize: "11px", color: "#6e4b2d" }}>
          <strong>GPT / LLaMA</strong> · Decoder-only
        </div>
        <div style={{ padding: "8px", background: "#fbf6ec", border: "1px solid #d9c4ad", borderRadius: "6px", textAlign: "center", fontSize: "11px", color: "#6e4b2d" }}>
          <strong>T5 / BART</strong> · Encoder-Decoder
        </div>
      </div>
    );
  }

  return (
    <div className="output-bars">
      <i style={{ "--h": "70%" } as CSSProperties} data-token="token" />
    </div>
  );
}

function App() {
  const [generated, setGenerated] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const responseTokens = [
    "Photosynthesis",
    " is",
    " the",
    " biochemical",
    " process",
    " by",
    " which",
    " plants",
    " convert",
    " sunlight",
    " into",
    " cellular",
    " energy.",
  ];

  useEffect(() => {
    if (!isGenerating) return undefined;
    if (generated >= responseTokens.length) {
      setIsGenerating(false);
      return undefined;
    }
    const timer = window.setTimeout(() => setGenerated((n) => n + 1), 400);
    return () => window.clearTimeout(timer);
  }, [isGenerating, generated, responseTokens.length]);

  return (
    <main className="site">
      <div className="topline" />
      <header className="masthead">
        <a href="#top" className="brand" aria-label="Transformer, back to top">
          fieldnotes / AI
        </a>
        <nav className="nav" aria-label="Page sections">
          <a href="#model">01 Model</a>
          <a href="#understanding">02 Architectural Concepts</a>
          <a href="#example">03 Real World</a>
          <a href="#presentation">04 Presentation PDF</a>
        </nav>
      </header>

      {/* Hero Section */}
      <section className="hero" id="top">
        <div className="hero-orbit" aria-hidden="true">
          <svg viewBox="0 0 1200 800">
            <path
              d="M70 440 C250 220 370 620 570 395 S900 180 1130 405"
              fill="none"
              stroke="#d89d6d"
              strokeOpacity=".28"
              strokeWidth="1.5"
            />
            <path
              d="M90 480 C270 260 400 670 590 425 S930 210 1110 435"
              fill="none"
              stroke="#9eb4a2"
              strokeOpacity=".26"
              strokeWidth="1.5"
            />
            <circle cx="600" cy="400" r="5" fill="#d39a69" fillOpacity=".6" />
            <circle cx="240" cy="325" r="4" fill="#8fa995" fillOpacity=".6" />
            <circle cx="958" cy="315" r="4" fill="#d39a69" fillOpacity=".6" />
          </svg>
        </div>
        <div className="hero-inner reveal">
          <div className="eyebrow">A guided architectural learning experience</div>
          <h1>
            TRANS<br />FORMER
          </h1>
          <p className="hero-subtitle">
            A sentence goes in. Context comes alive.<br />
            Step inside the architecture that transformed machine learning.
          </p>
        </div>
        <a className="hero-scroll" href="#model">
          SCROLL TO EXPLORE <span />
        </a>
      </section>

      {/* Interactive Transformer Lab */}
      <TransformerLab />

      {/* Editorial Explanation Section */}
      <section className="section chapter-section" id="understanding">
        <div className="chapter-heading">
          <div>
            <div className="kicker">02 / The architectural foundations</div>
            <h2>
              Understanding<br />the Transformer.
            </h2>
          </div>
          <p className="section-intro">
            From the sequential limitations of RNNs to multi-head self-attention and autoregressive generation: explore the sixteen essential components that power the modern Transformer.
          </p>
        </div>

        {concepts.map(([title, summary, detail, kind, techNote], index) => (
          <article className={`chapter ${index % 2 ? "reverse" : ""}`} key={title}>
            <div className="chapter-art">
              <ChapterVisual kind={kind} />
            </div>
            <div className="chapter-text">
              <span className="chapter-no">
                {String(index + 1).padStart(2, "0")} / ARCHITECTURAL FOUNDATION
              </span>
              <h3>{title}</h3>
              <p>
                {summary} {detail}
              </p>
              <details>
                <summary>TECHNICAL NOTE</summary>
                <p>{techNote}</p>
              </details>
            </div>
          </article>
        ))}
      </section>

      {/* Real-World Demonstration Section */}
      <section className="section real-section" id="example">
        <div className="real-inner">
          <div className="real-intro">
            <div>
              <div className="kicker">03 / From prompt to generation</div>
              <h2>See it in the real world.</h2>
            </div>
            <p>
              Modern conversational LLMs (like GPT, Claude, and LLaMA) use a decoder-only Transformer. Given a prompt and all previously generated tokens, the model samples one next token at a time autoregressively.
            </p>
          </div>

          <div className="chat-demo">
            <div className="prompt-card">
              <span className="tag">Prompt input</span>
              <blockquote>“Explain photosynthesis in simple words.”</blockquote>
              <div className="flow-list">
                {[
                  "Prompt tokenization into subword IDs",
                  "Token embeddings + positional encoding",
                  "Deep decoder Transformer layers",
                  "Next-token logit prediction via softmax",
                  "Append chosen token and repeat",
                ].map((x, i) => (
                  <div className="flow-step" key={x}>
                    <b>0{i + 1}</b>
                    {x}
                  </div>
                ))}
              </div>
            </div>

            <div className="response-card">
              <div className="response-top">
                <span>Generated response stream</span>
                <span aria-live="polite">
                  {generated} / {responseTokens.length} tokens
                </span>
              </div>
              <p className="response-copy" aria-live="polite">
                {responseTokens.slice(0, generated).map((token, i) => (
                  <span className="generated-token" key={`${i}-${token}`}>
                    {token}
                  </span>
                ))}
                {generated === 0 && (
                  <span style={{ color: "#9a8b78" }}>Click Generate response to begin generation.</span>
                )}
              </p>

              <div className="control-row">
                <button
                  type="button"
                  className="btn primary"
                  onClick={() => {
                    if (generated >= responseTokens.length) setGenerated(0);
                    setIsGenerating(true);
                  }}
                  disabled={isGenerating}
                  data-testid="button-generate"
                >
                  {generated >= responseTokens.length
                    ? "Generate again"
                    : isGenerating
                    ? "Generating…"
                    : "Generate response"}
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setIsGenerating(false);
                    setGenerated(0);
                  }}
                >
                  Reset response
                </button>
              </div>

              <div style={{ marginTop: 24 }}>
                <div className="prob-title">Next-token candidate distribution (Softmax)</div>
                {[
                  ["biochemical", 64],
                  ["vital", 21],
                  ["natural", 10],
                  ["biological", 5],
                ].map(([word, amount]) => (
                  <div className="prob-row" key={word}>
                    <span>{word}</span>
                    <span className="prob-bar">
                      <i style={{ width: `${amount}%` }} />
                    </span>
                    <span>{amount}%</span>
                  </div>
                ))}
                <div className="prob-note">
                  Illustrative probability distribution sampled from simulated vocabulary logits.
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* PDF Presentation Download Section */}
      <section className="pdf-section" id="presentation">
        <div className="pdf-card">
          <div className="pdf-preview" aria-label="Presentation cover preview">
            <div className="pdf-sheet">
              <small>FIELDNOTES / LEARNING SERIES</small>
              <strong>
                Transformer<br />Networks
              </strong>
              <span className="pdf-line" />
              <span className="pdf-line short" />
              <small>ATTENTION · CONTEXT · GENERATION</small>
            </div>
          </div>
          <div className="pdf-copy">
            <div className="kicker">Keep exploring, at your own pace</div>
            <h2>Download the Transformer Presentation</h2>
            <p>
              Revisit the complete Transformer Networks presentation offline. The official 15-page presentation slide deck is bundled directly with this project as a PDF document.
            </p>
            <a
              className="btn primary"
              href={`${import.meta.env.BASE_URL}Transformer_Networks.pdf`}
              download="Transformer_Networks.pdf"
              data-testid="link-download-pdf"
            >
              Download the Transformer Presentation (PDF) <span aria-hidden="true">↗</span>
            </a>
          </div>
        </div>
      </section>

      {/* Closing Ribbon */}
      <section className="closing">
        <div className="kicker">Summary</div>
        <h2>
          That’s the<br />Transformer.
        </h2>
        <div className="flow-ribbon">
          <span>Tokens</span>
          <b>→</b>
          <span>Embeddings & Position</span>
          <b>→</b>
          <span>Q / K / V Projections</span>
          <b>→</b>
          <span>Scaled Dot-Product</span>
          <b>→</b>
          <span>Multi-Head Attention</span>
          <b>→</b>
          <span>Add & Norm</span>
          <b>→</b>
          <span>Feed-Forward</span>
          <b>→</b>
          <span>Softmax Generation</span>
        </div>
        <p>
          What begins as a discrete sentence of words transforms into dynamic high-dimensional representations, pairwise attention weights, and next-token probability distributions.
        </p>
        <a className="btn" href="#top">
          Return to top ↑
        </a>
      </section>

      <footer className="footer">
        <span>A guided interactive exhibit on Transformer neural network architectures</span>
        <a href="#presentation">Download Presentation (PDF)</a>
      </footer>
    </main>
  );
}

export default App;
