import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { simulateTransformer, stageData, tensorMatrix, type Matrix, type TensorName } from "../lib/transformer-simulation";

const examples = ["The animal was tired", "I love machine learning", "The cat sat on the mat", "I like deep learning"];
const tensorByStage: TensorName[] = ["E", "X", "X", "Q", "Scores", "Scaled", "Attention", "AV", "Concat", "Projected", "Norm", "FFN", "Encoder", "Attention", "Decoder", "Logits", "Probabilities"];
const fmt = (x: number) => (Object.is(x, -0) ? "0.0000" : x.toFixed(4));
const dims = (m: Matrix) => `${m.length} × ${m[0]?.length ?? 0}`;

function MatrixView({ matrix, tokens, label, selectedCell, onCell, testPrefix }: { matrix: Matrix; tokens: string[]; label: string; selectedCell: [number, number]; onCell: (r: number, c: number) => void; testPrefix: string }) {
  const isWide = matrix[0]?.length > 6;
  return <div className="lab-matrix-scroll">
    <table className={`lab-matrix ${isWide ? "wide" : ""}`} aria-label={`${label} matrix, dimensions ${dims(matrix)}`}>
      <thead><tr><th scope="col">{label} · {dims(matrix)}</th>{matrix[0]?.map((_, j) => <th scope="col" key={j}>d{j + 1}</th>)}</tr></thead>
      <tbody>{matrix.map((row, i) => <tr key={i}>
        <th scope="row"><span className="matrix-row-token">{tokens[i] ?? `row ${i + 1}`}</span></th>
        {row.map((value, j) => <td key={j}><button type="button" className={`matrix-cell ${selectedCell[0] === i && selectedCell[1] === j ? "selected" : ""}`} onClick={() => onCell(i, j)} aria-label={`${label} row ${i + 1}, column ${j + 1}, value ${fmt(value)}`} aria-pressed={selectedCell[0] === i && selectedCell[1] === j} data-testid={`${testPrefix}-cell-${i}-${j}`} style={{ "--cell-strength": `${Math.min(.42, Math.abs(value) * .25 + .08)}` } as CSSProperties}>{fmt(value)}</button></td>)}
      </tr>)}</tbody>
    </table>
  </div>;
}

export default function TransformerLab() {
  const [input, setInput] = useState(examples[0]);
  const [stage, setStage] = useState(-1);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [mode, setMode] = useState<"conceptual" | "technical">("technical");
  const [tokenIndex, setTokenIndex] = useState(0);
  const [cell, setCell] = useState<[number, number]>([0, 0]);
  const [head, setHead] = useState(0);
  const [probabilityView, setProbabilityView] = useState<"probabilities" | "logits">("probabilities");
  const [generation, setGeneration] = useState<string[]>([]);
  const computedInput = generation.length ? `${input} ${generation.join(" ")}` : input;
  const simulation = useMemo(() => simulateTransformer(computedInput), [computedInput]);
  const activeStage = stage < 0 ? 0 : stage;
  const tensor = tensorByStage[activeStage];
  const selectedMatrix = tensor === "Q" || tensor === "K" || tensor === "V" ? simulation[tensor.toLowerCase() as "q" | "k" | "v"] : tensorMatrix(simulation, tensor);
  const displayMatrix = tensor === "Attention" && activeStage === 13 ? simulation.decoderMasked : selectedMatrix;
  const selectedValue = displayMatrix[Math.min(cell[0], displayMatrix.length - 1)]?.[Math.min(cell[1], (displayMatrix[0]?.length ?? 1) - 1)] ?? 0;
  const isAttentionMatrix = tensor === "Scores" || tensor === "Scaled" || tensor === "Attention" || (activeStage === 13);
  const selectedRow = Math.min(cell[0], simulation.tokens.length - 1);
  const selectedCol = Math.min(cell[1], simulation.tokens.length - 1);
  const rawValue = simulation.scores[selectedRow]?.[selectedCol] ?? 0;
  const scaledValue = simulation.scaled[selectedRow]?.[selectedCol] ?? 0;
  const weightValue = simulation.attention[selectedRow]?.[selectedCol] ?? 0;
  const tokenMatrixName = tensor === "Probabilities" || tensor === "Logits" ? "Vocabulary" : tensor;
  const headMatrix = simulation.heads[head];

  useEffect(() => {
    if (!running) return undefined;
    const timer = window.setTimeout(() => {
      if (stage >= stageData.length - 1) setRunning(false);
      else setStage(stage + 1);
    }, 1500 / speed);
    return () => window.clearTimeout(timer);
  }, [running, stage, speed]);

  const run = () => { setGeneration([]); setStage(0); setRunning(true); };
  const reset = () => { setRunning(false); setStage(-1); setTokenIndex(0); setCell([0, 0]); setGeneration([]); };
  const changeInput = (value: string) => { setRunning(false); setInput(value); setStage(-1); setTokenIndex(0); setCell([0, 0]); setGeneration([]); };
  const advanceGeneration = () => {
    const choice = simulation.vocabulary[simulation.probabilities[0].indexOf(Math.max(...simulation.probabilities[0]))];
    setGeneration((current) => [...current, choice].slice(0, 4));
  };

  return <section className="section model-section" id="model">
    <div className="section-head"><div className="kicker">01 / The working model</div><h2>See the Transformer work.</h2><p className="section-intro">A computation lab for following data through the architecture. Move step by step, then inspect the tensors that make each operation precise.</p></div>
    <div className="model-shell transformer-lab" data-testid="transformer-lab">
      <div className="lab-topline"><span className="lab-eyebrow">Interactive computation lab</span><span className="illustrative-stamp">Illustrative computation · no trained model</span></div>
      <div className="lab-controls">
        <label className="lab-input-wrap"><span className="field-label">Input sentence</span><input className="text-field" aria-label="Sentence to process" value={input} maxLength={100} onChange={(event) => changeInput(event.target.value)} data-testid="input-transformer-sentence" /></label>
        <label className="lab-example-wrap"><span className="field-label">Load an example</span><select value={examples.includes(input) ? input : ""} onChange={(event) => event.target.value && changeInput(event.target.value)} aria-label="Choose example sentence" data-testid="select-transformer-example"><option value="">Custom sentence</option>{examples.map((example) => <option key={example}>{example}</option>)}</select></label>
        <div className="mode-switch" role="group" aria-label="Visualization mode"><button type="button" onClick={() => setMode("conceptual")} aria-pressed={mode === "conceptual"} data-testid="button-mode-conceptual">Conceptual</button><button type="button" onClick={() => setMode("technical")} aria-pressed={mode === "technical"} data-testid="button-mode-technical">Technical</button></div>
      </div>
      <div className="lab-run-controls">
        <div className="control-row">
          <button type="button" className="btn primary" onClick={run} data-testid="button-run-transformer">Run Transformer</button>
          <button type="button" className="btn" onClick={() => { setRunning(false); setStage((s) => Math.min(stageData.length - 1, s + 1)); }} data-testid="button-step">Step</button>
          <button type="button" className="btn" onClick={() => setRunning((v) => !v)} disabled={stage < 0 || stage >= stageData.length - 1} aria-pressed={running} data-testid="button-pause">{running ? "Pause" : "Resume"}</button>
          <button type="button" className="btn" onClick={reset} data-testid="button-reset">Reset</button>
        </div>
        <div className="speed" role="group" aria-label="Timeline speed">{[["Slow", .55], ["Normal", 1], ["Fast", 1.8]].map(([name, value]) => <button type="button" key={name} aria-pressed={speed === value} onClick={() => setSpeed(value as number)} data-testid={`button-speed-${String(name).toLowerCase()}`}>{name}</button>)}</div>
      </div>
      <div className="model-status" aria-live="polite" data-testid="status-transformer-stage"><span>{stage < 0 ? "Ready · select a stage or run the computation" : `Stage ${String(stage + 1).padStart(2, "0")} of ${stageData.length} · ${stageData[stage][0]}`}</span><span>{simulation.tokens.length} tokens · d_model 8 · 4 heads</span></div>
      <div className="progress-track" aria-label="Computation progress"><div className="progress-fill" style={{ width: `${stage < 0 ? 0 : ((stage + 1) / stageData.length) * 100}%` }} /></div>
      <div className="lab-timeline" role="group" aria-label="Select computation stage">{stageData.map(([name], i) => <button key={name} type="button" className={`stage-pill ${stage === i ? "active" : ""} ${stage > i ? "done" : ""}`} onClick={() => { setRunning(false); setStage(i); }} aria-current={stage === i ? "step" : undefined} data-testid={`button-stage-${i}`}>{String(i + 1).padStart(2, "0")} {name}</button>)}</div>
      <div className="lab-content">
        <main className="lab-main">
          <div className="lab-stage-heading"><div className="stage-index">{stage < 0 ? "READY / INPUT TENSOR" : `COMPUTATION ${String(stage + 1).padStart(2, "0")} / ${stageData[activeStage][0].toUpperCase()}`}</div><h3>{stage < 0 ? "Start with a sentence." : stageData[activeStage][0]}</h3><p>{stage < 0 ? "Select any checkpoint in the timeline, or run the deterministic example from tokenization through next-token selection." : stageData[activeStage][1]}</p></div>
          {mode === "conceptual" ? <div className="conceptual-flow" aria-label="Conceptual Transformer flow"><span>Tokens</span><b>→</b><span>Attention</span><b>→</b><span>Transformer layer</span><b>→</b><span>Prediction</span></div> : <>
            <div className="token-strip" aria-label="Illustrative token IDs">{simulation.tokens.map((word, i) => <button type="button" key={`${word}-${i}`} onClick={() => { setTokenIndex(i); setCell([i, Math.min(cell[1], simulation.tokens.length - 1)]); }} className={tokenIndex === i ? "selected" : ""} aria-pressed={tokenIndex === i} data-testid={`button-token-row-${i}`}><strong>{word}</strong><small>ID {simulation.ids[i]} · position {i}</small></button>)}</div>
            <div className="tensor-heading"><div><span className="lab-eyebrow">{isAttentionMatrix ? "Attention score matrix" : "Current tensor"}</span><h4>{tokenMatrixName}{tensor === "Q" || tensor === "K" || tensor === "V" ? ` · head ${head + 1} slice` : ""}</h4></div><span className="shape-chip" data-testid="text-current-shape">{dims(displayMatrix)} · illustrative</span></div>
            {tensor === "Probabilities" || tensor === "Logits" ? <div className="logit-toggle" role="group" aria-label="Logits or probabilities"><button type="button" aria-pressed={probabilityView === "logits"} onClick={() => setProbabilityView("logits")} data-testid="button-view-logits">Logits</button><button type="button" aria-pressed={probabilityView === "probabilities"} onClick={() => setProbabilityView("probabilities")} data-testid="button-view-probabilities">Probabilities</button></div> : null}
            {isAttentionMatrix && <div className="matrix-note">Rows are query positions; columns are key positions. Click any cell to inspect the corresponding score pipeline.</div>}
            {tensor === "Q" || tensor === "K" || tensor === "V" ? <div className={`stream-badge stream-${tensor.toLowerCase()}`}>{tensor} = XW{tensor} · projection stream</div> : null}
            {activeStage === 3 && <details className="weight-inspection"><summary>Inspect projection weights WQ, WK, WV</summary><div className="projection-weights">{[["WQ", simulation.wq], ["WK", simulation.wk], ["WV", simulation.wv]].map(([name, weight]) => <MatrixView key={name as string} matrix={weight as Matrix} tokens={Array.from({ length: 8 }, (_, i) => `row ${i + 1}`)} label={name as string} selectedCell={cell} onCell={setCell} testPrefix={`weight-${String(name).toLowerCase()}`} />)}</div><p>Fixed deterministic educational parameters; each projection is 8 × 8 and not trained weights.</p></details>}
            {activeStage === 1 && <details className="weight-inspection"><summary>Inspect embedding E and positional signal P</summary><div className="projection-weights"><MatrixView matrix={simulation.embedding} tokens={simulation.tokens} label="E" selectedCell={cell} onCell={setCell} testPrefix="embedding" /><MatrixView matrix={simulation.position} tokens={simulation.tokens} label="P" selectedCell={cell} onCell={setCell} testPrefix="position" /></div><p>Input representations are formed elementwise: X = E + P.</p></details>}
            {activeStage === 8 && <div className="head-selector" role="group" aria-label="Select attention head">{simulation.heads.map((_, i) => <button key={i} type="button" className={head === i ? "active" : ""} onClick={() => setHead(i)} aria-pressed={head === i} data-testid={`button-head-${i + 1}`}>Head {i + 1}</button>)}<span>Illustrative relationship pattern</span></div>}
            {activeStage === 8 ? <MatrixView matrix={headMatrix} tokens={simulation.tokens} label={`Head ${head + 1} output`} selectedCell={cell} onCell={setCell} testPrefix="head-output" /> : <MatrixView matrix={tensor === "Logits" && probabilityView === "probabilities" ? simulation.probabilities : tensor === "Probabilities" && probabilityView === "logits" ? simulation.logits : displayMatrix} tokens={tensor === "Logits" || tensor === "Probabilities" ? simulation.vocabulary : simulation.tokens} label={tensor === "Logits" || tensor === "Probabilities" ? probabilityView : tensor} selectedCell={cell} onCell={setCell} testPrefix="tensor" />}
            {isAttentionMatrix && <div className="attention-bars"><span className="lab-eyebrow">Row weights · query: {simulation.tokens[selectedRow]}</span>{simulation.tokens.map((word, i) => <div className="attention-bar-row" key={`${word}-${i}`}><span>{word}</span><i><b style={{ width: `${simulation.attention[selectedRow][i] * 100}%` }} /></i><strong>{fmt(simulation.attention[selectedRow][i])}</strong></div>)}<small>Row sum = {fmt(simulation.attention[selectedRow].reduce((a, b) => a + b, 0))}</small></div>}
            <div className="equation-ribbon">{activeStage === 4 ? "S = QKᵀ" : activeStage === 5 ? "S_scaled = QKᵀ / √dₖ" : activeStage === 6 ? "A = softmax(S_scaled)" : activeStage === 7 ? "Z = AV" : activeStage === 8 || activeStage === 9 ? "MultiHead(X) = Concat(head₁ … head₄)W_O" : activeStage === 10 ? "Y = LayerNorm(X + Attention(X))" : activeStage === 11 ? "FFN(x) = W₂ GELU(W₁x + b₁) + b₂" : activeStage === 15 || activeStage === 16 ? "p(token) = softmax(W_vocab h)" : "X = E + P · illustrative values"}</div>
            <p className="technical-explanation">Every number shown is generated from this same fixed, deterministic educational computation. Token IDs, embeddings, projection weights, scores, and probabilities are illustrative—not values from a trained model.</p>
          </>}
          {activeStage >= 15 && <div className="generation-panel"><div><span className="lab-eyebrow">Autoregressive generation</span><p>{input}{generation.length ? ` ${generation.join(" ")}` : ""}</p><small>One selected token is appended, then the computation can be repeated on the extended context.</small></div><button type="button" className="btn primary" onClick={advanceGeneration} disabled={generation.length >= 4} data-testid="button-append-next-token">{generation.length ? "Append next token" : "Select next token"} · {simulation.vocabulary[simulation.probabilities[0].indexOf(Math.max(...simulation.probabilities[0]))]}</button></div>}
        </main>
        {mode === "technical" && <aside className="lab-inspector">
          <div className="inspector-card"><span className="lab-eyebrow">Matrix / vector inspector</span><h4>Cell detail</h4><div className="inspector-token-pair"><b>{simulation.tokens[selectedRow] ?? "vocabulary"}</b><span>→</span><b>{simulation.tokens[selectedCol] ?? simulation.vocabulary[selectedCol] ?? "column"}</b></div><dl><div><dt>Selected value</dt><dd data-testid="text-inspector-value">{fmt(selectedValue)}</dd></div><div><dt>Raw score</dt><dd>{fmt(rawValue)}</dd></div><div><dt>Scaled score</dt><dd>{fmt(scaledValue)}</dd></div><div><dt>Softmax weight</dt><dd>{fmt(weightValue)}</dd></div><div><dt>Dimensions</dt><dd>{dims(displayMatrix)}</dd></div></dl><p>Illustrative calculation. Cell coordinates follow the visible matrix.</p></div>
          <div className="inspector-card dimension-card"><span className="lab-eyebrow">Dimension tracker</span><h4>Tensor shapes</h4>{[["X", simulation.x], ["Q", simulation.q], ["K", simulation.k], ["V", simulation.v], ["Attention", simulation.attention], ["Output", simulation.encoder]].map(([name, value]) => <div className={`dimension-row ${tensor === name ? "current" : ""}`} key={name as string} data-testid={`dimension-${String(name).toLowerCase()}`}><span>{name as string}</span><b>{dims(value as Matrix)}</b></div>)}<small>n = {simulation.tokens.length} · d_model = 8 · dₖ = 4 · 4 heads</small></div>
          <div className="inspector-card model-distinction"><span className="lab-eyebrow">Architecture note</span><h4>Encoder ≠ decoder</h4><p><b>Original Transformer:</b> encoder builds source context; decoder uses causal self-attention, then cross-attention to encoder outputs.</p><p><b>Decoder-only:</b> causal self-attention only; no separate encoder or encoder–decoder cross-attention.</p></div>
        </aside>}
      </div>
      <div className="lab-bottom-controls"><button type="button" className="btn" onClick={() => { const details = document.getElementById("complete-computation"); if (details instanceof HTMLDetailsElement) details.open = true; details?.scrollIntoView({ behavior: "smooth", block: "nearest" }); }} data-testid="button-show-complete">Show complete computation</button><span>Keyboard: Tab to navigate · Enter/Space to inspect and select stages</span></div>
      <details className="complete-computation" id="complete-computation"><summary data-testid="button-complete-details">Show complete computation · encoder–decoder and decoder-only paths</summary><div className="complete-columns"><div><span className="lab-eyebrow">Encoder path</span><ol>{["Text → tokens → illustrative token IDs", "Embedding lookup E + positional signal P = X (n × 8)", "Q = XWQ · K = XWK · V = XWV (n × 4)", "QKᵀ → divide by √dₖ → row-wise softmax", "Attention weights × V = AV", "Four head outputs → concatenate → Wₒ maps 4 → 8 channels", "Residual add → row-wise LayerNorm", "Position-wise 8→12→8 GELU FFN", "Second residual add → LayerNorm → three encoder layers"].map((x) => <li key={x}>{x}</li>)}</ol></div><div><span className="lab-eyebrow">Original decoder + generation</span><ol>{["Causal mask blocks future target positions", "Masked decoder self-attention → residual + norm", "Cross-attention: Q from decoder, K/V from encoder", "Feed-forward → residual + norm", "Vocabulary linear projection → logits", "Softmax → probabilities → select token", "Append token and recompute on the extended context"].map((x) => <li key={x}>{x}</li>)}</ol><p className="technical-explanation">A decoder-only model omits the encoder and cross-attention path. Both forms generate autoregressively from left to right.</p></div></div></details>
    </div>
  </section>;
}
