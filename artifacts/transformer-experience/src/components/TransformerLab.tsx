import { useEffect, useMemo, useState, type CSSProperties } from "react";
import {
  simulateTransformer,
  stageData,
  tensorMatrix,
  type Matrix,
  type TensorName,
  PROJECTION,
  D,
  FFN_EXPANSION,
  HEADS,
} from "../lib/transformer-simulation";

const examples = [
  "The animal didn't cross the road because it was tired",
  "The cat sat on the mat and fell asleep",
  "I love machine learning and neural networks",
  "Transformers understand context through attention",
];

const tensorByStage: TensorName[] = [
  "Tokens",
  "E",
  "P",
  "X",
  "Q",
  "Scores",
  "Scaled",
  "Attention",
  "Heatmap",
  "AV",
  "Heads",
  "Residual1",
  "Norm1",
  "FFN",
  "Norm2",
  "EncoderStack",
  "DecoderMasked",
  "CrossAttention",
  "Logits",
  "Probabilities",
  "Autoregressive",
];

const fmt = (x: number) => (Object.is(x, -0) ? "0.0000" : x.toFixed(4));
const pct = (x: number) => `${Math.round(x * 100)}%`;
const dims = (m: Matrix) => `${m.length} × ${m[0]?.length ?? 0}`;

function MatrixView({
  matrix,
  rowLabels,
  colLabels,
  label,
  selectedCell,
  onCell,
  testPrefix,
  isMasked = false,
}: {
  matrix: Matrix;
  rowLabels: string[];
  colLabels?: string[];
  label: string;
  selectedCell: [number, number];
  onCell: (r: number, c: number) => void;
  testPrefix: string;
  isMasked?: boolean;
}) {
  const isWide = (matrix[0]?.length ?? 0) > 6;
  const numCols = matrix[0]?.length ?? 0;

  return (
    <div className="lab-matrix-scroll">
      <table
        className={`lab-matrix ${isWide ? "wide" : ""}`}
        aria-label={`${label} matrix, dimensions ${dims(matrix)}`}
      >
        <thead>
          <tr>
            <th scope="col">{label} · {dims(matrix)}</th>
            {Array.from({ length: numCols }, (_, j) => (
              <th scope="col" key={j}>
                {colLabels?.[j] ? colLabels[j] : `d${j + 1}`}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th scope="row">
                <span className="matrix-row-token" title={rowLabels[i] ?? `row ${i + 1}`}>
                  {rowLabels[i] ?? `row ${i + 1}`}
                </span>
              </th>
              {row.map((value, j) => {
                const isCellMasked = isMasked && j > i;
                const isSelected = selectedCell[0] === i && selectedCell[1] === j;
                const strength = Math.min(0.48, Math.abs(value) * 0.35 + 0.08);

                return (
                  <td key={j}>
                    <button
                      type="button"
                      className={`matrix-cell ${isSelected ? "selected" : ""} ${isCellMasked ? "masked" : ""}`}
                      onClick={() => onCell(i, j)}
                      aria-label={`${label} row ${i + 1} (${rowLabels[i]}), column ${j + 1}, value ${isCellMasked ? "masked (-∞)" : fmt(value)}`}
                      aria-pressed={isSelected}
                      data-testid={`${testPrefix}-cell-${i}-${j}`}
                      style={{ "--cell-strength": `${strength}` } as CSSProperties}
                      disabled={isCellMasked}
                    >
                      {isCellMasked ? "-∞" : fmt(value)}
                    </button>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HeatmapView({
  matrix,
  tokens,
  selectedCell,
  onCell,
  coreferenceInfo,
}: {
  matrix: Matrix;
  tokens: string[];
  selectedCell: [number, number];
  onCell: (r: number, c: number) => void;
  coreferenceInfo?: {
    queryIndex: number;
    keyIndex: number;
    queryToken: string;
    keyToken: string;
    weight: number;
    description: string;
  };
}) {
  return (
    <div className="heatmap-container" style={{ margin: "16px 0" }}>
      <div className="heatmap-header" style={{ marginBottom: "12px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <span className="lab-eyebrow">Interactive Attention Heatmap</span>
          <p style={{ margin: "4px 0 0", fontSize: "12px", color: "#776553" }}>
            Rows represent <strong>Query</strong> tokens; columns represent <strong>Key</strong> tokens. Darker amber indicates stronger attention.
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "11px", color: "#8a755e" }}>
          <span>0.00</span>
          <div style={{ width: "80px", height: "10px", borderRadius: "5px", background: "linear-gradient(90deg, #fbf7ee, #d88048)" }} />
          <span>1.00</span>
        </div>
      </div>

      {coreferenceInfo && (
        <div
          style={{
            margin: "0 0 16px",
            padding: "12px 16px",
            background: "#fdf3e7",
            border: "1px solid #ecc9a4",
            borderRadius: "10px",
            fontSize: "12px",
            color: "#6b4728",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: "10px",
          }}
        >
          <div>
            <strong>Coreference Highlight: </strong>
            <span>{coreferenceInfo.description}</span>
          </div>
          <button
            type="button"
            className="btn"
            style={{ fontSize: "11px", padding: "6px 12px" }}
            onClick={() => onCell(coreferenceInfo.queryIndex, coreferenceInfo.keyIndex)}
          >
            Inspect "{coreferenceInfo.queryToken}" → "{coreferenceInfo.keyToken}"
          </button>
        </div>
      )}

      <div style={{ overflowX: "auto", border: "1px solid #e5d9ca", borderRadius: "10px", padding: "12px", background: "#fffdf9" }}>
        <table style={{ borderCollapse: "separate", borderSpacing: "3px", width: "100%", minWidth: "420px" }}>
          <thead>
            <tr>
              <th style={{ fontSize: "10px", textAlign: "left", color: "#9a6544", padding: "6px" }}>
                Query \ Key
              </th>
              {tokens.map((token, j) => (
                <th key={j} style={{ fontSize: "10px", color: "#795f46", padding: "6px", textAlign: "center" }}>
                  {token}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <th scope="row" style={{ fontSize: "10px", color: "#544331", textAlign: "left", padding: "6px", whiteSpace: "nowrap" }}>
                  {tokens[i]}
                </th>
                {row.map((val, j) => {
                  const isSelected = selectedCell[0] === i && selectedCell[1] === j;
                  const isCoreference =
                    coreferenceInfo &&
                    coreferenceInfo.queryIndex === i &&
                    coreferenceInfo.keyIndex === j;
                  const intensity = Math.min(1, Math.max(0.04, val));

                  return (
                    <td key={j} style={{ padding: 0 }}>
                      <button
                        type="button"
                        onClick={() => onCell(i, j)}
                        title={`Query "${tokens[i]}" attends to Key "${tokens[j]}": ${(val * 100).toFixed(1)}%`}
                        style={{
                          width: "100%",
                          minWidth: "38px",
                          height: "36px",
                          borderRadius: "6px",
                          border: isCoreference ? "2px solid #b55325" : isSelected ? "2px solid #4a3625" : "1px solid rgba(200, 150, 100, 0.15)",
                          backgroundColor: `rgba(202, 110, 50, ${intensity})`,
                          color: intensity > 0.45 ? "#ffffff" : "#453627",
                          fontWeight: isSelected || isCoreference ? "600" : "400",
                          fontSize: "10px",
                          cursor: "pointer",
                          transition: "transform 0.15s ease",
                        }}
                      >
                        {fmt(val)}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function TransformerLab() {
  const [input, setInput] = useState(examples[0]);
  const [stage, setStage] = useState(-1);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [mode, setMode] = useState<"conceptual" | "technical">("technical");
  const [tokenIndex, setTokenIndex] = useState(0);
  const [cell, setCell] = useState<[number, number]>([0, 0]);
  const [qkvTab, setQkvTab] = useState<"Q" | "K" | "V">("Q");
  const [head, setHead] = useState(0);
  const [encoderLayer, setEncoderLayer] = useState(0);
  const [probabilityView, setProbabilityView] = useState<"probabilities" | "logits">("probabilities");
  const [generation, setGeneration] = useState<string[]>([]);

  const computedInput = generation.length ? `${input} ${generation.join(" ")}` : input;
  const simulation = useMemo(() => simulateTransformer(computedInput), [computedInput]);

  const activeStage = stage < 0 ? 0 : stage;
  const tensor = tensorByStage[activeStage] ?? "X";

  // Matrix selection based on stage and active sub-view
  const currentMatrix = useMemo((): Matrix => {
    if (activeStage === 0) return simulation.embedding;
    if (activeStage === 1) return simulation.embedding;
    if (activeStage === 2) return simulation.position;
    if (activeStage === 3) return simulation.x;
    if (activeStage === 4) {
      if (qkvTab === "Q") return simulation.q;
      if (qkvTab === "K") return simulation.k;
      return simulation.v;
    }
    if (activeStage === 5) return simulation.scores;
    if (activeStage === 6) return simulation.scaled;
    if (activeStage === 7 || activeStage === 8) return simulation.attention;
    if (activeStage === 9) return simulation.av;
    if (activeStage === 10) return simulation.heads[head] ?? simulation.heads[0];
    if (activeStage === 11) return simulation.residual1;
    if (activeStage === 12) return simulation.norm1;
    if (activeStage === 13) return simulation.ffn;
    if (activeStage === 14) return simulation.norm2;
    if (activeStage === 15) return simulation.encoderStack[encoderLayer] ?? simulation.encoderStack[0];
    if (activeStage === 16) return simulation.decoderMasked;
    if (activeStage === 17) return simulation.crossAttention;
    if (activeStage === 18) return simulation.logits;
    if (activeStage === 19 || activeStage === 20) {
      return probabilityView === "logits" ? simulation.logits : simulation.probabilities;
    }
    return tensorMatrix(simulation, tensor);
  }, [activeStage, qkvTab, head, encoderLayer, probabilityView, simulation, tensor]);

  const selectedRow = Math.min(cell[0], Math.max(0, currentMatrix.length - 1));
  const selectedCol = Math.min(cell[1], Math.max(0, (currentMatrix[0]?.length ?? 1) - 1));
  const selectedValue = currentMatrix[selectedRow]?.[selectedCol] ?? 0;

  const rawScore = simulation.scores[selectedRow]?.[selectedCol] ?? 0;
  const scaledScore = simulation.scaled[selectedRow]?.[selectedCol] ?? 0;
  const attentionWeight = simulation.attention[selectedRow]?.[selectedCol] ?? 0;

  useEffect(() => {
    if (!running) return undefined;
    const timer = window.setTimeout(() => {
      if (stage >= stageData.length - 1) {
        setRunning(false);
      } else {
        setStage((s) => s + 1);
      }
    }, 1600 / speed);
    return () => window.clearTimeout(timer);
  }, [running, stage, speed]);

  const run = () => {
    setGeneration([]);
    setStage(0);
    setRunning(true);
  };

  const reset = () => {
    setRunning(false);
    setStage(-1);
    setTokenIndex(0);
    setCell([0, 0]);
    setGeneration([]);
  };

  const changeInput = (value: string) => {
    setRunning(false);
    setInput(value);
    setStage(-1);
    setTokenIndex(0);
    setCell([0, 0]);
    setGeneration([]);
  };

  const advanceGeneration = () => {
    const maxIndex = simulation.probabilities[0].indexOf(Math.max(...simulation.probabilities[0]));
    const nextWord = simulation.vocabulary[maxIndex];
    if (nextWord && generation.length < 5) {
      setGeneration((prev) => [...prev, nextWord]);
    }
  };

  const isAttentionStage =
    activeStage === 5 || activeStage === 6 || activeStage === 7 || activeStage === 8 || activeStage === 16;

  return (
    <section className="section model-section" id="model">
      <div className="section-head">
        <div className="kicker">01 / The working model</div>
        <h2>SEE THE TRANSFORMER WORK</h2>
        <p className="section-intro">
          Follow a sentence step by step through every computational stage of the Transformer architecture.
          Every matrix multiplication, normalization, attention score, and probability distribution is computed live.
        </p>
      </div>

      <div className="model-shell transformer-lab" data-testid="transformer-lab">
        <div className="lab-topline">
          <span className="lab-eyebrow">Interactive Transformer Computation Bench</span>
          <span className="illustrative-stamp">
            Deterministic educational computation · D={D}, d_k={PROJECTION}, {HEADS} heads
          </span>
        </div>

        {/* Input & Mode Controls */}
        <div className="lab-controls">
          <label className="lab-input-wrap">
            <span className="field-label">Input sentence</span>
            <input
              className="text-field"
              aria-label="Sentence to process"
              value={input}
              maxLength={120}
              onChange={(e) => changeInput(e.target.value)}
              data-testid="input-transformer-sentence"
            />
          </label>

          <label className="lab-example-wrap">
            <span className="field-label">Load an example sentence</span>
            <select
              value={examples.includes(input) ? input : ""}
              onChange={(e) => e.target.value && changeInput(e.target.value)}
              aria-label="Choose example sentence"
              data-testid="select-transformer-example"
            >
              <option value="">Custom sentence</option>
              {examples.map((example) => (
                <option key={example} value={example}>
                  {example}
                </option>
              ))}
            </select>
          </label>

          <div className="mode-switch" role="group" aria-label="Visualization mode">
            <button
              type="button"
              onClick={() => setMode("conceptual")}
              aria-pressed={mode === "conceptual"}
              data-testid="button-mode-conceptual"
            >
              Conceptual Mode
            </button>
            <button
              type="button"
              onClick={() => setMode("technical")}
              aria-pressed={mode === "technical"}
              data-testid="button-mode-technical"
            >
              Technical Mode
            </button>
          </div>
        </div>

        {/* Playback Controls */}
        <div className="lab-run-controls">
          <div className="control-row">
            <button
              type="button"
              className="btn primary"
              onClick={run}
              data-testid="button-run-transformer"
            >
              Run Transformer
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => {
                setRunning(false);
                setStage((s) => Math.min(stageData.length - 1, s + 1));
              }}
              data-testid="button-step"
            >
              Step
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => setRunning((v) => !v)}
              disabled={stage < 0 || stage >= stageData.length - 1}
              aria-pressed={running}
              data-testid="button-pause"
            >
              {running ? "Pause" : "Resume"}
            </button>
            <button type="button" className="btn" onClick={reset} data-testid="button-reset">
              Reset
            </button>
          </div>

          <div className="speed" role="group" aria-label="Timeline speed">
            {[
              ["Slow", 0.6],
              ["Normal", 1],
              ["Fast", 1.8],
            ].map(([name, value]) => (
              <button
                type="button"
                key={name as string}
                aria-pressed={speed === value}
                onClick={() => setSpeed(value as number)}
                data-testid={`button-speed-${String(name).toLowerCase()}`}
              >
                {name as string}
              </button>
            ))}
          </div>
        </div>

        {/* Status & Progress */}
        <div className="model-status" aria-live="polite" data-testid="status-transformer-stage">
          <span>
            {stage < 0
              ? "Ready · select a stage or click Run Transformer"
              : `Stage ${String(stage + 1).padStart(2, "0")} of ${stageData.length} · ${stageData[stage][0]}`}
          </span>
          <span>
            {simulation.tokens.length} tokens · d_model={D} · {HEADS} heads · d_k={PROJECTION}
          </span>
        </div>

        <div className="progress-track" aria-label="Computation progress">
          <div
            className="progress-fill"
            style={{ width: `${stage < 0 ? 0 : ((stage + 1) / stageData.length) * 100}%` }}
          />
        </div>

        {/* 21-Stage Timeline */}
        <div className="lab-timeline" role="group" aria-label="Select computation stage">
          {stageData.map(([name], i) => (
            <button
              key={name}
              type="button"
              className={`stage-pill ${stage === i ? "active" : ""} ${stage > i ? "done" : ""}`}
              onClick={() => {
                setRunning(false);
                setStage(i);
              }}
              aria-current={stage === i ? "step" : undefined}
              data-testid={`button-stage-${i}`}
            >
              {name}
            </button>
          ))}
        </div>

        {/* Main Content Area */}
        <div className="lab-content">
          <main className="lab-main">
            <div className="lab-stage-heading">
              <div className="stage-index">
                {stage < 0
                  ? "READY / INPUT"
                  : `STAGE ${String(stage + 1).padStart(2, "0")} / ${stageData[activeStage][0].toUpperCase()}`}
              </div>
              <h3>{stage < 0 ? "Start with a sentence." : stageData[activeStage][0]}</h3>
              <p>
                {stage < 0
                  ? "Select any checkpoint in the timeline or click Run to step through Tokenization, Embedding, Q/K/V, Scaled Softmax Attention, Multi-Head projection, Add & Norm, Feed-Forward, and Autoregressive Output."
                  : stageData[activeStage][1]}
              </p>
            </div>

            {/* CONCEPTUAL MODE */}
            {mode === "conceptual" ? (
              <div className="conceptual-experience" style={{ display: "grid", gap: "20px" }}>
                <div
                  style={{
                    padding: "20px",
                    background: "#fbf6ec",
                    border: "1px solid #e7dac9",
                    borderRadius: "14px",
                  }}
                >
                  <span className="lab-eyebrow">High-Level Flow of the Transformer</span>
                  <div className="conceptual-flow" style={{ margin: "16px 0", minHeight: "120px" }}>
                    <span>1. Input Text</span>
                    <b>→</b>
                    <span>2. Tokenization & Position</span>
                    <b>→</b>
                    <span>3. Multi-Head Attention</span>
                    <b>→</b>
                    <span>4. Add & Norm</span>
                    <b>→</b>
                    <span>5. Feed-Forward</span>
                    <b>→</b>
                    <span>6. Output Prediction</span>
                  </div>
                  <p style={{ margin: 0, fontSize: "13px", color: "#6e5d4c", lineHeight: 1.6 }}>
                    Unlike Recurrent Neural Networks (RNNs) that process text sequentially one token at a time, the Transformer ingests all tokens simultaneously. Self-attention determines how much each word should attend to every other word, creating deep contextual representations in parallel.
                  </p>
                </div>

                {/* Token Connectivity Graph */}
                <div
                  style={{
                    padding: "20px",
                    background: "#fffdf9",
                    border: "1px solid #e7dac9",
                    borderRadius: "14px",
                  }}
                >
                  <span className="lab-eyebrow">Token Relationship Network</span>
                  <h4 style={{ margin: "6px 0 12px", fontFamily: "var(--app-font-serif)", fontSize: "20px" }}>
                    How Attention Connects Tokens
                  </h4>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", margin: "14px 0" }}>
                    {simulation.tokens.map((token, i) => (
                      <button
                        key={`${token}-${i}`}
                        type="button"
                        onClick={() => {
                          setTokenIndex(i);
                          setCell([i, cell[1]]);
                        }}
                        style={{
                          padding: "8px 14px",
                          borderRadius: "999px",
                          border: tokenIndex === i ? "2px solid #b55325" : "1px solid #dccbb9",
                          background: tokenIndex === i ? "#fae8d4" : "#fff9f0",
                          color: "#4e3b2b",
                          fontSize: "12px",
                          cursor: "pointer",
                        }}
                      >
                        <strong>{token}</strong> (pos {i})
                      </button>
                    ))}
                  </div>

                  {simulation.coreferenceInfo && (
                    <div
                      style={{
                        padding: "14px",
                        background: "#fcf4e8",
                        borderLeft: "3px solid #b55325",
                        borderRadius: "0 8px 8px 0",
                        fontSize: "12px",
                        color: "#6b492b",
                        lineHeight: 1.6,
                      }}
                    >
                      <strong>Coreference Example: </strong>
                      In the sentence <em>"{input}"</em>, the pronoun <strong>"{simulation.coreferenceInfo.queryToken}"</strong> receives high attention weight ({(simulation.coreferenceInfo.weight * 100).toFixed(1)}%) from <strong>"{simulation.coreferenceInfo.keyToken}"</strong>. This allows the model to understand that "{simulation.coreferenceInfo.queryToken}" refers to the "{simulation.coreferenceInfo.keyToken}".
                    </div>
                  )}

                  <div style={{ marginTop: "16px", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "12px" }}>
                    {simulation.headDescriptions.map((desc, i) => (
                      <div
                        key={desc}
                        style={{
                          padding: "12px",
                          background: "#f9f4ec",
                          border: "1px solid #e5d8c8",
                          borderRadius: "8px",
                          fontSize: "11px",
                          color: "#6e5d4c",
                        }}
                      >
                        <strong style={{ color: "#9c572f", display: "block", marginBottom: "4px" }}>
                          Head {i + 1}
                        </strong>
                        {desc}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              /* TECHNICAL MODE */
              <>
                {/* Token Strips */}
                <div className="token-strip" aria-label="Illustrative token IDs">
                  {simulation.tokens.map((word, i) => (
                    <button
                      type="button"
                      key={`${word}-${i}`}
                      onClick={() => {
                        setTokenIndex(i);
                        setCell([i, Math.min(cell[1], simulation.tokens.length - 1)]);
                      }}
                      className={tokenIndex === i ? "selected" : ""}
                      aria-pressed={tokenIndex === i}
                      data-testid={`button-token-row-${i}`}
                    >
                      <strong>{word}</strong>
                      <small>ID {simulation.ids[i]} · pos {i}</small>
                    </button>
                  ))}
                </div>

                {/* Stage 0: Tokenization Specific View */}
                {activeStage === 0 && (
                  <div style={{ margin: "16px 0", padding: "18px", background: "#fffdf8", border: "1px solid #e7ded1", borderRadius: "12px" }}>
                    <span className="lab-eyebrow">Input Tokenization Breakdown</span>
                    <p style={{ margin: "6px 0 14px", fontSize: "13px", color: "#6a5745" }}>
                      The text is split into {simulation.tokens.length} discrete tokens. Each token is assigned an illustrative vocabulary ID.
                    </p>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))", gap: "10px" }}>
                      {simulation.tokens.map((token, i) => (
                        <div
                          key={i}
                          style={{
                            padding: "10px 12px",
                            border: "1px solid #ded0be",
                            borderRadius: "8px",
                            background: tokenIndex === i ? "#fae7d2" : "#fdf9f2",
                          }}
                        >
                          <div style={{ fontSize: "14px", fontWeight: "600", color: "#3d3023" }}>"{token}"</div>
                          <div style={{ fontSize: "10px", color: "#8c6b4d", fontFamily: "var(--app-font-mono)", marginTop: "4px" }}>
                            Token ID: {simulation.ids[i]}
                          </div>
                          <div style={{ fontSize: "9px", color: "#8a7562", marginTop: "2px" }}>Position index: {i}</div>
                        </div>
                      ))}
                    </div>
                    <small style={{ display: "block", marginTop: "12px", color: "#927e69", fontSize: "10px" }}>
                      Educational / illustrative values — deterministic stand-in vocabulary IDs for demonstration purposes.
                    </small>
                  </div>
                )}

                {/* Stage 4: Q/K/V Tabs */}
                {activeStage === 4 && (
                  <div style={{ margin: "14px 0" }}>
                    <div style={{ display: "flex", gap: "8px", marginBottom: "12px" }}>
                      {(["Q", "K", "V"] as const).map((tab) => (
                        <button
                          key={tab}
                          type="button"
                          className="btn"
                          style={{
                            backgroundColor: qkvTab === tab ? "#ad5b31" : "#fffdf8",
                            color: qkvTab === tab ? "#ffffff" : "#544432",
                            borderColor: qkvTab === tab ? "#ad5b31" : "#d8c7b4",
                          }}
                          onClick={() => setQkvTab(tab)}
                        >
                          {tab === "Q" && "Query (Q = XWQ)"}
                          {tab === "K" && "Key (K = XWK)"}
                          {tab === "V" && "Value (V = XWV)"}
                        </button>
                      ))}
                    </div>

                    <div style={{ padding: "12px", background: "#fbf6ec", border: "1px solid #e7ded1", borderRadius: "8px", fontSize: "11px", color: "#745f4b", marginBottom: "12px" }}>
                      <strong>Dimension check: </strong>
                      Input X ({simulation.tokens.length} × {D}) multiplied by W{qkvTab} ({D} × {PROJECTION}) produces {qkvTab} ({simulation.tokens.length} × {PROJECTION}).
                    </div>
                  </div>
                )}

                {/* Stage 8: Dedicated Heatmap */}
                {activeStage === 8 ? (
                  <HeatmapView
                    matrix={simulation.attention}
                    tokens={simulation.tokens}
                    selectedCell={cell}
                    onCell={(r, c) => setCell([r, c])}
                    coreferenceInfo={simulation.coreferenceInfo}
                  />
                ) : (
                  <>
                    {/* Matrix View Header */}
                    <div className="tensor-heading">
                      <div>
                        <span className="lab-eyebrow">
                          {isAttentionStage ? "Attention Matrix" : "Active Tensor"}
                        </span>
                        <h4>
                          {activeStage === 4 ? `${qkvTab} Projection Matrix` : tensor}
                          {activeStage === 10 ? ` · Head ${head + 1} slice` : ""}
                        </h4>
                      </div>
                      <span className="shape-chip" data-testid="text-current-shape">
                        Dimensions: {dims(currentMatrix)}
                      </span>
                    </div>

                    {/* Multi-Head Selector for Stage 10 */}
                    {activeStage === 10 && (
                      <div className="head-selector" role="group" aria-label="Select attention head">
                        {simulation.heads.map((_, i) => (
                          <button
                            key={i}
                            type="button"
                            className={head === i ? "active" : ""}
                            onClick={() => setHead(i)}
                            aria-pressed={head === i}
                            data-testid={`button-head-${i + 1}`}
                          >
                            Head {i + 1}
                          </button>
                        ))}
                        <span>{simulation.headDescriptions[head]}</span>
                      </div>
                    )}

                    {/* Encoder Stack Layer Selector for Stage 15 */}
                    {activeStage === 15 && (
                      <div className="head-selector" role="group" aria-label="Select encoder layer">
                        {simulation.encoderStack.map((_, l) => (
                          <button
                            key={l}
                            type="button"
                            className={encoderLayer === l ? "active" : ""}
                            onClick={() => setEncoderLayer(l)}
                            aria-pressed={encoderLayer === l}
                          >
                            Encoder Layer {l + 1}
                          </button>
                        ))}
                        <span>Stack of {simulation.encoderStack.length} identical architectural blocks</span>
                      </div>
                    )}

                    {/* Logits vs Probabilities Toggle */}
                    {(activeStage === 18 || activeStage === 19 || activeStage === 20) && (
                      <div className="logit-toggle" role="group" aria-label="Logits or probabilities" style={{ marginBottom: "12px" }}>
                        <button
                          type="button"
                          aria-pressed={probabilityView === "logits"}
                          onClick={() => setProbabilityView("logits")}
                          data-testid="button-view-logits"
                        >
                          Logits (Raw scores)
                        </button>
                        <button
                          type="button"
                          aria-pressed={probabilityView === "probabilities"}
                          onClick={() => setProbabilityView("probabilities")}
                          data-testid="button-view-probabilities"
                        >
                          Probabilities (Softmax normalized)
                        </button>
                      </div>
                    )}

                    {/* Main Matrix Grid */}
                    <MatrixView
                      matrix={currentMatrix}
                      rowLabels={
                        activeStage === 18 || activeStage === 19 || activeStage === 20
                          ? ["last token"]
                          : simulation.tokens
                      }
                      colLabels={
                        activeStage === 18 || activeStage === 19 || activeStage === 20
                          ? simulation.vocabulary
                          : isAttentionStage
                          ? simulation.tokens
                          : undefined
                      }
                      label={
                        activeStage === 4
                          ? qkvTab
                          : activeStage === 18 || activeStage === 19
                          ? probabilityView
                          : tensor
                      }
                      selectedCell={cell}
                      onCell={(r, c) => setCell([r, c])}
                      testPrefix="tensor"
                      isMasked={activeStage === 16}
                    />
                  </>
                )}

                {/* LayerNorm Breakdown for Stage 12 & 14 */}
                {(activeStage === 12 || activeStage === 14) && (
                  <div style={{ marginTop: "16px", padding: "14px", background: "#fbf6ec", border: "1px solid #e7ded1", borderRadius: "10px" }}>
                    <span className="lab-eyebrow">Layer Normalization Step-by-Step Breakdown</span>
                    <div style={{ fontSize: "11px", color: "#6e5d4c", marginTop: "6px" }}>
                      Row {selectedRow + 1} ("{simulation.tokens[selectedRow]}"):
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "10px", marginTop: "8px" }}>
                        <div><strong>Mean μ: </strong>{fmt(simulation.norm1Stats[selectedRow]?.mean ?? 0)}</div>
                        <div><strong>Variance σ²: </strong>{fmt(simulation.norm1Stats[selectedRow]?.variance ?? 0)}</div>
                        <div><strong>Std Dev √(σ²+ε): </strong>{fmt(simulation.norm1Stats[selectedRow]?.std ?? 0)}</div>
                        <div><strong>Formula: </strong>y = γ · (x - μ)/σ + β</div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Softmax Row Sum Validation */}
                {isAttentionStage && (
                  <div className="attention-bars">
                    <span className="lab-eyebrow">
                      Row Attention Weights · Query: "{simulation.tokens[selectedRow]}"
                    </span>
                    {simulation.tokens.map((word, i) => {
                      const w = simulation.attention[selectedRow]?.[i] ?? 0;
                      return (
                        <div className="attention-bar-row" key={`${word}-${i}`}>
                          <span>{word}</span>
                          <i>
                            <b style={{ width: `${Math.min(100, w * 100)}%` }} />
                          </i>
                          <strong>{fmt(w)} ({pct(w)})</strong>
                        </div>
                      );
                    })}
                    <small>
                      Row Sum = {fmt(simulation.attention[selectedRow]?.reduce((a, b) => a + b, 0) ?? 0)} (Normalized to 1.00)
                    </small>
                  </div>
                )}

                {/* Dynamic Mathematical Equation Ribbon */}
                <div className="equation-ribbon">
                  {activeStage === 0 && "Tokens: T = [t₁, t₂, ... tₙ] → IDs: [id₁, id₂, ... idₙ]"}
                  {activeStage === 1 && "E = EmbeddingLookup(IDs) ∈ ℝ^(n × 8)"}
                  {activeStage === 2 && "P_{(pos, 2i)} = sin(pos / 10000^(2i/d)), P_{(pos, 2i+1)} = cos(pos / 10000^(2i/d))"}
                  {activeStage === 3 && "X = E + P ∈ ℝ^(n × 8)"}
                  {activeStage === 4 && `${qkvTab} = X W_${qkvTab} ∈ ℝ^(n × 4)`}
                  {activeStage === 5 && "S = Q Kᵀ ∈ ℝ^(n × n)"}
                  {activeStage === 6 && "S_scaled = (Q Kᵀ) / √dₖ = (Q Kᵀ) / 2 ∈ ℝ^(n × n)"}
                  {activeStage === 7 || activeStage === 8 && "A = softmax(S_scaled) ∈ ℝ^(n × n), ∑_j A_ij = 1.00"}
                  {activeStage === 9 && "Z = A V ∈ ℝ^(n × 4)"}
                  {activeStage === 10 && "MultiHead(X) = Concat(head₁, head₂, head₃, head₄) W_O ∈ ℝ^(n × 8)"}
                  {activeStage === 11 && "Residual₁ = X + Attention(X) ∈ ℝ^(n × 8)"}
                  {activeStage === 12 && "Norm₁ = LayerNorm(X + Attention(X)) = γ(x - μ)/√(σ² + ε) + β"}
                  {activeStage === 13 && "FFN(x) = W₂ · GELU(W₁ x + b₁) + b₂ ∈ ℝ^(n × 8)"}
                  {activeStage === 14 && "LayerOutput = LayerNorm(Norm₁ + FFN(Norm₁)) ∈ ℝ^(n × 8)"}
                  {activeStage === 15 && "EncoderStack = EncoderLayer₃(EncoderLayer₂(EncoderLayer₁(X)))"}
                  {activeStage === 16 && "MaskedSelfAttention(Q, K, V) = softmax((QKᵀ / √dₖ) + M)V, where M_ij = -∞ for j > i"}
                  {activeStage === 17 && "CrossAttention = softmax(Q_dec K_encᵀ / √dₖ) V_enc"}
                  {activeStage === 18 && "Logits = decoder_final · W_vocab ∈ ℝ^(1 × |V|)"}
                  {activeStage === 19 || activeStage === 20 && "p(w) = softmax(Logits) = exp(z_i) / ∑ exp(z_j)"}
                </div>

                <p className="technical-explanation">
                  Every matrix cell and dimension is computed live and deterministically. Token IDs, weights, and embeddings are illustrative educational stand-ins to visualize the mathematical mechanics of the Transformer architecture.
                </p>
              </>
            )}

            {/* Autoregressive Next-Token Generation Panel */}
            {activeStage >= 18 && (
              <div className="generation-panel">
                <div>
                  <span className="lab-eyebrow">Autoregressive Generation</span>
                  <p>{computedInput}</p>
                  <small>
                    The predicted token is selected via greedy argmax, appended to the sequence, and the Transformer re-computes attention over the lengthened context.
                  </small>
                </div>
                <button
                  type="button"
                  className="btn primary"
                  onClick={advanceGeneration}
                  disabled={generation.length >= 5}
                  data-testid="button-append-next-token"
                >
                  {generation.length >= 5
                    ? "Max demo tokens reached"
                    : `Append next token · "${simulation.vocabulary[simulation.probabilities[0].indexOf(Math.max(...simulation.probabilities[0]))]}"`}
                </button>
              </div>
            )}
          </main>

          {/* Inspector Sidebar */}
          {mode === "technical" && (
            <aside className="lab-inspector">
              <div className="inspector-card">
                <span className="lab-eyebrow">Tensor & Cell Inspector</span>
                <h4>Cell Detail</h4>
                <div className="inspector-token-pair">
                  <b>{simulation.tokens[selectedRow] ?? "query"}</b>
                  <span>→</span>
                  <b>
                    {isAttentionStage
                      ? simulation.tokens[selectedCol] ?? "key"
                      : `col ${selectedCol + 1}`}
                  </b>
                </div>
                <dl>
                  <div>
                    <dt>Selected cell value</dt>
                    <dd data-testid="text-inspector-value">{fmt(selectedValue)}</dd>
                  </div>
                  <div>
                    <dt>Raw score (QKᵀ)</dt>
                    <dd>{fmt(rawScore)}</dd>
                  </div>
                  <div>
                    <dt>Scaled score (/√dₖ)</dt>
                    <dd>{fmt(scaledScore)}</dd>
                  </div>
                  <div>
                    <dt>Attention weight</dt>
                    <dd>{fmt(attentionWeight)}</dd>
                  </div>
                  <div>
                    <dt>Active tensor shape</dt>
                    <dd>{dims(currentMatrix)}</dd>
                  </div>
                </dl>
              </div>

              {/* Dimension Tracker */}
              <div className="inspector-card dimension-card">
                <span className="lab-eyebrow">Dimension Tracker</span>
                <h4>Matrix Shapes</h4>
                {[
                  ["X (Input)", simulation.x],
                  ["WQ, WK, WV", simulation.wq],
                  ["Q, K, V", simulation.q],
                  ["QKᵀ (Scores)", simulation.scores],
                  ["A (Softmax)", simulation.attention],
                  ["AV (Weighted)", simulation.av],
                  ["Concat Heads", simulation.concat],
                  ["WO (Projection)", simulation.wo],
                  ["FFN Output", simulation.ffn],
                  ["Encoder Output", simulation.norm2],
                ].map(([name, val]) => (
                  <div
                    className={`dimension-row ${tensor === name ? "current" : ""}`}
                    key={name as string}
                  >
                    <span>{name as string}</span>
                    <b>{dims(val as Matrix)}</b>
                  </div>
                ))}
                <small>
                  n = {simulation.tokens.length} · d_model = {D} · d_k = {PROJECTION} · d_v = {PROJECTION}
                </small>
              </div>

              {/* Architecture Note */}
              <div className="inspector-card model-distinction">
                <span className="lab-eyebrow">Architectural Role</span>
                <h4>Encoder vs. Decoder</h4>
                <p>
                  <b>Encoder:</b> Bidirectional self-attention allowing all input tokens to attend to each other without masks.
                </p>
                <p>
                  <b>Decoder:</b> Employs causal masking to prevent future token lookahead, plus cross-attention to attend back to encoder outputs.
                </p>
              </div>
            </aside>
          )}
        </div>

        {/* Complete Computation Details */}
        <div className="lab-bottom-controls">
          <button
            type="button"
            className="btn"
            onClick={() => {
              const details = document.getElementById("complete-computation");
              if (details instanceof HTMLDetailsElement) details.open = true;
              details?.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }}
            data-testid="button-show-complete"
          >
            Show complete computation pipeline
          </button>
          <span>Tab / Space / Enter keyboard accessible · Full screen responsive</span>
        </div>

        <details className="complete-computation" id="complete-computation">
          <summary data-testid="button-complete-details">
            Complete Mathematical Pipeline · Encoder & Decoder Paths
          </summary>
          <div className="complete-columns">
            <div>
              <span className="lab-eyebrow">Encoder Pipeline</span>
              <ol>
                <li>Text is tokenized into n tokens and assigned Token IDs</li>
                <li>Embedding lookup E (n × 8) + sinusoidal positional encoding P (n × 8) = Input matrix X (n × 8)</li>
                <li>Linear projections: Q = XWQ, K = XWK, V = XWV (each n × 4)</li>
                <li>Raw dot products: S = QKᵀ (n × n)</li>
                <li>Scaling: S_scaled = S / √4 = S / 2 (n × n)</li>
                <li>Row-wise Softmax: A = softmax(S_scaled), where each row sums to 1.00</li>
                <li>Weighted aggregation: Z = AV (n × 4)</li>
                <li>Multi-Head: 4 parallel heads concatenated to n × 4, projected via WO (4 × 8) to n × 8</li>
                <li>Residual Add: X + Attention(X) (n × 8)</li>
                <li>LayerNorm: Row-wise mean μ and variance σ² normalization with learnable γ and β</li>
                <li>Position-wise FFN: W₁ (8 × 16) with GELU activation, then W₂ (16 × 8) + b₂</li>
                <li>Second Residual Add & LayerNorm, repeated across N encoder layers</li>
              </ol>
            </div>
            <div>
              <span className="lab-eyebrow">Decoder & Generation Pipeline</span>
              <ol>
                <li>Causal Self-Attention: Future positions masked with -∞ in upper triangle</li>
                <li>Masked Softmax ensures generation cannot peek at future tokens</li>
                <li>Residual connection and Layer Normalization</li>
                <li>Cross-Attention: Queries from decoder attend to Keys and Values from encoder</li>
                <li>Position-wise Feed-Forward Network and Layer Normalization</li>
                <li>Linear vocabulary projection maps last token state (1 × 8) to Logits (1 × |V|)</li>
                <li>Softmax over logits yields probability distribution p(token)</li>
                <li>Greedy or sampled token selection</li>
                <li>Autoregressive recurrence: selected token appended and pipeline recomputed</li>
              </ol>
            </div>
          </div>
        </details>
      </div>
    </section>
  );
}
