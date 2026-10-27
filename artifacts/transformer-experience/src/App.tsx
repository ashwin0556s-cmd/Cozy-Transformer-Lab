import { type CSSProperties, useEffect, useState } from 'react';
import TransformerLab from './components/TransformerLab';

const concepts = [
  ['Tokenization', 'Text is divided into token IDs the network can process.', 'Token boundaries depend on the tokenizer: a token may be a word, part of a word, punctuation, or whitespace.', 'token'],
  ['Embeddings', 'A token ID becomes a learned vector.', 'The embedding table maps each ID to a dense numerical representation. Similar contexts can shape related vectors during training.', 'embed'],
  ['Positional information', 'Order is added to the token representations.', 'Self-attention alone does not encode sequence order. Positional encodings let the model distinguish “small bird” from “bird small.”', 'position'],
  ['Query, Key & Value', 'Each token gets three learned projections.', 'Attention scores compare queries with keys. Those scores weight values to produce a context-aware output.', 'qkv'],
  ['Self-attention', 'Every position can gather relevant context from other positions.', 'Scaled dot-product attention uses softmax(QKᵀ / √dₖ)V. Decoder self-attention is causally masked during generation.', 'attention'],
  ['Multi-head attention', 'Several attention patterns are learned in parallel.', 'Each head has its own projections. Concatenated head outputs are passed through a final linear projection.', 'heads'],
  ['Add & Norm', 'Residual connections preserve a path for information.', 'The sublayer output is added to its input, then normalized. Layer ordering differs in modern Transformer variants.', 'norm'],
  ['Feed-forward network', 'A shared MLP transforms each position independently.', 'Typically two linear transformations with a non-linear activation expand and project the channel dimension.', 'ffn'],
  ['Encoder', 'Encoder layers turn the input into contextual representations.', 'Each layer combines self-attention and a feed-forward network, with residual connections and normalization.', 'encoder'],
  ['Decoder', 'A decoder generates output from left to right.', 'Masked self-attention restricts access to future output positions. Encoder–decoder designs also use cross-attention.', 'decoder'],
  ['Cross-attention', 'The decoder can retrieve relevant information from the encoder.', 'Decoder states supply queries; encoder outputs supply keys and values. GPT-style decoder-only models do not use encoder–decoder cross-attention.', 'cross'],
  ['Output prediction', 'The model scores possible next tokens.', 'A vocabulary projection gives logits; softmax can convert them to a probability distribution, then a decoding strategy chooses a token.', 'output'],
];

function ChapterVisual({ kind }: { kind: string }) {
  if (kind === 'token') return <div><div className="diagram-token"><span>Trans</span><span>form</span><span>er</span><span>!</span></div><div className="diagram-label">one possible token split</div></div>;
  if (kind === 'embed') return <div className="diagram-vector"><span className="diagram-token"><span>word</span></span><b className="vector-arrow">→</b><span className="dot-cloud">{Array.from({ length: 20 }, (_, i) => <i key={i} style={{ opacity: .35 + (i % 5) * .13 }} />)}</span><span className="diagram-label">vector space</span></div>;
  if (kind === 'position') return <div><div className="pos-list">{['01', '02', '03', '04', '05'].map((x) => <span key={x}>{x}</span>)}</div><div className="diagram-label">position signal + token embedding</div></div>;
  if (kind === 'qkv') return <div className="branch"><span>Query · seeks</span><span>Key · matches</span><span>Value · carries</span></div>;
  if (kind === 'attention' || kind === 'cross') return <div className="mini-network"><svg viewBox="0 0 320 200" aria-hidden="true"><path d="M25 55 Q155 5 294 65 M25 55 Q150 180 294 145 M25 145 Q150 30 294 65 M25 145 Q155 190 294 145" fill="none" stroke="#bb8056" strokeWidth="2" opacity=".68" /></svg><div className="network-nodes"><span>it</span><span>animal</span></div></div>;
  if (kind === 'heads') return <div className="mini-heads">{['syntax', 'nearby', 'reference', 'topic'].map((x, i) => <span className="mini-head" style={{ '--tilt': `${(i - 1) * 4}deg` } as CSSProperties} key={x}>{x}</span>)}</div>;
  if (kind === 'norm') return <svg className="residual-svg" viewBox="0 0 300 160" role="img" aria-label="Residual path adds the original input to a transformed output"><path d="M30 80 H116 Q150 80 150 45 H263" fill="none" stroke="#b96f42" strokeWidth="3"/><path d="M30 80 Q30 138 150 138 Q263 138 263 80" fill="none" stroke="#859e8b" strokeWidth="3"/><circle cx="30" cy="80" r="8" fill="#c77c4c"/><circle cx="263" cy="80" r="11" fill="#829a86"/><text x="88" y="67" fill="#80694f" fontSize="11">sublayer</text><text x="117" y="154" fill="#6a806f" fontSize="11">residual shortcut</text></svg>;
  if (kind === 'ffn') return <div className="ffn"><i /><i /><i /><i /><i /></div>;
  if (kind === 'encoder') return <div className="encoder-blocks"><span>Self-attention</span><span>Add & Norm</span><span>Feed-forward</span><span>Add & Norm</span></div>;
  if (kind === 'decoder') return <div className="mask-grid">{Array.from({ length: 25 }, (_, i) => <i className={i % 5 > Math.floor(i / 5) ? 'masked' : ''} key={i} />)}</div>;
  return <div className="output-bars"><i style={{ '--h': '45%' } as CSSProperties} data-token="the" /><i style={{ '--h': '82%' } as CSSProperties} data-token="green" /><i style={{ '--h': '57%' } as CSSProperties} data-token="sun" /><i style={{ '--h': '32%' } as CSSProperties} data-token="leaf" /></div>;
}

function App() {
  const [generated, setGenerated] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const responseTokens = ['Photosynthesis', ' is', ' the', ' process', ' by', ' which', ' plants', ' use', ' sunlight', ' to', ' make', ' food.'];

  useEffect(() => {
    if (!isGenerating) return undefined;
    if (generated >= responseTokens.length) { setIsGenerating(false); return undefined; }
    const timer = window.setTimeout(() => setGenerated((n) => n + 1), 500);
    return () => window.clearTimeout(timer);
  }, [isGenerating, generated, responseTokens.length]);

  return <main className="site">
    <div className="topline" />
    <header className="masthead">
      <a href="#top" className="brand" aria-label="Transformer, back to top">fieldnotes / AI</a>
      <nav className="nav" aria-label="Page sections">
        <a href="#model">01 Model</a><a href="#understanding">02 Explore</a><a href="#example">03 In use</a>
      </nav>
    </header>
    <section className="hero" id="top">
      <div className="hero-orbit" aria-hidden="true"><svg viewBox="0 0 1200 800"><path d="M70 440 C250 220 370 620 570 395 S900 180 1130 405" fill="none" stroke="#d89d6d" strokeOpacity=".23" strokeWidth="1"/><path d="M90 480 C270 260 400 670 590 425 S930 210 1110 435" fill="none" stroke="#9eb4a2" strokeOpacity=".24" strokeWidth="1"/><circle cx="600" cy="400" r="5" fill="#d39a69" fillOpacity=".5"/><circle cx="240" cy="325" r="4" fill="#8fa995" fillOpacity=".6"/><circle cx="958" cy="315" r="4" fill="#d39a69" fillOpacity=".6"/></svg></div>
      <div className="hero-inner reveal"><div className="eyebrow">A guided learning experience</div><h1>TRANS<br />FORMER</h1><p className="hero-subtitle">A sentence goes in. Context comes alive.<br />Step inside the architecture that changed language.</p></div>
      <a className="hero-scroll" href="#model">SCROLL TO EXPLORE <span /></a>
    </section>

    <TransformerLab />

    <section className="section chapter-section" id="understanding">
      <div className="chapter-heading"><div><div className="kicker">02 / The parts, up close</div><h2>Understanding<br />the Transformer.</h2></div><p className="section-intro">The walkthrough showed what happens. Now, meet the pieces—and the job each one does. This exhibit follows the terminology of the accompanying Transformer Networks presentation.</p></div>
      {concepts.map(([title, summary, detail, kind], index) => <article className={`chapter ${index % 2 ? 'reverse' : ''}`} key={title}>
        <div className="chapter-art"><ChapterVisual kind={kind} /></div>
        <div className="chapter-text"><span className="chapter-no">{String(index + 1).padStart(2, '0')} / COMPONENT</span><h3>{title}</h3><p>{summary} {detail}</p><details><summary>TECHNICAL NOTE</summary><p>{index === 4 ? 'The √dₖ scaling keeps dot products from growing too large as vector width increases. Softmax is applied across keys for each query.' : index === 10 ? 'Cross-attention belongs to the original encoder–decoder pattern used in tasks such as translation. Decoder-only GPT-style models rely on causal self-attention instead.' : 'These diagrams are conceptual illustrations. A production model uses learned parameters and vector operations across many layers.'}</p></details></div>
      </article>)}
    </section>

    <section className="section real-section" id="example">
      <div className="real-inner">
        <div className="real-intro"><div><div className="kicker">03 / From prompt to response</div><h2>See it in the real world.</h2></div><p>ChatGPT-style systems use a decoder-only Transformer. Given the prompt and earlier generated tokens, the model predicts one next token at a time.</p></div>
        <div className="chat-demo">
          <div className="prompt-card"><span className="tag">A prompt to begin</span><blockquote>“Explain photosynthesis in simple words.”</blockquote><div className="flow-list">{['Prompt & tokenization', 'Embeddings + position', 'Decoder Transformer layers', 'Next-token prediction', 'Repeat until complete'].map((x, i) => <div className="flow-step" key={x}><b>0{i + 1}</b>{x}</div>)}</div></div>
          <div className="response-card"><div className="response-top"><span>Generated response</span><span aria-live="polite">{generated} / {responseTokens.length} tokens</span></div><p className="response-copy" aria-live="polite">{responseTokens.slice(0, generated).map((token, i) => <span className="generated-token" key={`${i}-${token}`}>{token}</span>)}{generated === 0 && <span style={{ color: '#9a8b78' }}>Press play to reveal the next token.</span>}</p>
            <div className="control-row"><button type="button" className="btn primary" onClick={() => { if (generated >= responseTokens.length) setGenerated(0); setIsGenerating(true); }} disabled={isGenerating} data-testid="button-generate">{generated >= responseTokens.length ? 'Play again' : isGenerating ? 'Generating…' : 'Generate response'}</button><button type="button" className="btn" onClick={() => { setIsGenerating(false); setGenerated(0); }}>Reset response</button></div>
            <div style={{ marginTop: 24 }}><div className="prob-title">Next-token candidates</div>{[['is', 68], ['can', 17], ['the', 9]].map(([word, amount]) => <div className="prob-row" key={word}><span>{word}</span><span className="prob-bar"><i style={{ width: `${amount}%` }} /></span><span>{amount}%</span></div>)}<div className="prob-note">Illustrative probability distribution—not measured from a language model.</div></div>
          </div>
        </div>
      </div>
    </section>

    <section className="pdf-section" id="presentation"><div className="pdf-card">
      <div className="pdf-preview" aria-label="Presentation cover preview"><div className="pdf-sheet"><small>FIELDNOTES / LEARNING SERIES</small><strong>Transformer<br />Networks</strong><span className="pdf-line" /><span className="pdf-line short" /><small>ATTENTION · CONTEXT · GENERATION</small></div></div>
      <div className="pdf-copy"><div className="kicker">Keep exploring, at your own pace</div><h2>Take the presentation with you.</h2><p>Want to revisit the Transformer Networks presentation later? The original 15-page presentation is here as a PDF.</p><a className="btn primary" href={`${import.meta.env.BASE_URL}Transformer_Networks.pdf`} download="Transformer_Networks.pdf" data-testid="link-download-pdf">Download the PDF <span aria-hidden="true">↗</span></a></div>
    </div></section>
    <section className="closing"><div className="kicker">A final thought</div><h2>That’s the<br />Transformer.</h2><div className="flow-ribbon"><span>Attention</span><b>→</b><span>Context</span><b>→</b><span>Representation</span><b>→</b><span>Prediction</span><b>→</b><span>Generation</span></div><p>What looks like a simple sentence becomes a flow of representations, relationships and predictions inside the model.</p><a className="btn" href="#top">Explore again ↑</a></section>
    <footer className="footer"><span>A small guided exhibit about Transformer networks</span><a href="#presentation">Presentation PDF</a></footer>
  </main>;
}

export default App;
