# Transformer — A Guided Learning Experience

An interactive, calm science-exhibit-style learning platform exploring Transformer neural networks with deterministic live computation, interactive multi-head attention heatmaps, mathematical step breakdowns, and next-token generation.

---

## 🚀 Deploying to Vercel

This repository is pre-configured for seamless zero-config deployment to [Vercel](https://vercel.com).

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. Push this repository to your GitHub, GitLab, or Bitbucket account.
2. Go to the [Vercel Dashboard](https://vercel.com/dashboard) and click **"Add New..."** → **"Project"**.
3. Import your repository.
4. **Project Settings**:
   - **Framework Preset**: Vite (detected automatically)
   - **Root Directory**: `./` (leave default)
   - **Build Command**: `pnpm --filter @workspace/transformer-experience run build` (configured automatically via `vercel.json`)
   - **Output Directory**: `artifacts/transformer-experience/dist/public` (configured automatically via `vercel.json`)
   - **Install Command**: `pnpm install` (default)
5. Click **Deploy**.

> **Note:** If you choose to set the **Root Directory** to `artifacts/transformer-experience` in the Vercel project settings, that is also fully supported via the nested [vercel.json](file:///artifacts/transformer-experience/vercel.json).

---

### Option 2: Deploy via Vercel CLI

If you have the [Vercel CLI](https://vercel.com/docs/cli) installed:

```bash
# Preview deployment
npx vercel

# Production deployment
npx vercel --prod
```

---

## 🛠 Local Development

This project uses [pnpm workspaces](https://pnpm.io/workspaces).

### Prerequisites

- Node.js >= 20
- pnpm (or run commands via `npx pnpm`)

### Running Locally

```bash
# Install dependencies
pnpm install

# Start the interactive Transformer experience
pnpm --filter @workspace/transformer-experience run dev

# Build the production bundle
pnpm run build
```

---

## 📁 Project Structure

```text
├── api/                                      # Vercel Serverless Functions (/api/healthz)
├── artifacts/
│   ├── transformer-experience/               # Main interactive React + Vite web application
│   │   ├── src/
│   │   │   ├── components/TransformerLab.tsx # 21-stage Transformer pipeline & attention heatmap
│   │   │   ├── lib/transformer-simulation.ts # Live deterministic Transformer computations
│   │   │   ├── App.tsx                       # Exhibit interface & conceptual explanations
│   │   │   └── main.tsx                      # Entry point
│   │   ├── public/                           # Static assets & PDF guide
│   │   └── vite.config.ts                    # Vite build configuration
│   ├── api-server/                           # Express API server package
│   └── mockup-sandbox/                       # UI component preview sandbox
├── lib/                                      # Shared workspace libraries (api-spec, api-zod, db)
├── vercel.json                               # Vercel deployment configuration
└── package.json                              # Root workspace configuration
```
