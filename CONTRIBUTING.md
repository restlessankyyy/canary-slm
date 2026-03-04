# Contributing to CANARY Enterprise Suite

First off, thank you for considering contributing to CANARY! It's people like you that make this Financial AI Suite a robust tool for the community.

## 1. Where do I go from here?

If you've noticed a bug or have a question that isn't answered in the `README.md`, search the [issue tracker](https://github.com/restlessankyyy/canary-slm/issues) to see if someone else has already created a ticket. If not, go ahead and make one using our Issue Templates!

## 2. Setting up your local environment

Fork the repository, clone it locally, and set up your Python environment (3.11+ recommended).

```bash
git clone https://github.com/YOUR_USERNAME/canary-slm.git
cd canary-slm
pip install -r requirements.txt
```

If you are modifying the Next.js Analyst Dashboard:
```bash
cd dashboard
npm install
npm run dev
```

## 3. Core Development Principles

### 🧠 Modifying the ML Core (`model.py`, `aml/`)
* **Keep it Lightweight**: CANARY is a *Small Language Model* (SLM). Changes to the PyTorch Transformer architecture should maintain the <1M parameter constraint to ensure millisecond-latency inference.
* **Evaluate Performance**: If you change the model architecture or data pipeline, run `evaluate.py` or `aml/evaluate_aml.py` to ensure AUC-ROC and F1 scores remain stable or improve.

### 🛡️ Modifying the API / Rules Engine (`api/`)
* The `api/rules.py` engine exists to evaluate deterministic logic *before* hitting the PyTorch model. 
* Any new rules added (e.g., new OFAC sanctions, new transaction limits) should fail-fast and return standard JSON payloads.

### 🖥️ Modifying the Next.js Dashboard (`dashboard/`)
* **Aesthetic Consistency**: The UI uses an ultra-minimalist, "Global Threat Mesh" aesthetic inspired by Zoox and Starlink. Stick to pure black backgrounds (`bg-black`), neon telemetry highlights (cyan, yellow, red), and geometric typography (`JetBrains Mono`, `Outfit`).
* Avoid converting the layout back to dense, typical SaaS data grids.

## 4. Linting and Code Quality

We use `ruff` to ensure Python code quality. Your PR will fail the GitHub Actions CI if there are linting errors.

Run the linter locally before committing:
```bash
ruff check .
```

To automatically fix obvious errors:
```bash
ruff check . --fix
```

## 5. Making a Pull Request

1. Create a branch with a descriptive name (`git checkout -b feature/awesome-new-rule`).
2. Commit your changes (`git commit -am 'Add an awesome new rule'`).
3. Push to the branch (`git push origin feature/awesome-new-rule`).
4. Open a Pull Request on GitHub and fill out our provided **Pull Request Template**.

We will review your PR as soon as possible. Thank you for making CANARY better!
