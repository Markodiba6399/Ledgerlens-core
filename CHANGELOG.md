# Changelog

All notable changes to `ledgerlens-core` are documented in this file.

## Unreleased

### Added
- **Monte Carlo bootstrap p-values for Benford chi-square** (`detection/benford_engine.py`):
  wallets with fewer than `BENFORD_BOOTSTRAP_THRESHOLD` (default 100) transactions
  in a window now use an empirical p-value derived from 10,000 multinomial samples
  drawn from the theoretical Benford distribution, eliminating false positives caused
  by asymptotic chi-square approximation failures in small-sample regimes common on
  SDEX short time windows (1h, 4h).
- `bootstrap_chi_square_pvalue` function with fully vectorised NumPy implementation
  (single `rng.multinomial` call; < 500 ms for N = 50, n = 10,000).
- `BENFORD_PROBS` numpy array constant (normalised Benford probabilities for digits 1–9).
- `BENFORD_BOOTSTRAP_THRESHOLD` and `BENFORD_BOOTSTRAP_SAMPLES` module constants,
  overridable via environment variables.
- `compute_chi_square_pvalue(counts, N) -> (p_value, method)` function that dispatches
  to bootstrap or asymptotic computation based on sample size.
- LRU cache (`maxsize=512`) on `_cached_bootstrap_pvalue` to avoid recomputing p-values
  for repeated wallet-window evaluations with the same digit counts.
- `BenfordWindowFeatures` dataclass with `chi_square_pvalue_method` field so callers and
  audit logs know whether a flagging decision used bootstrap or asymptotic estimates.
- `chi_square_pvalue` and `pvalue_method` keys added to the dict returned by
  `compute_benford_metrics` (backward-compatible; existing keys unchanged).
- `--bootstrap-threshold` and `--bootstrap-samples` CLI flags on `ledgerlens score`.
- `BENFORD_BOOTSTRAP_THRESHOLD` and `BENFORD_BOOTSTRAP_SAMPLES` documented in `.env.example`.
- `docs/benford_analysis.md` with "Small-Sample P-Value Estimation" methodology section.
- Synthetic SDEX trade generator (`ingestion/synthetic_data.py`) with
  labelled wash-trading rings for local training and testing.
- Labelled training dataset builder (`detection/dataset.py`).
- SQLite-backed local `RiskScore` store (`detection/storage.py`).
- Local read-only FastAPI app (`api/main.py`) serving `/scores`, `/alerts`,
  and `/assets/risk-ranking`.
- `ledgerlens` CLI (`cli.py`): `generate-data`, `train`, `score`, `serve`.
- Retrying HTTP client for Horizon API calls (`ingestion/http_client.py`).
- Dockerfile, docker-compose, and GitHub Actions CI workflow.

### Fixed
- `detection/shap_explainer.py` updated for the current SHAP `TreeExplainer`
  output shape.

## 0.1.0

- Initial scaffold: Horizon ingestion, Benford's Law engine, ML feature
  engineering, ensemble model training/inference, `RiskScore` schema.
