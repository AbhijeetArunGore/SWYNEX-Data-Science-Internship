# Security, Data Privacy & Governance Policy

## 1. Security Overview & Principles
This repository adheres to strict secure software development lifecycle (SSDLC) and healthcare data protection practices:
- **Zero Hardcoded Secrets**: No API keys, passwords, database URLs, cloud tokens, or personal credentials may ever be committed to source control.
- **Environment Separation**: Application configuration is managed through decoupled environment variables using `.env.example` templates.
- **Privacy by Design**: No identifiable patient data (PHI) is collected, persisted, or transmitted.

---

## 2. Health Data Privacy & HIPAA Alignment

### Public De-Identified Research Data
- **Dataset**: UCI Diabetes 130-US Hospitals dataset (1999–2008).
- **De-Identification**: Sourced exclusively from a public, retrospectively de-identified clinical research database approved under institutional research guidelines.
- **No Direct Identifiers**:
  - Direct identifiers (patient names, social security numbers, street addresses, exact admission/discharge dates) are absent.
  - Dates are grouped into historical multi-year intervals (1999–2008).
  - Patient ages are aggregated into 10-year age brackets (e.g., `[70-80)`).
  - `encounter_id` and `patient_nbr` are random surrogate research keys and do not correspond to hospital Medical Record Numbers (MRNs).
- **Prohibition on Re-Identification**: Attempting to cross-reference or re-identify individuals from this dataset is strictly prohibited.

### Application Input Privacy
- The interactive Streamlit application operates purely in-memory:
  - User-entered patient parameters in the risk estimator are processed in volatile memory.
  - No inputs are logged to disk, transmitted across external networks, or stored in persistent databases.

---

## 3. Secret Management & Credential Policy

### Rules for Developers & Contributors:
1. **Never Commit Secrets**: Do not write API tokens, cloud credentials (AWS/GCP/Azure), private keys (`.pem`, `.key`), or database connection strings in Python files, notebooks, scripts, or documentation.
2. **Use Environment Variables**: If external integrations are introduced, load configurations via environment variables (`os.environ.get(...)`) and define placeholder templates in `.env.example`.
3. **Strict Git Ignore Rules**: Ensure `.gitignore` covers:
   - `.env`, `.env.*`, `*.env`
   - `*secret*`, `*credential*`, `*.token`, `*.key`, `*.pem`
   - `__pycache__/`, `.pytest_cache/`, `*.log`
   - Virtual environments (`venv/`, `.venv/`, `env/`)

---

## 4. Security Incident Response Protocol

If an unauthorized secret, credential, or sensitive artifact is ever discovered:
1. **Immediate Revocation**: Immediately revoke, rotate, and invalidate the exposed secret at the issuing authority (e.g. GitHub, AWS, cloud provider).
2. **Do Not Push**: Halt all push operations immediately.
3. **Audit Git History**: Inspect all historical commits using `git log -S <secret_identifier>`.
4. **History Cleansing**: Use dedicated repository hygiene tools (such as `git-filter-repo` or BFG Repo-Cleaner) to purge the secret from historical commit objects if necessary, prior to any public release.
5. **Report & Document**: File a post-incident security review documenting root cause and updated prevention rules.

---

## 5. Pre-Commit & Pre-Push Checklist

Before staging, committing, or pushing code to any remote repository:
1. `git status` — Verify that only intended files are modified or untracked.
2. `git diff --check` — Ensure no whitespace errors, trailing markers, or accidental debug artifacts exist.
3. `python scripts/security_audit.py` — Run the automated security scanner to verify zero secrets and no absolute machine paths.
4. `python -m pytest tests/ -v` — Ensure all unit tests pass completely.
5. `git diff --cached` — Review all staged changes line-by-line prior to committing.
