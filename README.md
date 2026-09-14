# HR Policy Assistant

A stateless question-answering application grounded only in ten supplied HR
policy documents. It combines local semantic retrieval, grounded Gemini
generation, deterministic citation validation, a FastAPI backend, and a
Streamlit frontend.

## Quick start

### Windows PowerShell

Run these commands from the repository root:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your own Gemini API key:

```dotenv
GEMINI_API_KEY=replace_with_your_key
```

Start the backend and frontend together:

```powershell
.venv\Scripts\python.exe run_app.py
```

The launcher waits for the policy index, starts Streamlit, and opens
`http://127.0.0.1:8501`. Press `Ctrl+C` in the same terminal to stop both
services.

### macOS or Linux

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
# Replace GEMINI_API_KEY in .env, then run:
.venv/bin/python run_app.py
```

The public embedding model downloads automatically on first use and is reused
from the standard Hugging Face cache. No Hugging Face account or token is
required. A Gemini key is required only for live generated answers; the test
suite uses fakes and does not require one.

## Example outputs

These are unchanged outputs produced by the final application. Full examples
with source metadata are also available in [`sample_qa.md`](sample_qa.md).

### Conditional numeric rule

**Question:** How many days of annual leave am I entitled to after 6 years of service?

**Status:** `answered`

**Answer:** After 6 years of continuous service, you are entitled to 25 working
days of paid annual leave per calendar year [S1].

**Source:** Annual Leave Policy - Entitlement

### Direct procedural answer

**Question:** What is the procedure for filing a formal grievance?

**Status:** `answered`

**Answer:** To file a formal grievance, an employee must submit a formal written
complaint to HR within 30 calendar days of the event or the informal resolution
step, whichever is later [S1]. The complaint must include a clear description
of the concern, relevant dates, people involved, supporting information, and
the desired outcome [S1]. HR will acknowledge receipt of the complaint within 3
business days [S1].

**Source:** Grievance and Complaints Procedure - Step 2: Formal written complaint

### Multiple sections from one policy

**Question:** What happens if I work on a weekend — do I get extra pay?

**Status:** `answered`

**Answer:** If you work on a weekend, you may be eligible for overtime pay if
you are in a non-exempt role and the work is pre-approved [S2]. For non-exempt
roles, weekend overtime is paid at 2.0 times the standard hourly rate, though
time in lieu is preferred over cash payment whenever practical [S2]. Exempt
roles are not eligible for overtime payment, though time-in-lieu arrangements
may be available at a manager's discretion in exceptional circumstances [S2].
All weekend work must be pre-approved by both your line manager and department
head [S1].

**Sources:** Work Hours and Overtime Policy - Weekend and public-holiday work;
Overtime

### Cross-policy answer

**Question:** When I work remotely, how many days per week can I do it and during
what core hours must I be available?

**Status:** `answered`

**Answer:** Eligible employees may work remotely for up to 2 days per calendar
week [S1]. Regardless of whether you are working in the office or remotely, you
are expected to be available during the core hours of 10:00 AM to 4:00 PM [S2].

**Sources:** Remote Work Policy - Standard arrangement; Work Hours and Overtime
Policy - Core hours

### Partial answer

**Question:** What is the annual training budget, and what paid parental leave
does the company offer?

**Status:** `partial`

**Answer:** Each permanent employee has an annual training budget of USD 2,500
per calendar year [S1]. The provided policy documents do not contain
information regarding paid parental leave.

**Source:** Learning and Development Policy - Annual training budget

### Unsupported question

**Question:** What is the capital of Japan?

**Status:** `not_found`

**Answer:** I couldn't find this information in the supplied HR policies.

**Sources:** None

## Architecture and design choices

- **Knowledge base:** only `policies/*.md`; each Markdown `##` section becomes a
  validated semantic chunk with deterministic metadata and content hashes.
- **Retrieval:** `BAAI/bge-small-en-v1.5` runs locally on CPU. Normalized
  embeddings are ranked with exact NumPy cosine similarity using the calibrated
  configuration `top_k=3` and `min_relevance_score=0.625`.
- **Model selection:** BGE was compared with `all-MiniLM-L6-v2` and a TF-IDF
  baseline on fixed labelled questions before the production choice was locked.
- **Generation:** `gemini-3.1-flash-lite` receives only retrieved evidence and
  returns structured `answered`, `partial`, or `not_found` output.
- **Grounding:** generation is skipped when retrieval finds no evidence. Source
  IDs and inline citations are validated and mapped back to trusted metadata.
- **Application:** FastAPI initializes shared resources once; Streamlit calls
  the backend only through HTTP. The API exposes `GET /health` and
  `POST /api/v1/ask`.

## Evaluation

The fixed evaluation set contains 10 calibration and 5 untouched holdout
questions covering all policies, numeric and conditional rules, compound
questions, and unsupported requests.

| Evaluation | Hit@1 | Hit@3 | MRR |
|---|---:|---:|---:|
| BGE calibration | 0.875 | 1.000 | 0.938 |
| BGE holdout | 0.667 | 1.000 | 0.778 |

In the live 15-case answer review, 14 cases passed each of factual correctness,
groundedness, citation correctness, and completeness. Results and review notes
are stored under `evaluation/results/`.

## Tests

Deterministic tests do not download embedding models or call Gemini:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

The suite covers ingestion, ranking and threshold behavior, generation and
citation contracts, service orchestration, API responses, logging, and the
one-command launcher.

## Repository structure

```text
backend/      FastAPI, ingestion, retrieval, generation, and orchestration
frontend/     Streamlit HTTP client
policies/     ten authoritative Markdown policies
evaluation/   labelled cases, evaluation harnesses, and recorded results
tests/        deterministic tests using synthetic vectors and fake providers
run_app.py    one-command local launcher
sample_qa.md  six unchanged example runs
```

## Limitations and next steps

- Gemini latency, availability, and quota are external dependencies.
- The in-memory index and single-worker setup target this small corpus; a larger
  multi-user system would need persistent vector storage and broader evaluation.

With more time, I would expand unsupported and paraphrased test coverage,
recalibrate on a larger split, and evaluate reranking only if wider-candidate
recall is strong but top-rank quality remains weak.

The policies are fictional. Real HR or personal data would require a separate
privacy, retention, access-control, and provider review.

## AI assistance and references

AI tools, including Codex, approach
brainstorming, and documentation. I reviewed the design choices,and validated the final system
with deterministic tests and direct output review.

Technical references used when considering the approach:

- [BGE small English v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [Sentence Transformers semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
- [Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output)

API keys, virtual environments, caches, model weights, and local `.env` files
are excluded from Git.
