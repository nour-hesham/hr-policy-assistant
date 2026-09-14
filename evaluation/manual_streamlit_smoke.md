# Streamlit Smoke Checklist

Environment: Windows, Python 3.10.11, pinned project dependencies.

The real FastAPI and Streamlit processes were started with the documented
commands. FastAPI reported 81 loaded chunks and Streamlit returned HTTP 200.
Interactive rendering was verified with Streamlit's supported `AppTest` surface
because the local Codex browser plugin installation was incomplete.

- [x] Initial page renders the title, introduction, four-step process strip,
  labelled question area, and enabled submit button when `/health` is ready.
- [x] Successful submission keeps the question visible and renders an `Answer`
  section, source count, and a source expander.
- [x] The first cited source is expanded and displays trusted document ID,
  version, source filename, and canonical evidence.
- [x] Partial response renders a restrained warning, the supported/missing
  answer, and its cited source.
- [x] `not_found` renders the canonical neutral information message and no empty
  `Sources used` section.
- [x] Backend-unavailable state renders a setup/connection error and disables
  submission.
- [x] No tested UI state displays similarity scores, embeddings, prompts, raw
  provider responses, credentials, or model controls.
- [x] The live backend/API smoke returned valid `answered`, `partial`, and
  `not_found` responses through the same HTTP contract used by Streamlit.

The two temporary local server processes were stopped after verification.
