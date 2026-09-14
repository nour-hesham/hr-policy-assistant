# Evaluation Summary

## Scope and method

The evaluation uses 15 English, manually labelled questions: 10 calibration and
5 holdout. Labels were fixed before model inspection. Before the first holdout
execution, `hold-01` was changed from a repeated calibration concept (gym
membership) to an unseen spouse-course component, and an irrelevant first-class
fact was removed from `cal-06`. The holdout was then run once after locking the
retrieval configuration.

Ranking metrics use unfiltered candidates so they measure ordering independently
of the relevance gate. Rejection and retained-evidence metrics use the configured
threshold. Answer quality was checked directly against cited canonical evidence
and pre-labelled facts; no BLEU, ROUGE, LLM judge, or second generator was used.

## Calibration retrieval comparison

| Model | Hit@1 | Hit@3 | Hit@5 | MRR | Rejection at comparison threshold | Median latency |
|---|---:|---:|---:|---:|---:|---:|
| BGE small en v1.5 | 0.875 | 1.000 | 1.000 | 0.938 | 0.500 at 0.60 | 65.835 ms |
| all-MiniLM-L6-v2 | 0.750 | 1.000 | 1.000 | 0.854 | 0.500 at 0.40 | 24.936 ms |
| TF-IDF | 0.625 | 0.875 | 1.000 | 0.754 | 0.000 at 0.10 | 1.074 ms |

BGE was retained because quality is primary: it had the best Hit@1 and MRR while
matching MiniLM at Hit@3 and Hit@5. TF-IDF's zero rejection rate at its comparison
threshold made it unsuitable as the production route.

At threshold 0.625, BGE rejected both unsupported calibration questions, found
evidence for all supported questions, and retained every labelled expected chunk.
The maximum unsupported calibration score was 0.613967 and the minimum score of
a required expected chunk was 0.640952. Threshold 0.625 sits inside that measured
gap. Since Hit@3 and Hit@5 were equal and every required chunk appeared within
the first three, the production maximum is `top_k=3`.

Selected configuration:

- `BAAI/bge-small-en-v1.5`
- exact L2-normalized NumPy cosine ranking
- `top_k=3`
- `min_relevance_score=0.625`

## Locked holdout retrieval

| Metric | Result |
|---|---:|
| Hit@1 | 0.6667 |
| Hit@3 | 1.0000 |
| Hit@5 | 1.0000 |
| MRR | 0.7778 |
| Unsupported rejection rate | 0.5000 |
| Median query latency | 61.893 ms |

All three supported/partial cases retained their expected evidence inside the
production top three. `hold-04` (sick-leave carryover) ranked the correct sick
leave chunk third because an annual-leave carryover section ranked first; the
production maximum recovered the needed evidence.

One of two unsupported questions failed the retrieval gate. `hold-03` asked about
daily commuting and scored 0.670554 against Business Travel / Ground
transportation, so unrelated evidence was admitted. This is the primary holdout
retrieval failure category. It is preserved rather than used to retune the locked
threshold. The generator subsequently read that chunk and correctly returned
`not_found`, demonstrating the second grounding gate while also showing that
retrieval-only rejection is not perfect.

## Live Gemini answer review

The final system processed all 15 fixed questions. Retrieval rejected three
questions without a Gemini call; Gemini was called 12 times. Each case was
reviewed pass/fail on five attributes.

| Attribute | Passes | Rate |
|---|---:|---:|
| Status correctness | 13/15 | 86.7% |
| Factual correctness | 14/15 | 93.3% |
| Groundedness | 14/15 | 93.3% |
| Citation correctness | 14/15 | 93.3% |
| Completeness | 14/15 | 93.3% |

Two failures were observed:

1. `cal-07` hit the configured 30-second provider timeout. No answer was accepted
   or fabricated. It is recorded as an operational provider timeout and counted
   as a citation/status failure because no reviewable response existed.
2. `hold-01` correctly gave USD 2,500 and explicitly stated that spouse-course
   payment was not specified, with a valid citation. It nevertheless returned
   `answered` instead of the labelled `partial`; its sole failed attribute is
   status correctness, categorized as citation/status error.

Every completed answer preserved its required numeric values and conditions.
All exposed citations mapped to supplied retrieval records. The daily-commute
case returned the canonical no-evidence response after Gemini determined that
the admitted travel evidence did not answer the question.

## Operational observations

The accepted Gemini model and structured JSON output were verified live. The
official SDK's standard `response_schema` conversion rejected Pydantic's
`additionalProperties` constraint, so the implementation uses the supported
`response_json_schema` field and still validates the returned object locally.

On the tested Windows AMD CPU, the installed PyTorch build intermittently raised
a native access violation during BGE encoding. The production retriever now uses
one Torch thread, a small batch, CPU explicitly, and disables MKLDNN. Three
consecutive full 81-chunk startups then completed successfully.

## Limitations and next measurements

These results are directional evidence from a tiny hand-authored English set, not
statistical proof of production accuracy, robustness, or demographic fairness.
Latency includes local hardware and cache effects. Gemini latency and availability
depend on an external free-tier service.

The first next experiment should expand unsupported questions—especially near
policy vocabulary such as commuting versus business travel—then recalibrate on a
new calibration split without touching this recorded holdout. A cross-encoder is
worth evaluating only if a wider first-stage candidate pool shows high Recall@10
but poor Hit@1/Hit@3 or MRR. PostgreSQL with pgvector remains the scaling path for
shared persistence, frequent updates, concurrency, relational filtering,
historical versions, or a materially larger corpus.
