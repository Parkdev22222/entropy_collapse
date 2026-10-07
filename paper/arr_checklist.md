# ARR Responsible NLP Checklist — draft answers

Draft for the OpenReview submission form (ARR, October 2026 cycle). Licenses and the
compute total were filled on 2026-10-07 (sources noted inline); E is the authors'
own declaration and must be confirmed by them before pasting. Section numbers refer to `paper/steerf.tex` as built (main body = Sections
1–7, appendices A–J).

## A. For every submission

- **A1. Did you describe the limitations of your work?** Yes — *Limitations* (unnumbered
  section after Section 7): the met pre-registered falsification condition, the
  forecaster's cost, three seeds, a single 30-problem validation benchmark, the
  unmeasured sibling-spread prediction, the small support, the base method's
  non-reproduced gain over GRPO, the uncontrolled published-number comparison,
  Assumption B2, and a single model.
- **A2. Did you discuss any potential risks of your work?** Yes — *Ethics Statement*:
  the method inherits whatever the reward function encodes, and carries an extra
  compute cost that Appendix E.12 quantifies.

## B. Did you use or create scientific artifacts?

Yes.

- **B1. Cite the creators?** Yes — Qwen2.5-Math-1.5B (Yang et al., 2024),
  DAPO-Math-17k (Yu et al., 2025), verl (Sheng et al., 2025), the base method
  (Hao et al., 2026), and the evaluation sets in Appendix E.9 (MATH500, Minerva,
  OlympiadBench; AIME 2024/2025 and AMC 2023 are public competition problems).
- **B2. Discuss the license or terms?** Yes — Appendix D.1 lists them:
  Qwen2.5-Math-1.5B, DAPO-Math-17k (Hugging Face dataset card: apache-2.0) and verl —
  Apache-2.0; MATH, GSM8K and OlympiadBench (OpenBMB repository: MIT) — MIT; the
  Minerva evaluation problems (272, from MIT OpenCourseWare) — CC BY-NC-SA, which our
  non-commercial research use satisfies; AIME/AMC — public competition problems,
  used for evaluation only.
- **B3. Use consistent with intended use?** Yes — all are research releases for
  training/evaluating mathematical reasoning, used for that purpose only.
- **B4. Data containing personal information or offensive content?** No — competition
  mathematics problems; no personal data.
- **B5. Documentation of the artifacts?** Yes for what we submit: the training and
  evaluation logs of every reported run and the per-seed result tables
  (`per_seed.tsv` etc.) are in the anonymized supplementary archive
  (`paper/arr_supplement.zip`, built by `scripts/make_supplement.py`), with a
  README. Code is not submitted for review; it will be released upon publication.
- **B6. Statistics of the data?** Yes — Section 5 and Appendix D: 17k training
  problems, 512 prompts × 110 steps, AIME24 30 problems × 32 samples; Appendix E.9
  for the six benchmarks.

## C. Did you run computational experiments?

Yes.

- **C1. Parameters, compute budget, infrastructure?** Yes — 1.5B parameters;
  4×H100 and 2-GPU boxes (Appendix E.8); measured step times on the four-GPU box
  are 449–538 s per optimization step depending on the arm (Appendix E.12), i.e.
  about 15 h of a 4-GPU box (≈60 GPU-hours) per 110-step run. Total: the logged
  optimization-step times of every training run in the supplement (five-arm
  campaign, failed and excluded runs, follow-ups, pilots and probes, the two
  200-step runs), multiplied by each run's GPU count, sum to about 2,340 GPU-hours;
  evaluation, MTP-head warm-up and idle time are not in that figure, so the total
  is roughly 2,400–2,600 GPU-hours.
- **C2. Experimental setup and hyperparameters, and how they were chosen?** Yes —
  Section 5 and Appendix D. λ_min = 0.7 is taken from the base method's paper and not
  tuned; λ = 0.25 was not tuned on these runs (sweep in Table 7); the two-root tree of
  Section 6.5 was chosen after the five-arm campaign on training prompts and is
  labelled exploratory.
- **C3. Descriptive statistics, and whether results are single runs or summaries?**
  Yes — seed-level means ± standard error over 3 seeds, paired *t* with df = n−1,
  per-seed ranges, and the across-problem SE of one AIME24 evaluation; single-seed
  ablations are marked n = 1 with no error bar.
- **C4. Existing packages used, with versions and parameters?** Yes — verl, vLLM 0.8.4,
  PyTorch 2.6, transformers 4.x (Appendix H; exact versions will be
  released with the code). Answer extraction and grading: the base method's
  released scorer (Math-Verify-style rule plus Qwen grader).

## D. Did you use human annotators or research with human subjects?

No. (D1–D5 not applicable.)

## E. Did you use AI assistants in your research, coding, or writing?

Yes. Proposed answer, to be confirmed by the authors (only they can state how they
used it):

- **E1. Information about the use of AI assistants?** An AI assistant was used for
  writing code (training and analysis scripts), running and checking analyses, and
  drafting and editing the text, including literature search. All numbers in the
  paper are emitted by scripts from the training and evaluation logs; the authors
  reviewed the code, checked every claim and citation, and take responsibility for
  the content. The AI assistant is not an author.
