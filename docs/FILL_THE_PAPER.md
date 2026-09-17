# Filling the manuscript when the runs are done

Every number in `paper/steerf.tex` is a `\num{name}` slot resolved from
`results/numbers*.tex`, which `scripts/analyze_seeds.py` writes from the
training logs. Nothing is typed by hand, so "filling the paper" is running the
analysis in the right order.

A slot with no value renders as a red `--`. That is the whole safety property:
an unmeasured number can never be typeset as a number.

## 0. Get every log onto one branch

The queues judge what is finished from the logs on *their own* box, and the
analysis needs all of them together. On **each** box:

```bash
cd /workspace/entropy_collapse
bash run/publish_logs.sh --push      # this box's logs -> origin/paper
bash run/publish_logs.sh --pull      # the other box's logs -> here
```

Check that nothing is still missing:

```bash
for q in campaign followups backbones; do
  printf '%-11s %s left\n' "$q" "$(DRY=1 bash run/run_$q.sh 2>/dev/null | grep -c QUEUE)"
done
```

All three should read `0`.

## 1. The six-benchmark evaluation

Section 12.8's direction-consistency count comes from the evaluation table,
not from the training logs:

```bash
ROLE=final bash run/run_paper.sh          # eval + analysis
# or, just the table:
python3 scripts/collect_results.py --logs logs/experiments --out results/summary.tsv
```

## 2. The backbones, one invocation each

A backbone is the same five-arm analysis under another model tag, so it reuses
the same script with a macro prefix. The manuscript `\input`s one file per
backbone.

```bash
python3 scripts/analyze_seeds.py --git-ref origin/paper \
    --model-tag Qwen2.5-Math-7B --macro-prefix Bqwenbig \
    --out results --tex-macros results/numbers-qwenbig.tex

python3 scripts/analyze_seeds.py --git-ref origin/paper \
    --model-tag Llama-3.2-3B-Instruct --macro-prefix Bllama \
    --out results --tex-macros results/numbers-llama.tex

python3 scripts/analyze_seeds.py --git-ref origin/paper \
    --model-tag Mistral-7B-v0.3 --macro-prefix Bmistral \
    --out results --tex-macros results/numbers-mistral.tex
```

## 3. The main analysis — LAST

It must run after step 2: `Bsigncount` counts how many backbones keep the sign
of STEER-F $-$ GRPO, and it reads the files step 2 wrote.

```bash
python3 scripts/analyze_seeds.py --git-ref origin/paper \
    --eval-table results/summary.tsv --out results
```

This writes `results/numbers.tex` plus `per_seed.tsv`, `arm_means.tsv`,
`contrasts.tsv` and `compute_match.tsv` — the tables to read while checking the
prose.

## 4. Build, and see what is left

```bash
cd paper && pdflatex steerf && bibtex steerf && pdflatex steerf && pdflatex steerf && cd ..
```

```bash
python3 - <<'PY'
import re
tex = "\n".join(l for l in open("paper/steerf.tex").read().splitlines()
                if not l.lstrip().startswith("%"))
have = {}
import glob
for f in glob.glob("results/numbers*.tex"):
    have.update(dict(re.findall(r"\\providecommand\{\\([A-Za-z]+)\}\{([^}]*)\}",
                                open(f).read())))
want = [m.group(1) for m in re.finditer(r"\\num\{([A-Za-z]+)\}", tex)]
left = sorted({n for n in want if have.get(n, "\\PENDING") == "\\PENDING"})
print(f"filled {len(want) - sum(1 for n in want if n in left)} / {len(want)}")
print("still empty:", " ".join(left) if left else "nothing")
PY
```

`still empty: nothing` means the manuscript is numerically complete.

## 5. Re-read the prose against the numbers

The slots fill themselves; sentences do not. Two greps:

```bash
grep -n 'tocheck' paper/steerf.tex      # wording that presumes a direction
grep -c Overfull paper/steerf.log       # 0; a filled table can change width
```

`\tocheck{...}` marks every claim whose *wording* assumes a result the campaign
had not confirmed when it was written. Each one has to be read against
`results/contrasts.tsv` and either kept or rewritten. This is the step that
cannot be automated, and it is the one that decides whether the paper is honest.

Three statements in particular are written for one seed and must be revisited
once there are five:

- Section 12.7 quotes GRPO's two-seed spread as the measure of run-to-run
  variation. With five seeds the seed-level standard error replaces it.
- The Limitations paragraph "A single seed, and a gap the same order as the
  noise on it" is written to be deleted.
- Section 12.9's "every comparison in it is therefore n = 1" holds only while
  the follow-ups stay at one seed.
