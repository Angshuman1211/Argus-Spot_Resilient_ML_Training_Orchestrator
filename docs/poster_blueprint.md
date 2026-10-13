# Argus — 4-page Paper Blueprint (ML for Systems @ NeurIPS 2026)

**Framing (v2, 2026-08-04): a BENCHMARK / EVALUATION-METHODOLOGY paper.** Central
question — *"when does predicting Spot interruptions actually beat simple
checkpointing?"* — not a system tour. Motivate via **expensive large-model
training** (that is where a lost checkpoint costs real money); CIFAR-10 is the
controlled testbed. This matches how the lighter accepted ML4Sys papers win
(paper19 MoE-GPS "guidelines", paper12 "methodology", paper6 "benchmark").

Rules: write prose from this; the `.tex` (`Paper/neurips_2026.tex`) carries the
same plan as `%` comments — **you write the prose, not me.** 4 pages STRICT (excl
refs/appendix). Non-anonymized (real names). Deadline **Aug 29 2026 AoE**. Every
number below is sourced from `docs/objective{1,2,3}_result.md`,
`docs/week7_model_evaluation.md`, `problems_and_decisions.md`. **No hype / no
resume language** — the accepted papers are sober.

Figures: **Fig 1** real-EKS survival (lead with it) or architecture; **Fig 2**
Grafana (`docs/Figures/grafana_dashboard.png`); **Fig 3** wasted-compute-by-arm
(`benchmark/results/figures/`); **Fig 4 (NEW, needs runs)** lead-time crossover.
Tables: **Table 1** Obj1 timeline; **Table 2** benchmark sweep.

---

## Title + Abstract (~120 words)

**Title = the question, not the system.** e.g. *"When Does Predicting EC2 Spot
Interruptions Beat Simple Checkpointing? A Real-EKS Study of Cost-Resilient
Training."*

**Abstract — cover in order:**
1. Spot is 70–90% cheaper but reclaimed on 2 min notice; the loss matters **most
   for expensive large-model training** (a reclaim on an *N*-GPU job discards all
   uncheckpointed progress = \$thousands).
2. Question: does *predicting* interruptions beat simple checkpointing? We build
   Argus (a Kubernetes operator) to answer it empirically on a controlled
   CIFAR-10 testbed.
3. Real-EKS result: survived a real Spot drain (SIGTERM checkpoint → resumed from
   epoch 8).
4. Benchmark (80 trials): reactive checkpointing collapses toward no-protection
   once interruptions outpace the fixed 2-min notice; a lead-time sweep shows
   predictive wins only **above a threshold lead** → a concrete guideline.
5. Honest scope: the predictor is advisory (proxy label); real labels and
   large-model-scale validation are future work.

---

## §1 Introduction (~0.5 page)

**Cover in order:**
1. Spot cost gap → **the motivation move (make it load-bearing, not a buzzword):**
   the problem barely matters for cheap jobs but is severe for expensive
   large-model / LLM training. Substantiate:
   - back-of-envelope: one reclaim on an *N*-GPU synchronous job discards all
     uncheckpointed progress; at \$X/GPU-hr that is \$Y ≫ the checkpoint cost;
   - one node reclaim stalls the **whole** synchronous job → more nodes = higher
     effective interruption exposure → *ties directly to the §3.2 finding.*
   Then scope honestly: *"CIFAR-10 is a cheap controlled testbed; the mechanism
   is workload-agnostic."* **Never claim LLM experiments.**
2. The catch (2-min notice) + **state the reviewer's question up front** — *"why
   predict when AWS gives a free 2-min notice?"* — promise §3.2/§3.3 answer it.
3. **Contributions (bulleted):** (i) real-EKS-validated survival; (ii) a benchmark
   isolating *when* prediction beats reactive/periodic; (iii) a lead-time
   sensitivity result → a *guideline*; (iv) honest model characterization + artifacts.

## §2 System Design (~0.75 page, Fig 1)

Keep tight — context for the evaluation, don't let it eat the budget. Cover: the
3 layers; the **decoupled** checkpoint design (operator signals, pod flushes);
and **the key point** — *predictive* (pre-migrate) and *reactive* (real NTH 2-min
warning) paths converge on one SIGTERM→checkpoint→reschedule→resume. One sentence
each on infra (EKS+IRSA zero-static-creds, kopf CRD+reconcile).

## §3 Evaluation (~2 pages)

### §3.1 Real-EKS interruption survival (Table 1, lead figure)
Timeline: drain 10 s → SIGTERM checkpoint (epoch 8 → S3) → replacement pod ~75 s
→ "Resuming from epoch 8"; only the in-progress epoch lost. **Disclosure #1:**
injected schema-conformant warning, *not* an AWS-issued FIS reclaim (NTH can't
distinguish; path genuinely exercised; FIS = future work). cite ADR-007 / P-019.

### §3.2 When does prediction pay? (Table 2, Fig 3) — the core
4 arms × 4 rates × 5 reps = 80 trials, Poisson kills. Headline at MTBF=120 s:
no-protection 202.3 s wasted, reactive 169.4, periodic 4.3, predictive 0.0.
**The argument:** reactive's 120 s notice ≈ the interval → rarely enough lead →
wasted compute collapses toward no-protection. *The free 2-min notice stops
helping exactly when interruptions arrive faster than ~once/2 min* — and (tie to
§1) that is the large multi-node regime. **Disclosure #2:** predictive's 0.0
assumes `risk_lead_seconds=300` → handled in §3.3. **Disclosure #3:** periodic
(no ML) nearly matches on waste and wins makespan/cost; predictive's only clean win =
zero wasted compute (recovery time is not comparable: 1 of 5 predictive runs completes at this rate). Credit periodic openly.

### §3.3 How much lead time does prediction need? (Fig 4) — the guideline
**DONE** (data: `benchmark/results/leadsweep_table.csv`; Fig 4:
`benchmark/results/figures/leadtime_sensitivity.png`; write-up:
`docs/leadtime_sensitivity_result.md`). Swept `risk_lead_seconds` ∈ {2…60}s at a
20 s interruption interval, 5 reps + periodic. **The finding inverts the naive
"more lead is better":**
- **Wasted compute ≈ 0 at *every* lead** (even 2 s) — the operator checkpoints
  synchronously *before* migrating, so zero-waste does **not** need a large lead
  (practical lower bound = the checkpoint write time). *This defuses the
  "assumed 600 s oracle lead" objection.*
- **The cost of excess lead is over-migration.** Predictive beats periodic on
  makespan only up to a knee ≈ the interruption interval (~10 s here); at 60 s
  lead (3× interval) it migrates ~93× and makespan nearly triples (218 s vs 73 s
  at 2 s). Below the knee it's *strictly* better than periodic (0 waste, ≤
  makespan, fewer checkpoints, 5–9 vs 29).
- **Guideline:** set lead just above checkpoint duration, well below the
  interruption interval; more is strictly wasteful.
- **Explains the §3.2 anomaly** (predictive over-checkpointed + higher makespan):
  that arm used a 600 s lead ≈ 30× the fast-regime interval → deep in the
  over-migration zone. `measure_lead_time.py` gives 600 s mean proxy-lead — safe
  for waste, costly on makespan; fix = cap effective lead, not a bigger model.
  *(Synthetic caveat: checkpoint is near-instant here, so the lower bound is
  tiny; in production it = the real checkpoint write time — state this.)*

### §3.4 The risk model (brief — 2–3 sentences)
Advisory only: **14.68× base-rate lift** (5-seed mean, 95% CI ≈ [9.9×, 19.5×]) on
a **proxy label**; calibrated scores are tiny (best-F1 threshold ≈ 0.0015, see
`docs/week7_model_evaluation.md`). **Disclosure #4:** ceiling = label quality;
never claim it predicts real interruptions. Full 4-round history → **appendix.**

### §3.5 Observability (2 sentences + Fig 2)
Operator emits Prometheus metrics; Grafana shows risk crossing the threshold →
proactive checkpoint firing at that instant (real operator, real S3/SQS).

## §4 Limitations (~0.25 page)
One paragraph consolidating disclosures #1–#4 + synthetic benchmark job + **no
large-model-scale validation** (stated honestly, as future work). Owning these is
the credibility.

## §5 Related Work + Conclusion (~0.5 page)
Related (4–6 refs): checkpoint-restart / Spot-resilience systems; ML-for-systems
failure/interruption prediction; K8s operators for stateful workloads; Spot for
large-model training. Conclusion: the benchmark maps where prediction beats the
free reactive notice; validated on real EKS. Future work: FIS reclaims; forward
label capture; measure real lead vs L\*; multi-node large-model runs.

---

## Pre-submission checklist
- [ ] Motivation is load-bearing (cost arg + interruption-rate bridge), **not** a
      buzzword; CIFAR scoped as testbed; **zero** LLM-result claims.
- [ ] Framed as a benchmark/guideline, not a system tour.
- [ ] §3.3 sensitivity experiment actually run; L\* reported.
- [ ] All four disclosures present and not buried.
- [ ] 14.7× always tagged "proxy label"; model demoted to §3.4 + appendix.
- [ ] Obj1 always "injected schema-conformant event," never "AWS reclaimed it."
- [ ] `neurips_2026.sty` + `checklist.tex` downloaded; compiles; ≤4 pages.
- [ ] Real names (non-anonymized). No hype/resume language anywhere.
- [ ] Figures legible at print size; Fig 2 trimmed to the clean climb.
