# Long-context retrieval: why peer 4B base models max out NIAH and we do not

**2026-09-09 - diagnosis of the NIAH/RULER gap between our 1.06B and comparable
open base checkpoints, and a training strategy for the 4B configuration -
Context_Parallelism / peer_eval, bluscriptCP, cp_data_by_length**

---

## 1. The one-line diagnosis

**Our long-context extension budget is not the problem. Our base model is.**

The literature is unanimous that context extension is cheap: 0.5-5B tokens
(Fu et al. 2024), or 1B in one step (NVIDIA UltraLong 2025), is enough to take a
model to 128K-4M with 100% NIAH. We spent ~1.0B tokens on extension, which is
squarely inside that window. What we did *not* do is build a base model worth
extending: our 4k pretrain saw **1.049B tokens for a 1.06B model = 0.99 tokens
per parameter**, against Chinchilla-optimal 20 and modern practice of 900-1900.

Every peer that scores 100% is a strong base model given a cheap extension. We
have a weak base model given a correctly-sized extension.

---

## 2. What we measured (this session, our harness, base checkpoints only)

All models scored with the SAME prompts, SAME haystack text, SAME substring-match
scorer. Peers were fed raw token ids as pure completions - no chat template - the
identical protocol our own base model gets, including the answer-prefix.

### niah_single (recall a 7-digit number given its key; guess floor ~1e-7)

| context | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Spark-X2.5-4B | **ours 1.06B** |
|---|---|---|---|---|---|---|
| 1,024  | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 96 |
| 2,048  | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 94 |
| 4,096  | 100.0 | 100.0 | 100.0 | 100.0 |  99.0 | 92 |
| 8,192  | 100.0 | 100.0 | 100.0 |  99.0 | 100.0 | 78 |
| 16,384 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | **48** |
| 32,736 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | - |

### val_backward (given the number, name the key; 35-word answer space, chance ~2.9%)

| context | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Spark-4B | **ours** |
|---|---|---|---|---|---|---|
| 1,024  | 36.0 | 79.0 | 69.0 | 78.0 | 72.0 | 1-3 |
| 4,096  | 31.0 | 68.0 | 56.0 | 61.0 | 67.0 | ~4 |
| 8,192  | 25.0 | 61.0 | 48.0 | 50.0 | 53.0 | ~2 |
| 16,384 | 23.0 | 56.0 | 51.0 | 58.0 | 50.0 | ~0 |
| 32,736 | 24.0 | 55.0 | 52.0 | 63.0 | 63.0 | - |

### Three facts these tables establish

1. **Scale is not the variable.** 0.6B -> 8B is a 13x parameter range with zero
   variation on niah_single. A 0.6B base model - smaller than ours - is at 100%
   at 32,736 where we are at 48% at 16,384. A 4B version of our model, trained
   our way, would not fix this.
2. **32k is not where anyone breaks.** Six lengths spanning 32x, five models, not
   one cell below 99. niah_single is saturated and has no remaining power.
3. **val_backward is the discriminating task.** It scales with parameters
   (0.6B ~24%, larger 50-63%), declines from 1k to 8k, then plateaus. We are at
   chance on it everywhere. This is our real benchmark from here on.

### Our own confirmed levers, for reference (measured earlier in this project)

| lever | effect at 16k | resolvable |
|---|---|---|
| YaRN cache correctness (`YARN_ORIG_MAXPOS` 1024 -> 4096) | 0% -> 27% (**+27pp**) | yes |
| training document length (flux median 826 -> b2 8-16k docs) | 33% -> 48% (**+15pp**) | yes |
| IN2 synthetic needles (arm C vs arm B) | +7.0pp vs a +-7.3pp bar | no |
| parameters | untested by us; **now measured externally as ~0** | - |

---

## 3. Why the peers max out: what the literature actually says

### 3.1 Context extension is cheap - this is the biggest correction

- **Fu et al., ICML 2024, "Data Engineering for Scaling Language Models to 128K
  Context"**: *500 million to 5 billion tokens are enough to enable the model to
  retrieve information anywhere within the 128K context*, and continual
  pretraining on 1B-5B tokens is "effective and affordable".
- **NVIDIA UltraLong (2025), "From 128K to 4M"**: **1B tokens, single epoch,
  one step**, from Llama-3.1-8B-Instruct. Result: **100% NIAH at every length and
  depth tested**, RULER 86.6 at <128K. 256 H100s, 5-13 hours.
- **ProLong (Princeton, ACL 2025)**: 20B @ 64K then 20B @ 512K = 40B total, from
  Llama-3-8B. Beats Llama-3.1-8B-Instruct on most long tasks using **5% of the
  long-context tokens**.

**We spent ~1.0B tokens on extension. That is not the deficiency.**

I previously told you Fu et al. required ~80B tokens and that we were "short by
40-400x" on extension data. That was wrong - the figure is 0.5-5B and we are
inside it. Corrected here.

### 3.2 What the extension data must look like

- **Qwen3 Technical Report**: the final pretraining stage extends to 32K using a
  long corpus that is **75% text of 16,384-32,768 tokens and 25% text of
  4,096-16,384 tokens**. This maps exactly onto our b3 and b2 buckets.
- **Fu et al.**: *naively upsampling longer data on certain domains like books
  gives suboptimal performance* - **domain balance matters as much as length
  upsampling**. Per-source length upsampling, not global book-heavy upsampling.
- **ProLong**: code repositories and books are the best long sources, but they
  **must be mixed with high-quality short data**; roughly 60-80% long / 20-40%
  short. Training at a sequence length *beyond* the evaluation length helps.
- **UltraLong**: downsample <4K, upsample >8K, concatenate to form long
  sequences; full attention across the concatenation boundary, no cross-document
  masking.

### 3.3 The base model is the precondition

Every recipe above starts from a model with trillions of tokens behind it:

| model | pretrain tokens | tokens/param |
|---|---|---|
| Qwen3 (all sizes) | ~36 T | ~9,000 (4B) |
| Llama-3.1-8B | ~15 T | ~1,875 |
| Phi-3.5-mini 3.8B | ~3.4 T | ~895 |
| **ours 1.06B** | **1.049 B** | **0.99** |

Qwen3's 32K stage is stage 3 of 3, after ~30T tokens of general pretraining and a
knowledge-intensive stage. The long-context ability is grafted onto a model that
already has the representations; the graft is cheap, the host is not.

### 3.4 Position-frequency: why even good models fall short of their window

**Xu et al. 2024, "Why Does the Effective Context Length of LLMs Fall Short?"**
diagnoses a **left-skewed frequency distribution of relative positions** formed
during pretraining: short relative distances are seen enormously more often than
long ones, so the model never learns to use distant positions well. Their fix
(STRING - shifting well-trained positions over the untrained ones at inference)
gains **>10 points on RULER for Llama-3.1-70B and Qwen2-72B with no training**.

This is the mechanism behind our own measured doc-length result. Our flux corpus
had median document length 826 and p90 2,721; almost no training example ever
required a long relative position. Our b1/b2 experiment moved retrieval +15pp by
changing exactly that distribution - the intercept moved, but the slope did not
(-29/-28/-30 pp per doubling across all three arms), because 210M tokens of
long-document exposure is not enough to reshape the position distribution laid
down over a 1.05B-token pretrain.

### 3.5 Synthesising long data when you do not have it

**NExtLong (ICML 2025)** decomposes documents into meta-chunks and extends them by
**interleaving hard negative distractors retrieved from the pretraining corpus**,
beating models trained on real long documents on HELMET and RULER.

This is the correct version of what our IN2 arms attempted. Our needles were
**random** 7-digit values in unrelated filler - no semantic hard negatives - which
is very likely why arm C came out at +7.0pp against a +-7.3pp bar. Retrieval with
random distractors is a much easier and less transferable task than retrieval
against semantically confusable ones.

### 3.6 Progressive vs one-step extension - the literature disagrees

- **Llama 3** used **six stages** from 8K to 128K over ~800B tokens, advancing
  only when *short-context performance recovered completely AND the model solved
  needle-in-a-haystack perfectly up to that length*.
- **UltraLong** ablated it and found **one step better**: their one-step 1M model
  scored 85.63 / 82.28 / 80.17 against the two-step variant's 84.22 / 79.83 /
  77.52.

Reading: with a strong base and a good mixture, one step is fine and cheaper.
Progressive staging is insurance for weaker bases or aggressive scale factors.
**Llama 3's advancement gate is the part to copy regardless** - it is exactly the
failure mode we measured, where the mismatched cache improved 16k (0 -> 14-17%)
while collapsing 4k (74% -> 16%). Perplexity at the new length would not have
caught that.

---

## 4. What we are missing, ranked

| # | gap | evidence | fixable on our hardware? |
|---|---|---|---|
| 1 | **Base model undertrained ~20-900x** | 0.99 tok/param vs 895-9,000 | partially - see 5.1 |
| 2 | **Extension data mixture unbalanced** | Qwen3 75/25, Fu domain balance; ours was single-bucket | yes, cheap |
| 3 | **No hard-negative synthetic long data** | NExtLong; our IN2 used random needles -> null | yes, cheap |
| 4 | **No short-context replay in the extension** | ProLong 20-40% short; our arms were 100% long | yes, free |
| 5 | **No per-stage retrieval gate** | Llama 3 gates on NIAH + short recovery | yes, free |
| 6 | Position-frequency skew | Xu et al. STRING | yes, inference-time |
| 7 | Extension token budget | 1.0B, inside Fu's 0.5-5B | **already adequate** |

---

## 5. Strategy for the 4B configuration

### 5.1 The hard constraint: we cannot pretrain a 4B from scratch

C = 6ND with our measured throughput (548 TFLOP/s aggregate on 4x H200):

| budget | tokens | 8x RTX 6000 Ada | 4x H100 | 8x H100 |
|---|---|---|---|---|
| Chinchilla 20 tok/param | 79 B | 67 d | 13.4 d | 6.7 d |
| modest 100 tok/param | 393 B | 336 d | 67 d | 34 d |
| Phi-3.5 class ~900 | 3,540 B | 3,021 d | 604 d | 302 d |

And our whole corpus is 99.84B tokens - **25 tok/param for a 4B, barely
Chinchilla, once, with zero repetition**. A competitive-from-scratch 4B is not
reachable with this data or this hardware.

**Therefore split the research question from the capability question:**

- **Track A (research, cheap, publishable):** do long-context method work on
  **Qwen3-4B-Base** as the host. It is 4B, open, base, native 32K - the exact
  configuration we are targeting. Every extension recipe in section 3 assumes a
  host like this. This is the only way to test *our* long-context ideas at 128K+
  in weeks rather than months.
- **Track B (our own model, honest scope):** keep bluscriptCP's 1.06B/4B as a
  *systems* contribution (CP, ring attention, YaRN, Muon+ZeRO) and report its
  long-context numbers against its own ablations, not against Qwen3.

### 5.2 Step-by-step plan

**Step 0 - fix the benchmark (1 day, no GPU).**
niah_single is saturated; stop reporting it as the headline. Promote
`val_backward`, `niah_multikey`, `niah_multivalue`, `vt` - all already implemented
in `longeval_gen.py` and currently unused. Add RULER's aggregate so numbers are
comparable to published tables. Gate: peers must NOT be at 100% on the new suite.

**Step 1 - build the extension corpus properly (2-3 days, CPU).**
Target ~4B tokens for a 32K stage, mirroring Qwen3:
- 75% b3 (16-32k)  = 3.03B - we have exactly this
- 25% b2 (8-16k)   = 1.01B
Then for a 128K stage, ~2B tokens:
- b4+b5+b6 real long data = 2.26B total (all we have above 32k)
- plus NExtLong-style synthesis from b1/b2 with **retrieved hard negatives**, to
  make up the volume without recycling 7,715 documents
- **20-40% short replay** from b0 mixed into every stage (ProLong)
- per-source domain balance, not global length upsampling (Fu et al.)

**Step 2 - one-step 32K extension on Qwen3-4B-Base (4-8 h on 4x H100).**
1B tokens, `YARN_ORIG_MAXPOS` = the host's true 32768, scale exactly
`target/orig`. Validate the invariant `scale * orig == seq_len` in code - this is
the bug that cost us 27pp.

**Step 3 - the Llama-3 gate, at every stage.**
Advance only when (a) NIAH is ~100% up to the new length AND (b) all shorter
lengths have not regressed. Our mismatched-cache experiment is the proof this
matters: 16k improved while 4k collapsed 74% -> 16%.

**Step 4 - 128K stage (one step, ~1-2B tokens, ~4 h on 4x H100).**
YaRN factor 4 from 32768. Compare one-step against 32K->64K->128K progressive -
the literature disagrees and our data would settle it for this host.

**Step 5 - port the winning recipe to bluscriptCP's 4B** as the systems
demonstration, with the honest caveat that its base is undertrained.

### 5.3 Cost summary

| stage | tokens | 8x Ada | 4x H100 |
|---|---|---|---|
| 32K extension | 1 B | 20.5 h | 4.1 h |
| 128K extension | 2 B | 41 h | 8.2 h |
| full eval suite (5 models x 6 lengths) | - | ~1 h | - |

The entire Track A programme is **days**, not months, because we are no longer
paying for the base model.

---

## 6. Corrections to my own earlier claims in this project

Recorded because they materially changed the recommendation:

1. **"Fu et al. need ~80B tokens; we are short by 40-400x on extension data."**
   Wrong. The figure is **0.5-5B**, and our ~1.0B is inside it. Extension budget
   was never the bottleneck.
2. **"Post-training substantially improves long-context retrieval."** Overstated.
   Paired base-vs-instruct on RULER: Mistral 7B +0.9, LWM 7B +5.1, Together 7B
   +2.1 on the 13-task average. Modest, and Mistral *loses* ground at 32k/64k.
3. **"A 0.626B ladder rung would tell us whether scale helps."** It would have
   cost days. The Qwen3 base ladder answered it for free: **13x parameters, zero
   change** on niah_single.
4. **"No 4B model has effective 128k, so 32k is our honest ceiling."** Still true
   for *effective* context under RULER's threshold, but UltraLong reaches 100%
   NIAH at 4M from an 8B host on 1B tokens - so the ceiling is a property of the
   host and the recipe, not of 4B as a size.
5. **Earlier chat-template check was not diagnostic.** I looked for a separate
   `chat_template.jinja`; Qwen ships the template inside `tokenizer_config.json`
   for base *and* instruct. Base status was confirmed instead by the model card
   ("Training Stage: Pretraining") and by generation behaviour - the peers
   continue the document after answering, which is base behaviour.

---

## 7. Open questions and risks

- **Is NIAH-style synthetic retrieval data in Qwen3's pretraining?** Qwen3 states
  synthetic data from Qwen2.5-Math and Qwen2.5-Coder, but does not disclose
  retrieval synthetics. If present, 100% partly reflects task familiarity rather
  than general retrieval - another argument for the harder task suite.
- **Spark-X2.5's 1M claim is untested.** At ~71 min/prompt for 1M on one Ada it
  cannot be falsified on this hardware at any sane N. It matches Qwen3-8B at 32k
  on val_backward (63.0 vs 63.0) with 27 of 36 layers windowed at 512.
- **Track A depends on an external base model.** If the goal is a fully in-house
  model, Track A results transfer as *method*, not as a shippable checkpoint.
- **Our corpus is CommonCrawl-derived (DCLM+BETR).** ProLong and Fu both stress
  code repositories and books as the best long sources; we have neither.

---

## 8. References

- Fu et al., *Data Engineering for Scaling Language Models to 128K Context*, ICML 2024. arXiv:2402.10171
- Gao et al., *How to Train Long-Context Language Models (Effectively)* (ProLong), ACL 2025. arXiv:2410.02660
- Xiong et al. / Meta, *The Llama 3 Herd of Models*, 2024. arXiv:2407.21783
- NVIDIA, *From 128K to 4M: Efficient Training of Ultra-Long Context LLMs*, 2025. arXiv:2504.06214
- Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context Language Models?*, 2024. arXiv:2404.06654
- Xu et al., *Why Does the Effective Context Length of LLMs Fall Short?*, 2024. arXiv:2410.18745
- *NExtLong: Toward Effective Long-Context Training without Long Documents*, ICML 2025. arXiv:2501.12766
- Qwen Team, *Qwen3 Technical Report*, 2025. arXiv:2505.09388
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*, 2023. arXiv:2309.00071

## 9. Artifacts from this session

- `~/peer_eval/peer_out/results/{1024,2048,4096,8192,16384,32736}/summary.csv` on 10.101.0.203
- `Tests/bluscriptcp/peer/{peer_gen,peer_run,peer_score,make_haystack,patch_spark,patch_spark_sdpa}.py`
- `Tests/bluscriptcp/peer/{run_peer_eval,prompt_all}.sh`
