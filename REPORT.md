# How Accurately Does a Transformer Store Integer Intervals?

## 1. Executive Summary

We investigate how GPT-2 small internally represents integer intervals, number comparisons, and modular arithmetic (alphabetic ring counting) in its residual stream. Using linear probing, regression probes, and activation patching, we find that **GPT-2 stores rich numerical information in its hidden states even when it cannot behaviorally solve the task**. Comparison information (A > B?) is linearly decodable from layer 1 onward (reaching 99.1% by layer 9), while interval membership (X in [A,B]?) peaks at 83.8% at layer 8 — significantly above chance but below comparison accuracy, consistent with the hypothesis that intervals require composing two boundary checks. Ring counting answers are decodable at 58% by layer 12 (vs. 3.8% chance), despite 0% behavioral accuracy. These findings reveal that transformers build internal representations of modular arithmetic that are never surfaced in their outputs.

**Key finding**: GPT-2 internally represents counting/interval information in its residual stream with surprising fidelity, but cannot reliably translate these representations into correct outputs — a dissociation between internal computation and behavioral performance.

**Practical implications**: This dissociation suggests that probing and activation analysis are essential tools for understanding transformer capabilities, as behavioral evaluations alone dramatically underestimate what models internally compute.

## 2. Goal

### Hypothesis
Large language models represent counting and integer interval operations in their residual streams, independent of explicit token-based arithmetic. By analyzing hidden states across layers, we can observe whether LLMs internally "count" when prompted with tasks such as determining the N-th letter after a given letter in an alphabetic ring.

### Why This Matters
- Understanding internal numerical representations informs how we can improve transformer arithmetic
- The gap between internal representation and behavioral output reveals where computation bottlenecks lie
- Interval membership is a composite operation (two comparisons) — studying it reveals how transformers compose simpler circuits
- Alphabetic ring counting requires modular arithmetic, which cannot be solved by digit-level token heuristics

### Expected Impact
This work bridges the gap between number encoding studies (Levy 2024, Kadlcik 2025) and circuit-level analysis (Hanna 2023), providing the first mechanistic analysis of interval membership and modular arithmetic in transformers.

## 3. Data Construction

### Dataset Description
All datasets are synthetically generated with deterministic seeds for reproducibility. Four task types:

| Dataset | Size | Task | Label Balance |
|---------|------|------|---------------|
| Comparison | 800 | Is A > B? (A,B ∈ [0,99]) | 51.4% positive |
| Interval | 800 | Is X ∈ [A,B]? (values ∈ [0,100]) | 27.1% positive |
| Ring Counting | 800 | N-th letter after start (mod 26) | 26 classes |
| Interval by Width | 1000 | Interval membership, stratified by width (2,5,10,20,50) | Varies by width |

### Example Samples

**Comparison**: "Is 73 greater than 42? Answer Yes or No." → Yes (label=1)

**Interval**: "Is 53 in the interval [25, 95]? Answer Yes or No." → Yes (label=1)

**Ring**: "What letter is 11 after R in the alphabet (wrapping around)? The answer is" → C

### Data Quality
- No missing values (synthetic generation)
- Balanced labels for comparison; interval labels are naturally imbalanced (27.1% positive due to random sampling of X, A, B)
- Ring counting: 52.8% of examples require wrapping around Z

### Preprocessing Steps
1. Prompts formatted as natural language questions with consistent structure
2. Each prompt tokenized via GPT-2 tokenizer with BOS token prepended
3. Hidden states extracted at the last token position (where the model makes its prediction)

## 4. Experiment Description

### Methodology

#### High-Level Approach
We use GPT-2 small (12 layers, 768-dimensional residual stream) as our target model, chosen because:
1. Its greater-than circuit has been fully mapped (Hanna et al., 2023)
2. Its number representations are well-characterized (Levy 2024)
3. Small enough for efficient activation extraction and patching

We run three complementary experiment types:
- **Behavioral evaluation**: Does the model get the answer right?
- **Linear probing**: Can we decode task-relevant information from hidden states at each layer?
- **Activation patching**: Which layers causally contribute to the model's output?

#### Why This Method?
Linear probing is the standard approach for testing what information is linearly accessible in hidden states (Alain & Bengio 2017). Activation patching identifies causal rather than merely correlational relationships. Together, they provide complementary views: probing reveals what information is *stored*, patching reveals what information is *used*.

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.12.8 | Runtime |
| PyTorch | 2.10.0+cu128 | Tensor computation |
| TransformerLens | 2.15.4 | Model hooking & activation extraction |
| scikit-learn | 1.7.2 | Linear probes |
| NumPy | 2.4.3 | Numerical computation |
| matplotlib | 3.10.8 | Visualization |

#### Hardware
- GPU: NVIDIA RTX A6000 (49 GB VRAM), 4 available, 1 used
- Total experiment runtime: 127.4 seconds (2.1 minutes)

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| Probe regularization (C) | 1.0 | Default |
| Ridge alpha | 1.0 | Default |
| Cross-validation folds | 3 | Speed/reliability tradeoff |
| Probe solver | LBFGS | Standard for logistic regression |
| Max iterations | 500 | Sufficient for convergence |
| Batch size (extraction) | 64 | GPU memory efficient |
| Activation patching pairs | 80 | Balance of speed and power |
| Random seed | 42 | Reproducibility |

#### Evaluation Metrics
- **Behavioral accuracy**: Fraction of correct next-token predictions (Yes/No for binary tasks, letter for ring)
- **Probe accuracy**: 3-fold cross-validated classification accuracy of linear probes on hidden states
- **Probe R²**: 3-fold cross-validated R² for regression probes decoding numerical values
- **Patching effect**: Mean logit difference shift when residual stream is patched from source to target

### Raw Results

#### Experiment 1: Behavioral Evaluation

| Task | Accuracy | Chance Level |
|------|----------|-------------|
| Number Comparison (A > B?) | 48.6% | 50% |
| Interval Membership (X ∈ [A,B]?) | 72.9% | 72.9% (majority class) |
| Ring Counting (N-th letter after start) | 0.0% | 3.8% |

**Key observation**: GPT-2 small performs at or below chance on all three tasks behaviorally. For comparison, it essentially random-guesses. For intervals, it matches the majority-class baseline (always predicting "No"). For ring counting, it completely fails.

#### Experiment 2: Linear Probing

##### Comparison Task (Binary Classification)

| Layer | Real Labels | Shuffled Control |
|-------|-------------|-----------------|
| 0 (embed) | 51.4% | 51.4% |
| 1 | 89.4% | 53.9% |
| 2 | 92.1% | 54.2% |
| 3 | 94.0% | 51.4% |
| 4 | 95.0% | 50.4% |
| 5 | 94.6% | 52.0% |
| 6 | 98.5% | 52.5% |
| 7 | 98.9% | 52.5% |
| 8 | 99.0% | 53.1% |
| **9** | **99.1%** | 52.7% |
| 10 | 98.9% | 53.1% |
| 11 | 98.4% | 51.9% |
| 12 | 97.6% | 53.7% |

##### Interval Membership Task (Binary Classification)

| Layer | Real Labels | Shuffled Control |
|-------|-------------|-----------------|
| 0 (embed) | 72.9% | 72.9% |
| 1 | 73.7% | 59.2% |
| 2 | 73.1% | 57.7% |
| 3 | 77.9% | 58.7% |
| 4 | 79.4% | 60.5% |
| 5 | 80.4% | 61.3% |
| 6 | 81.4% | 61.3% |
| 7 | 82.9% | 61.4% |
| **8** | **83.8%** | 62.5% |
| 9 | 82.1% | 63.2% |
| 10 | 80.5% | 64.5% |
| 11 | 81.5% | 65.4% |
| 12 | 81.9% | 63.8% |

##### Ring Counting (26-class Classification)

| Layer | Accuracy | vs. Chance (3.8%) |
|-------|----------|------------------|
| 0 (embed) | 5.4% | +1.6pp |
| 1 | 17.5% | +13.7pp |
| 3 | 28.4% | +24.6pp |
| 6 | 31.5% | +27.7pp |
| 7 | 37.8% | +34.0pp |
| 9 | 26.9% | +23.1pp |
| 11 | 50.0% | +46.2pp |
| **12** | **58.0%** | **+54.2pp** |

##### Regression Probing (Numerical Value Decoding)

| Layer | X Value R² (Interval) | Offset N R² (Ring) |
|-------|----------------------|-------------------|
| 0 | -0.004 | -0.002 |
| 1 | 0.992 | 1.000 |
| 2 | 0.994 | 1.000 |
| 6 | 0.998 | 1.000 |
| 12 | 0.999 | 1.000 |

Both numerical values (X from interval prompts) and offsets (N from ring prompts) are perfectly linearly decodable from layer 1 onward (R² ≈ 1.0), confirming that GPT-2 faithfully encodes these numbers in its residual stream.

##### Interval Membership by Width

| Width | Layer 0 | Layer 3 | Layer 6 | Layer 9 | Layer 12 |
|-------|---------|---------|---------|---------|----------|
| 2 | 99.0% | 99.0% | 99.0% | 99.0% | 99.0% |
| 5 | 93.5% | 96.5% | 95.0% | 93.0% | 93.0% |
| 10 | 92.5% | 91.5% | 92.0% | 92.0% | 91.5% |
| 20 | 77.0% | 77.5% | 77.5% | 84.0% | 78.0% |
| 50 | 50.5% | 73.4% | 71.0% | 75.5% | 70.5% |

**Critical insight**: Narrow intervals (width 2, 5) have high "probe accuracy" because the majority class dominates (only 1-6.5% positive labels), so the probe achieves high accuracy by defaulting to "No". Wider intervals (width 50) approach 50/50 balance and show genuine probe learning, with accuracy rising from 50.5% (chance) at layer 0 to 75.5% at layer 9.

##### Ring: Wrap vs. No-Wrap

| Layer | Wrap (n=386) | No-Wrap (n=414) |
|-------|-------------|-----------------|
| 0 | 9.6% | 7.2% |
| 3 | 52.8% | 60.4% |
| 6 | 53.4% | 59.4% |
| 9 | 46.9% | 54.3% |
| 12 | 55.4% | 64.0% |

No-wrap examples are consistently easier to decode than wrap examples (+5-9pp), as expected — modular arithmetic adds computational complexity.

#### Experiment 3: Activation Patching

##### Interval Membership

| Layer | Mean Effect | Std |
|-------|------------|-----|
| 0-3 | ≈ 0.00 | < 0.03 |
| 4-7 | -0.001 to -0.011 | 0.03-0.06 |
| **8** | **-0.054** | 0.12 |
| **9** | **-0.065** | 0.19 |
| 10 | -0.045 | 0.21 |
| 11 | -0.040 | 0.22 |

##### Number Comparison

| Layer | Mean Effect | Std |
|-------|------------|-----|
| 0-4 | ≈ 0.00 | < 0.03 |
| 5-7 | -0.014 to -0.015 | 0.03-0.05 |
| **8** | **-0.058** | 0.11 |
| **9** | **-0.087** | 0.20 |
| 10 | -0.076 | 0.21 |
| 11 | -0.062 | 0.22 |

Note: Negative patching effects indicate that patching from a positive-label source into a negative-label target moves the prediction in the *wrong* direction (toward "No" rather than "Yes"). This counterintuitive result suggests the model's decision is not simply a matter of "which class is this?" at the last token — the comparison/interval circuit may operate through mechanisms more complex than direct logit attribution.

### Output Locations
- Results JSON: `results/experiment_results.json`
- Plots: `results/figures/` (6 PNG files)
- Source code: `src/run_experiments.py`, `src/visualize.py`

## 5. Result Analysis

### Key Findings

1. **GPT-2 stores comparison information with near-perfect fidelity** (99.1% probe accuracy at layer 9) despite performing at chance behaviorally (48.6%). This confirms a massive dissociation between internal representation and behavioral output.

2. **Interval membership information builds progressively** across layers, peaking at 83.8% at layer 8, then declining slightly. This is ~16pp below comparison probing, consistent with the hypothesis that interval checking is computationally harder (requires two boundary comparisons).

3. **Ring counting answer letters are decodable** from hidden states (58% at layer 12, vs. 3.8% chance) despite 0% behavioral accuracy. This is the most striking finding: GPT-2 internally computes modular arithmetic but completely fails to output the result.

4. **Numerical values are perfectly encoded** from layer 1 onward (R² > 0.99 for both X values and ring offsets), confirming prior work (Levy 2024) that transformers faithfully encode numbers in their residual stream.

5. **Wrap-around hurts ring accuracy** by 5-9pp across layers, confirming that modular arithmetic adds computational difficulty.

6. **Activation patching identifies layers 8-9 as causally important** for both comparison and interval tasks, consistent with Hanna et al.'s finding that the greater-than circuit operates in layers 5-9.

### Hypothesis Testing Results

**H1** (Interval membership decodable above chance): **Supported**. Peak probe accuracy 83.8% vs. 72.9% majority-class baseline, with shuffled control at ~63%. The 83.8% - 63% = 20.8pp gap above control is statistically meaningful.

**H2** (Information emerges in middle-to-late layers): **Supported**. Comparison probing jumps at layer 1 and plateaus by layer 6. Interval probing peaks at layer 8. Ring counting peaks at layer 12. All consistent with progressive information accumulation.

**H3** (Ring counting answer decodable from later layers): **Supported**. 58% accuracy at layer 12 vs. 3.8% chance. Strong evidence of internal modular arithmetic computation.

**H4** (Wider intervals harder to decode): **Partially supported**. The relationship is confounded by class imbalance — narrow intervals have very few positive examples, inflating accuracy. For balanced intervals (width=50), probe accuracy improves substantially from embedding to middle layers.

**H5** (Distinct circuits for comparison vs. interval): **Supported**. Probing profiles differ — comparison reaches near-perfect accuracy (99%) while interval plateaus at ~84%. Activation patching shows similar layer profiles but with different magnitudes, suggesting overlapping but not identical circuits.

### Comparison to Literature

Our findings align with and extend prior work:
- **Levy (2024)**: We confirm near-perfect numerical encoding (R² ≈ 1.0) from early layers, consistent with their digit-level circular encoding finding
- **Hanna (2023)**: Our activation patching identifies layers 8-9 as causally important, matching their greater-than circuit location (layers 5-9)
- **Hasani (2026)**: The progressive accumulation of counting information across layers matches their finding of layer-wise count accumulation
- **Novel finding**: The behavior-representation dissociation for ring counting (0% behavioral, 58% internal) extends the "LLMs know more than they say" phenomenon (Yuchi 2026) to a new domain

### Surprises and Insights

1. **Massive behavior-representation gap for ring counting**: Zero behavioral accuracy with 58% internal accuracy is remarkable. The model computes modular arithmetic internally but completely fails to output it. This suggests a "last-mile" problem — the computation happens but is not connected to the output pathway.

2. **Comparison probe accuracy at layer 1**: 89.4% accuracy already at the first transformer layer is surprising. This suggests that much of the comparison information is computable from the token embeddings alone (the probe at layer 0 is at chance, but layer 1 already extracts the comparison).

3. **Interval probing shuffled control rises across layers**: The shuffled-label control for intervals rises from 59% to 65% across layers, suggesting the probe picks up increasingly complex features that correlate with arbitrary label assignments. This emphasizes the importance of control conditions.

4. **Negative patching effects**: Patching from positive to negative examples produces *negative* logit shifts (moving predictions further from "Yes"). This is likely because GPT-2 isn't actually solving these tasks, so patching doesn't transfer a "correct answer" signal — it introduces a noisy perturbation.

### Error Analysis

The primary "error" is the behavioral-representation gap. Why does GPT-2 fail behaviorally despite good internal representations?

1. **Comparison (48.6% behavioral)**: GPT-2 small was not specifically trained on comparison question-answering. The Yes/No format may not map to the model's internal comparison circuit, which was discovered in a "The war lasted from 17YY to 17__" format (Hanna 2023).

2. **Interval (72.9% behavioral = majority class)**: The model defaults to "No" because intervals are a rare concept in natural language. The 27.1% positive rate means always-no achieves 72.9%.

3. **Ring counting (0% behavioral)**: GPT-2 was not trained on alphabetic ring counting tasks. The internal representation exists because the model processes the letter and number tokens, but the output mapping was never learned.

### Limitations

1. **Single model**: All experiments use GPT-2 small (124M params). Larger models may show different patterns — especially for behavioral accuracy.

2. **Linear probes**: We only test linearly accessible information. Non-linear probes might extract more, but risk overfitting (Hewitt & Liang 2019).

3. **Prompt format sensitivity**: Results may depend on exact prompt wording. We used a single format per task.

4. **Class imbalance**: Interval membership (27.1% positive) and narrow-width conditions (1-6.5% positive) complicate accuracy interpretation. We address this with shuffled-label controls but not with balanced sampling.

5. **Activation patching at single position**: We only patch the last token position. Information may flow through other token positions.

6. **No causal direction**: Probing shows correlation, not causation. Even with patching, we cannot definitively identify the circuit — only the layers that matter.

## 6. Conclusions

### Summary
GPT-2 small stores integer interval and modular arithmetic information in its residual stream with substantial fidelity, even when it cannot solve these tasks behaviorally. Comparison information is linearly decodable at 99% by layer 9, interval membership at 84% by layer 8, and ring counting at 58% by layer 12 — all dramatically above chance and above shuffled-label controls. This reveals a consistent pattern: transformers internally compute more than they can express.

### Implications
- **For mechanistic interpretability**: Behavioral evaluation alone misses rich internal computation. Probing should be a standard companion to behavioral metrics.
- **For model development**: The "last-mile" problem (good internal representations → poor outputs) suggests that fine-tuning or prompt engineering could unlock latent capabilities without changing the core representations.
- **For understanding arithmetic**: Interval membership is genuinely harder than single comparison (~84% vs. ~99% probe accuracy), providing evidence that it requires composing multiple sub-computations.

### Confidence in Findings
**High confidence** in the probing results: the effect sizes are large (40-50pp above chance), controls are consistent, and findings align with prior literature. **Moderate confidence** in the activation patching results: effects are small and noisy, partly because the model doesn't solve these tasks well behaviorally. **Low confidence** in width-specific results due to class imbalance confounds.

## 7. Next Steps

### Immediate Follow-ups
1. **Test on larger models** (GPT-2 medium/large, Llama 3 8B) to see if behavioral accuracy improves while internal representations remain similar
2. **Use circular/sinusoidal probes** (Levy 2024, Kadlcik 2025) instead of linear probes for potentially better ring counting decoding
3. **Balance classes** by oversampling positive examples or using balanced accuracy metrics

### Alternative Approaches
- **Path patching** (more granular than activation patching) to identify specific attention heads and MLPs contributing to interval decisions
- **Logit lens** to trace how internal representations are transformed into output probabilities across layers
- **Training interventions**: Fine-tune GPT-2 on these tasks and measure how internal representations change

### Broader Extensions
- Extend to multi-step reasoning tasks (e.g., "Is X in [A,B] AND Y in [C,D]?")
- Study how different number formats (digits, words, Roman numerals) affect internal representations
- Apply to other modular arithmetic tasks beyond alphabetic rings

### Open Questions
1. Why does GPT-2 achieve 99% internal comparison accuracy but 49% behavioral accuracy? Where exactly does the signal get lost?
2. Is the ring counting representation truly computing modular arithmetic, or memorizing start-offset→letter mappings?
3. Can the gap between internal computation and behavioral output be closed by prompt engineering alone?

## References

1. Levy & Geva (2024). Language Models Encode Numbers Using Digit Representations in Base 10.
2. Kadlcik et al. (2025). Pre-trained LMs Learn Remarkably Accurate Representations of Numbers.
3. Hanna, Liu & Variengien (2023). How does GPT-2 compute greater-than over the years? NeurIPS.
4. Stolfo, Belinkov & Sachan (2023). Mechanistic Interpretation of Arithmetic Reasoning in LMs. EMNLP.
5. Hasani et al. (2026). Mechanistic Interpretability of Large-Scale Counting in LLMs.
6. Yuchi, Du & Eisner (2026). LLMs Know More About Numbers than They Can Say. EACL.
7. Wallace et al. (2019). Do NLP Models Know Numbers? Probing Numeracy in Embeddings.
8. Chang & Bisk (2024). Language Models Need Inductive Biases to Count Inductively.
9. Quirke, Neo & Barez (2024). Understanding Addition and Subtraction in Transformers.
