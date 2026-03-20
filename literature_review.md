# Literature Review: How Accurately Does a Transformer Store Integer Intervals?

## Research Area Overview

This review covers three intersecting areas of research relevant to understanding how Transformers internally represent and compute integer intervals: (1) how LLMs encode numbers in their hidden states, (2) mechanistic interpretability of arithmetic circuits, and (3) counting mechanisms and their limitations.

The central hypothesis is that LLMs may represent counting or integer interval operations in their residual streams, independent of explicit token-based arithmetic. The literature reveals substantial progress in understanding number representation and basic arithmetic circuits, but a gap in studying composite operations like interval membership.

## How Numbers Are Represented Internally

### Digit-Level Base-10 Encoding
Levy & Geva (2024) demonstrate that LLMs (Llama 3 8B, Mistral 7B) encode numbers using **per-digit circular features in base 10**, not as scalar values. Circular probes targeting `cos(2πt/10)` and `sin(2πt/10)` per digit position achieve 91-100% accuracy in reconstructing numbers from hidden states. Causal interventions (flipping a digit's representation by rotating ±5 mod 10) confirm these representations are functionally used. This holds regardless of tokenization (Llama uses whole-number tokens for 0-999; Mistral uses single-digit tokens). **Implication for intervals**: Checking if X ∈ [A, B] cannot be a simple threshold on a scalar — it must compose per-digit comparisons.

### Sinusoidal / Fourier Structure
Kadlcik et al. (2025) show that number embeddings exhibit **wave-like sinusoidal patterns** detectable via Fourier-basis probes. Their sinusoidal probe (`f_sin`) achieves 94-100% accuracy on integers 0-999 across Llama 3, Phi 4, and OLMo 2 families — dramatically outperforming linear probes used in prior work. Embedding precision predicts arithmetic accuracy. **Implication**: The periodic basis functions underlying integer representations could enable interval membership through simple operations in Fourier space.

### Logarithmic Mental Number Line
AlQuBoj et al. (2025) and Yuchi et al. (2026) independently find that LLMs encode numbers on a **logarithmic/sublinear scale** — smaller numbers have higher resolution. Yuchi et al. show linear probes recover log₂-magnitudes with ~2.3% median relative error, and comparison classifiers achieve >90% accuracy from hidden states even when verbalized accuracy is only 50-70%. **Implication**: Interval boundaries near zero will be more precisely represented than those at larger magnitudes.

### Foundational Work
Wallace et al. (2019) established that even early embeddings (GloVe, ELMo, BERT) encode numerical magnitude, but models **cannot extrapolate** to numbers outside the training range — a critical limitation for interval generalization.

## Mechanistic Interpretability of Arithmetic

### The Greater-Than Circuit (Most Directly Relevant)
Hanna et al. (NeurIPS 2023) identify the **exact circuit** GPT-2 small uses for numerical comparison. Given "The war lasted from 17YY to 17__":
- **Attention heads** (layers 5-9) attend to YY and communicate its value to downstream components
- **MLPs 8-11** at the last token position compute the greater-than operation by producing an **upper-triangular pattern** in logit space (upweighting all years > YY)
- The computation is **distributed across ~100+ neurons** that compose additively — no single neuron implements greater-than
- The circuit achieves 89.5% sufficiency (explaining most of the model's behavior)
- **Critical finding**: GPT-2 has NO less-than circuit and overgeneralizes greater-than inappropriately

**Implication for intervals**: Interval membership [A, B] requires TWO boundary checks. Since greater-than and less-than appear to use different circuits, interval storage may require composing distinct mechanisms — or a qualitatively different approach.

### Three-Phase Arithmetic Processing
Stolfo et al. (EMNLP 2023) use causal mediation analysis on GPT-J, Pythia, LLaMA, and Goat to identify three phases: (A) early MLPs at operand positions encode operands, (B) middle-layer attention transfers information to the last token, (C) late MLPs (layers 19-20 in GPT-J) compute the result. The result-computing MLPs show 40% Relative Importance when the result varies, dropping to 4-7% when it's fixed. Distinct neuron populations handle arithmetic vs. factual retrieval.

### Addition/Subtraction Circuits
Quirke et al. (2024) show small transformers learn identical cascading carry/borrow algorithms for n-digit arithmetic, achieving 99.999% accuracy. Yu & Ananiadou (2024) identify a four-stage logic chain (feature enhancing → transferring → predicting → prediction enhancing) with specialized attention heads per operation.

### Symbolic vs. Algorithmic Processing
Deng et al. (2024) argue LMs learn **symbolic shortcuts** rather than true algorithms, using pattern matching that breaks on out-of-distribution inputs. This raises the question of whether interval membership is similarly learned as pattern matching.

## Counting Mechanisms and Limitations

### Progressive Layer-wise Accumulation
Hasani et al. (2026) show counting information **accumulates progressively across layers** — lower layers encode small counts, higher layers encode larger ones. Count information is localized at boundary tokens (final item, separators) in middle-to-late layers (70-80% of model depth). Numbers are encoded in a **compressed, sublinear** manner. Accuracy degrades sharply beyond ~10 items for direct counting; System-2 decomposition (partitioning + CoT) is required for larger counts.

### Architectural Limitations
Zhang et al. (2024) note transformers are limited to TC⁰ complexity class, making deep counting theoretically difficult. Chang & Bisk (2024) show transformers cannot generalize counting inductively (OOD) without positional embeddings. These limitations likely extend to interval-related reasoning.

## Common Methodologies

| Method | Used In | Description |
|--------|---------|-------------|
| Circular / sinusoidal probes | Levy 2024, Kadlcik 2025 | Train probes on hidden states targeting periodic features |
| Linear probes | Wallace 2019, Yuchi 2026 | Linear regression/classification on hidden states |
| Causal mediation / path patching | Stolfo 2023, Hanna 2023 | Intervene on components to measure causal contribution |
| Activation patching | Hasani 2025, 2026 | Swap activations between contexts to test information flow |
| CountScope | Hasani 2025, 2026 | Specialized probing for numerical information in hidden states |
| Logit lens | Hanna 2023 | Project intermediate representations to vocabulary space |
| Ablation studies | Quirke 2024, Hanna 2023 | Zero-out or corrupt components and measure impact |

## Standard Baselines and Evaluation Metrics

**Baselines**: Linear probes, random baselines, corrupted-input controls, logit lens
**Metrics**: Probe accuracy (exact digit match), causal intervention effect (IE/RI), probability difference, MAE, cutoff sharpness

## Gaps and Opportunities

1. **No existing work studies interval membership representation directly.** Individual number encoding and one-sided comparison are well-studied, but the composite operation "is X in [A, B]?" has not been mechanistically analyzed.

2. **Greater-than exists but less-than does not** (in GPT-2). How models compose two boundary checks for intervals is unknown.

3. **The interaction between digit-level encoding and comparison circuits is unexplored.** Levy (2024) shows digit-level encoding; Hanna (2023) shows comparison circuits. How comparison circuits operate on digit-level representations is not understood.

4. **Interval width effects are unstudied.** Do narrow intervals have different representations than wide ones? Does logarithmic encoding cause systematic biases?

5. **The alphabetic ring counting task** (N-th letter after a given letter) has not been studied from a mechanistic interpretability perspective, despite being a natural test of modular arithmetic in residual streams.

## Recommendations for Our Experiment

### Recommended Models
- **GPT-2 small** (12 layers): Best understood mechanistically, greater-than circuit already mapped
- **Llama 3 8B**: Number representations well-characterized (Levy 2024, Kadlcik 2025)
- **Pythia 2.8B**: Good balance of size and interpretability (Stolfo 2023)

### Recommended Methodology
1. **Probing**: Use circular/sinusoidal probes (Levy 2024, Kadlcik 2025) to decode interval boundary information from hidden states at each layer
2. **Causal intervention**: Use path patching (Hanna 2023) to identify which components contribute to interval membership decisions
3. **Comparative analysis**: Compare circuits for interval membership vs. simple greater-than to understand composition

### Recommended Datasets
- Synthetic interval membership queries (controllable complexity)
- Alphabetic ring counting (modular arithmetic, per the hypothesis)
- Number comparison pairs (baseline task)
- Varying interval widths to test resolution effects

### Recommended Metrics
- Probe accuracy for boundary decoding
- Causal intervention effect (IE) for component identification
- Behavioral accuracy across interval widths and number ranges
- Comparison of interval boundary precision vs. single-number precision

### Key Tools
- **TransformerLens**: For hooking into model internals (`pip install transformer_lens`)
- **base10 repo**: For circular probe implementation
- **gpt2-greater-than repo**: For path patching infrastructure
- **numllama repo**: For sinusoidal probe architecture
