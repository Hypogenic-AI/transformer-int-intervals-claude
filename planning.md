# Research Plan: How Accurately Does a Transformer Store Integer Intervals?

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding how transformers internally represent and manipulate numerical intervals is fundamental to mechanistic interpretability. While prior work has mapped circuits for single comparisons (greater-than) and characterized number encodings (base-10 digits, sinusoidal), no work has examined how models compose these primitives for interval membership or modular arithmetic. This directly impacts our understanding of whether LLMs develop genuine computational mechanisms vs. relying on token-level heuristics.

### Gap in Existing Work
Based on the literature review:
- Levy (2024) and Kadlcik (2025) show numbers are encoded as per-digit circular features, but don't study how these enable range checks
- Hanna et al. (2023) discovered GPT-2's greater-than circuit but found NO less-than circuit — interval membership requires both bounds
- Hasani et al. (2026) show counting accumulates across layers but didn't study modular arithmetic
- **No work examines interval membership or alphabetic ring counting mechanistically**

### Our Novel Contribution
1. First mechanistic study of how transformers represent **interval membership** (two-sided bounds) vs. single comparisons
2. First probing study of **modular arithmetic** (alphabetic ring counting) in residual streams
3. Characterization of how interval width, boundary proximity, and wrap-around affect internal representations
4. Layer-by-layer analysis revealing where interval/counting information emerges

### Experiment Justification
- **Experiment 1 (Behavioral)**: Establish GPT-2's task accuracy on comparison, interval membership, and ring counting to know what internal representations to probe for
- **Experiment 2 (Probing)**: Train linear probes on hidden states to decode numerical values, interval boundaries, and modular offsets — reveals WHAT information is stored WHERE
- **Experiment 3 (Causal)**: Activation patching to identify which layers/components causally contribute to interval judgments vs. simple comparisons

## Research Question
Can we observe counting-like processes and interval membership representations in a transformer's hidden states? Specifically, do LLMs internally represent modular arithmetic (ring counting) and two-sided interval checks in their residual streams, beyond token-level heuristics?

## Hypothesis Decomposition
H1: GPT-2 encodes numerical values in hidden states that allow linear probes to recover interval membership labels with above-chance accuracy
H2: Interval membership representations emerge in middle-to-late layers (layers 5-10 in GPT-2 small), consistent with the greater-than circuit location
H3: Alphabetic ring counting (modular arithmetic) is represented in hidden states, with the result letter decodable from later layers
H4: Interval width affects representation quality — wider intervals are easier to decode than narrow ones
H5: The model uses distinct but overlapping circuits for greater-than vs. interval membership

## Proposed Methodology

### Approach
Use GPT-2 small (12 layers, well-characterized) with TransformerLens for activation extraction. Three experiment types:

1. **Behavioral evaluation**: Measure GPT-2's next-token predictions on our tasks (does it get them right?)
2. **Linear probing**: Train probes on residual stream activations at each layer to decode task-relevant information
3. **Activation patching**: Swap activations between examples to identify causally important layers

### Experimental Steps
1. Generate synthetic datasets (comparison, interval membership, ring counting) — 2000 samples each
2. Extract hidden states from GPT-2 small at all 13 residual stream positions (embedding + 12 layers)
3. Train linear probes per layer to predict: (a) the numerical value of X, (b) interval membership label, (c) ring counting answer
4. Measure probe accuracy by layer to map information flow
5. Perform activation patching: swap residual stream at layer L between an "in-interval" and "out-of-interval" example, measure effect on output
6. Analyze results by interval width, boundary proximity, and wrap-around condition

### Baselines
- Random baseline (chance-level probing)
- Majority class baseline
- Probing on shuffled labels (control for memorization)
- Single comparison (greater-than only) as simpler task baseline

### Evaluation Metrics
- **Behavioral accuracy**: % correct next-token predictions
- **Probe accuracy**: Classification accuracy of linear probes per layer
- **Probe R²**: For regression probes (decoding numerical values)
- **Patching effect**: Change in output logit difference when activations are patched

### Statistical Analysis Plan
- Bootstrap confidence intervals (95%) for all accuracy measures
- Paired t-tests comparing probe accuracy across layers (with Bonferroni correction)
- Effect size (Cohen's d) for patching experiments
- Significance level: α = 0.05

## Expected Outcomes
- Probe accuracy should increase from early to middle layers (information accumulation), possibly plateau or decrease in late layers
- Interval membership should be harder to decode than single comparison (requires composing two checks)
- Ring counting accuracy should peak in later layers (requires modular arithmetic computation)
- Wider intervals should yield higher probe accuracy (more margin for error)

## Timeline and Milestones
1. Environment setup + data generation: 10 min
2. Hidden state extraction: 15 min
3. Probing experiments: 20 min
4. Behavioral evaluation: 10 min
5. Activation patching: 20 min
6. Analysis + visualization: 20 min
7. Documentation: 20 min

## Potential Challenges
- GPT-2 small may not perform well on these tasks behaviorally → probing can still reveal latent representations
- Tokenization of numbers may confound results → use single/double digit numbers to keep tokens simple
- Linear probes may overfit on small datasets → use cross-validation and shuffled-label controls

## Success Criteria
1. Probe accuracy significantly above chance for at least one task at some layers
2. Clear layer-wise pattern showing information emergence
3. Distinct patterns between comparison, interval, and ring tasks
4. At least one causal patching result showing significant effect
