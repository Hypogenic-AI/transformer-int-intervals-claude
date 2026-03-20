# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project: "How accurately does a Transformer store integer intervals?" Resources include 20 academic papers, 4 synthetic datasets, and 5 code repositories.

## Papers

Total papers downloaded: 20

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Language Models Encode Numbers Using Digit Representations in Base 10 | Levy, Geva | 2024 | papers/levy2024_number_representations_base10.pdf | Circular probes, base-10 digit encoding |
| 2 | Pre-trained LMs Learn Remarkably Accurate Representations of Numbers | Kadlcik et al. | 2025 | papers/kadlcik2025_accurate_number_representations.pdf | Sinusoidal probes, near-perfect decoding |
| 3 | LLMs Know More About Numbers than They Can Say | Yuchi, Du, Eisner | 2026 | papers/yuchi2026_llms_know_numbers.pdf | Log-magnitude probing, internal vs. verbalized accuracy |
| 4 | Number Representations in LLMs: Human Perception Parallel | AlQuBoj et al. | 2025 | papers/alquboj2025_number_representations_human_perception.pdf | Logarithmic mental number line |
| 5 | Do NLP Models Know Numbers? | Wallace et al. | 2019 | papers/wallace2019_probing_numeracy_embeddings.pdf | Seminal probing work, extrapolation failures |
| 6 | How does GPT-2 compute greater-than? | Hanna, Liu, Variengien | 2023 | papers/hanna2023_gpt2_greater_than.pdf | **Most relevant** — greater-than circuit discovery |
| 7 | Mechanistic Interpretation of Arithmetic Reasoning | Stolfo, Belinkov, Sachan | 2023 | papers/stolfo2023_mechanistic_arithmetic_reasoning.pdf | Causal mediation, three-phase processing |
| 8 | Understanding Addition and Subtraction in Transformers | Quirke, Neo, Barez | 2024 | papers/quirke2024_addition_subtraction_transformers.pdf | Carry/borrow circuits, small transformers |
| 9 | Interpreting Arithmetic Mechanism via Comparative Neuron Analysis | Yu, Ananiadou | 2024 | papers/yu2024_arithmetic_neuron_analysis.pdf | Four-stage logic chain |
| 10 | The Validation Gap | Bertolazzi et al. | 2025 | papers/bertolazzi2025_validation_gap_arithmetic.pdf | Computation vs. validation dissociation |
| 11 | Language Models are Symbolic Learners in Arithmetic | Deng et al. | 2024 | papers/deng2024_symbolic_learners_arithmetic.pdf | Symbolic shortcuts hypothesis |
| 12 | Mechanistic Interpretability of Large-Scale Counting | Hasani et al. | 2026 | papers/hasani2026_counting_mechanisms_llms.pdf | Progressive counting accumulation |
| 13 | Understanding Counting Mechanisms in LLMs and VLMs | Hasani et al. | 2025 | papers/hasani2025_counting_llm_vlm.pdf | CountScope probing |
| 14 | Language Models Need Inductive Biases to Count | Chang, Bisk | 2024 | papers/chang2024_inductive_biases_counting.pdf | Counting generalization failures |
| 15 | Counting Ability of LLMs and Impact of Tokenization | Zhang, Cao, You | 2024 | papers/zhang2024_counting_ability_tokenization.pdf | TC⁰ limitations, tokenization effects |
| 16 | Why Do LLMs Struggle to Count Letters? | Fu et al. | 2024 | papers/fu2024_llms_count_letters.pdf | Counting complexity correlates |
| 17 | States Hidden in Hidden States | Chen et al. | 2024 | papers/chen2024_states_hidden_in_hidden_states.pdf | Implicit discrete state representations |
| 18 | A Fragile Number Sense | Rahman, Mishra | 2025 | papers/rahman2025_fragile_number_sense.pdf | Fragile numerical reasoning |
| 19 | Compact Proofs via Mechanistic Interpretability | Gross et al. | 2024 | papers/gross2024_compact_proofs_mechanistic.pdf | Max-of-K formal verification |
| 20 | Mechanistic Interpretability for AI Safety — A Review | Bereska, Gavves | 2024 | papers/bereska2024_mechanistic_interpretability_review.pdf | MI methodology review |

See papers/README.md for detailed descriptions.

## Datasets

Total datasets: 4 (synthetic, generated deterministically)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Interval Membership | Synthetic | 1000+ samples | Is X ∈ [A, B]? | datasets/integer_intervals/ | Core experimental task |
| Alphabetic Ring | Synthetic | 500+ samples | N-th letter after start | datasets/integer_intervals/ | Modular arithmetic test |
| Number Comparison | Synthetic | 1000+ samples | Is A > B? | datasets/integer_intervals/ | Baseline comparison task |
| Range Containment | Synthetic | 1000+ samples | Is X between A and B? (varying widths) | datasets/integer_intervals/ | Width-dependent analysis |

See datasets/README.md for generation code and download instructions.

## Code Repositories

Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| base10 | github.com/amitlevy/base10 | Circular probes for digit encoding | code/base10/ | Levy 2024 |
| lm-arithmetic | github.com/alestolfo/lm-arithmetic | Causal mediation for arithmetic | code/lm-arithmetic/ | Stolfo 2023, EMNLP |
| gpt2-greater-than | github.com/hannamw/gpt2-greater-than | Greater-than circuit analysis | code/gpt2-greater-than/ | Hanna 2023, NeurIPS |
| numllama | github.com/prompteus/numllama | Sinusoidal number probes | code/numllama/ | Kadlcik 2025, EMNLP |
| numeracy-probing | github.com/VCY019/Numeracy-Probing | Log-magnitude probing | code/numeracy-probing/ | Yuchi 2026, EACL |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Primary search via arXiv API across 4 query sets covering number representation, mechanistic interpretability, counting mechanisms, and numerical probing
- Supplemented with Semantic Scholar (rate-limited) and GitHub search
- Paper-finder service was unavailable; manual API search conducted instead

### Selection Criteria
- Direct relevance to integer representation, comparison, or counting in transformers
- Mechanistic interpretability methodology applicable to interval membership
- Recent work (2019-2026) with preference for papers with code
- Foundational papers establishing probing methodology

### Challenges Encountered
- Paper-finder service timed out; used direct arXiv API instead
- Semantic Scholar rate-limited on all queries
- No existing work directly studies interval membership representation (confirming novelty)

### Gaps and Workarounds
- No pre-existing interval membership datasets → created synthetic datasets
- No interval-specific code → identified composable tools (greater-than circuits + probing code)
- GPT-2 lacks less-than circuit → experiment design must account for asymmetric comparison

## Recommendations for Experiment Design

1. **Primary model**: GPT-2 small (greater-than circuit already mapped by Hanna et al.) and Llama 3 8B (number representations characterized by Levy 2024)
2. **Baseline methods**: Linear probes (Wallace 2019), circular probes (Levy 2024), sinusoidal probes (Kadlcik 2025)
3. **Core methodology**: Path patching (Hanna 2023) to discover interval membership circuits, combined with probing to decode boundary information from hidden states
4. **Evaluation metrics**: Probe accuracy for boundary recovery, causal intervention effect (IE/RI), behavioral accuracy across interval widths
5. **Key tool**: TransformerLens (`pip install transformer_lens`) for activation extraction and intervention
6. **Code to adapt**: gpt2-greater-than (path patching infrastructure), base10 (circular probes), numllama (sinusoidal probes)
