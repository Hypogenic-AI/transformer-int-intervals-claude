# Downloaded Papers

## Core Papers — Number Representation in Transformers

1. **Language Models Encode Numbers Using Digit Representations in Base 10** (levy2024_number_representations_base10.pdf)
   - Authors: Levy, Geva (2024)
   - arXiv: 2410.11781
   - Why relevant: Shows LLMs encode numbers as per-digit circular features in base 10, not as scalar values. Foundational for understanding interval boundary encoding.

2. **Pre-trained Language Models Learn Remarkably Accurate Representations of Numbers** (kadlcik2025_accurate_number_representations.pdf)
   - Authors: Kadlcik, Stefanik, Mickus, Spiegel, Kuchar (2025)
   - arXiv: 2506.08966
   - Why relevant: Sinusoidal probes achieve near-perfect number decoding from embeddings. Shows Fourier-basis structure in number representations.

3. **LLMs Know More About Numbers than They Can Say** (yuchi2026_llms_know_numbers.pdf)
   - Authors: Yuchi, Du, Eisner (2026)
   - arXiv: 2602.07812
   - Why relevant: Internal representations encode log-magnitudes more accurately than verbalized outputs. Linear probes recover number comparisons from hidden states.

4. **Number Representations in LLMs: A Computational Parallel to Human Perception** (alquboj2025_number_representations_human_perception.pdf)
   - Authors: AlQuBoj et al. (2025)
   - arXiv: 2502.16147
   - Why relevant: LLMs encode numbers on a logarithmic scale mirroring human mental number line.

5. **Do NLP Models Know Numbers? Probing Numeracy in Embeddings** (wallace2019_probing_numeracy_embeddings.pdf)
   - Authors: Wallace, Wang, Li, Singh, Gardner (2019)
   - arXiv: 1909.07940
   - Why relevant: Seminal paper establishing that embeddings encode numerical magnitude. Identifies extrapolation failures.

## Core Papers — Arithmetic Circuits & Mechanistic Interpretability

6. **How does GPT-2 compute greater-than?** (hanna2023_gpt2_greater_than.pdf)
   - Authors: Hanna, Liu, Variengien (NeurIPS 2023)
   - arXiv: 2305.00586
   - Why relevant: **Most directly relevant.** Identifies the exact circuit for numerical comparison in GPT-2 — attention heads route year info, MLPs 8-11 compute step function. Greater-than is the core operation for interval boundary checking.

7. **A Mechanistic Interpretation of Arithmetic Reasoning in LMs** (stolfo2023_mechanistic_arithmetic_reasoning.pdf)
   - Authors: Stolfo, Belinkov, Sachan (EMNLP 2023)
   - arXiv: 2305.15054
   - Why relevant: Causal mediation analysis identifies three-phase arithmetic processing: operand encoding → attention transfer → result computation in late MLPs.

8. **Understanding Addition and Subtraction in Transformers** (quirke2024_addition_subtraction_transformers.pdf)
   - Authors: Quirke, Neo, Barez (2024)
   - arXiv: 2402.02619
   - Why relevant: Mechanistic account of cascading carry/borrow circuits. Shows small transformers converge to same algorithmic solution.

9. **Interpreting Arithmetic Mechanism in LLMs through Comparative Neuron Analysis** (yu2024_arithmetic_neuron_analysis.pdf)
   - Authors: Yu, Ananiadou (2024)
   - arXiv: 2409.14144
   - Why relevant: Four-stage logic chain for arithmetic with specialized attention heads.

10. **The Validation Gap** (bertolazzi2025_validation_gap_arithmetic.pdf)
    - Authors: Bertolazzi, Mondorf, Plank, Bernardi (2025)
    - arXiv: 2502.11771
    - Why relevant: Arithmetic computation vs. validation use different circuits at different layers.

11. **Language Models are Symbolic Learners in Arithmetic** (deng2024_symbolic_learners_arithmetic.pdf)
    - Authors: Deng, Li, Xie, Chang, Chen (2024)
    - arXiv: 2410.15580
    - Why relevant: Argues LMs use symbolic shortcuts rather than true computation for arithmetic.

## Core Papers — Counting Mechanisms

12. **Mechanistic Interpretability of Large-Scale Counting in LLMs** (hasani2026_counting_mechanisms_llms.pdf)
    - Authors: Hasani et al. (2026)
    - arXiv: 2601.02989
    - Why relevant: Counting info accumulates progressively across layers in compressed/logarithmic form. Middle-to-late layers (70-80% depth) consolidate numerical information.

13. **Understanding Counting Mechanisms in Large Language and Vision-Language Models** (hasani2025_counting_llm_vlm.pdf)
    - Authors: Hasani et al. (2025)
    - arXiv: 2511.17699
    - Why relevant: CountScope probing method for decoding numerical info from hidden states.

14. **Language Models Need Inductive Biases to Count Inductively** (chang2024_inductive_biases_counting.pdf)
    - Authors: Chang, Bisk (2024)
    - arXiv: 2405.20131
    - Why relevant: Transformers cannot generalize counting OOD without positional embeddings.

15. **Counting Ability of LLMs and Impact of Tokenization** (zhang2024_counting_ability_tokenization.pdf)
    - Authors: Zhang, Cao, You (2024)
    - arXiv: 2410.19730
    - Why relevant: Tokenization impacts counting; transformers limited to TC0 complexity.

16. **Why Do LLMs Struggle to Count Letters?** (fu2024_llms_count_letters.pdf)
    - Authors: Fu, Ferrando, Conde, Arriaga, Reviriego (2024)
    - arXiv: 2412.18626
    - Why relevant: Counting errors correlate with counting complexity, not memorization.

## Supporting Papers

17. **States Hidden in Hidden States** (chen2024_states_hidden_in_hidden_states.pdf)
    - Authors: Chen, Hu, Liu, Sun (2024)
    - arXiv: 2407.11421
    - Why relevant: LLMs form implicit discrete state representations for multi-step arithmetic.

18. **A Fragile Number Sense** (rahman2025_fragile_number_sense.pdf)
    - Authors: Rahman, Mishra (2025)
    - arXiv: 2509.06332
    - Why relevant: LLM numerical reasoning is fragile and pattern-based.

19. **Compact Proofs of Model Performance via Mechanistic Interpretability** (gross2024_compact_proofs_mechanistic.pdf)
    - Authors: Gross et al. (2024)
    - arXiv: 2406.11779
    - Why relevant: Formal verification of Max-of-K circuits — comparison/ordering task related to intervals.

20. **Mechanistic Interpretability for AI Safety — A Review** (bereska2024_mechanistic_interpretability_review.pdf)
    - Authors: Bereska, Gavves (2024)
    - arXiv: 2404.14082
    - Why relevant: Comprehensive review of MI methodology used throughout this research.
