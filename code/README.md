# Cloned Repositories

## Repo 1: base10 (Levy & Geva, 2024)
- **URL**: https://github.com/amitlevy/base10
- **Purpose**: Code for "Language Models Encode Numbers Using Digit Representations in Base 10"
- **Location**: code/base10/
- **Key files**: `circular_probe.py`, `train_circ_probes.ipynb`
- **Relevance**: Demonstrates circular (sinusoidal) probes for decoding digit-level number representations from LLM hidden states. Core methodology for understanding how numbers are internally encoded.

## Repo 2: lm-arithmetic (Stolfo et al., 2023)
- **URL**: https://github.com/alestolfo/lm-arithmetic
- **Purpose**: Causal mediation analysis for arithmetic reasoning in LMs (EMNLP 2023)
- **Location**: code/lm-arithmetic/
- **Key files**: See README for experiment scripts
- **Relevance**: Framework for identifying which MLP/attention components contribute to arithmetic. Directly applicable for studying interval membership computation circuits.

## Repo 3: gpt2-greater-than (Hanna et al., 2023)
- **URL**: https://github.com/hannamw/gpt2-greater-than
- **Purpose**: Mechanistic interpretation of how GPT-2 computes greater-than comparison (NeurIPS 2023)
- **Location**: code/gpt2-greater-than/
- **Key files**: Path patching and circuit analysis code
- **Relevance**: **Most directly relevant repo.** Greater-than is the core operation for interval boundary checking. The discovered circuit (attention heads routing YY info → MLPs 8-11 computing step function) provides the template for understanding interval membership.

## Repo 4: numllama (Kadlcik et al., 2025)
- **URL**: https://github.com/prompteus/numllama
- **Purpose**: Sinusoidal probes for number representations in LLMs (EMNLP 2025)
- **Location**: code/numllama/
- **Key files**: Jupyter notebooks for probe training and analysis
- **Relevance**: Near-perfect number decoding from embeddings via Fourier-basis probes. Shows the periodic structure underlying integer representations.

## Repo 5: numeracy-probing (Yuchi et al., 2026)
- **URL**: https://github.com/VCY019/Numeracy-Probing
- **Purpose**: "LLMs Know More About Numbers than They Can Say" (EACL 2026)
- **Location**: code/numeracy-probing/
- **Key files**: Probing scripts for regression and classification on hidden states
- **Relevance**: Shows internal representations encode log-magnitudes more accurately than verbalized outputs. Linear probes for number comparison from hidden states.

## Additional Recommended Tools (Not Cloned)

- **TransformerLens**: `pip install transformer_lens` — Essential library for hooking into transformer internals, caching activations, and performing interventions. Should be installed for experiment phase.
- **quanta_mech_interp**: https://github.com/PhilipQuirke/quanta_mech_interp — Interpretability toolkit for analyzing transformer algorithms.
