# Datasets

This directory contains datasets for the research project: "How accurately does a Transformer store integer intervals?"

Data files are NOT committed to git due to size. Small sample files (*_sample.json) are included for reference.

## Dataset 1: Integer Interval Membership

### Overview
- **Source**: Synthetically generated
- **Size**: Configurable (default 1000+ samples per task variant)
- **Format**: JSON
- **Task**: Binary classification — is X in the interval [A, B]?
- **Splits**: Generated on-the-fly with random seeds

### Generation Instructions

```python
import json, random

random.seed(42)
samples = []
for _ in range(10000):
    a = random.randint(0, 90)
    b = random.randint(a+1, 100)
    x = random.randint(0, 100)
    label = 1 if a <= x <= b else 0
    samples.append({"x": x, "a": a, "b": b, "label": label,
                     "prompt": f"Is {x} in the interval [{a}, {b}]?"})
```

### Sample Data
See `integer_intervals/interval_membership_sample.json`

## Dataset 2: Alphabetic Ring Counting

### Overview
- **Source**: Synthetically generated
- **Task**: Given a start letter and offset N, compute the N-th letter after it on a circular alphabet
- **Format**: JSON
- **Relevance**: Tests modular arithmetic / counting in residual streams

### Generation Instructions

```python
import random
random.seed(42)
for _ in range(5000):
    start = random.randint(0, 25)
    n = random.randint(1, 26)
    answer = (start + n) % 26
    # start_letter = chr(ord('A') + start), answer_letter = chr(ord('A') + answer)
```

### Sample Data
See `integer_intervals/alphabetic_ring_sample.json`

## Dataset 3: Number Comparison Pairs

### Overview
- **Source**: Synthetically generated
- **Task**: Binary classification — is A greater than B?
- **Format**: JSON
- **Relevance**: Core operation for interval boundary checking

### Sample Data
See `integer_intervals/number_comparison_sample.json`

## Dataset 4: Range Containment (Varying Widths)

### Overview
- **Source**: Synthetically generated
- **Task**: Binary classification — is X between A and B?
- **Format**: JSON
- **Relevance**: Tests interval storage with varying interval widths (5, 10, 20, 50)

### Sample Data
See `integer_intervals/range_containment_sample.json`

## Notes

- All datasets are synthetic and generated deterministically with `random.seed(42)`
- The experiment runner should generate larger datasets as needed using the code patterns above
- For probing experiments, hidden states should be extracted from pre-trained LLMs (Llama, GPT-2, etc.) using TransformerLens
- No external dataset downloads are required — this research uses synthetic data to control experimental variables precisely
