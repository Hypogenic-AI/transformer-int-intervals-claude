"""Generate synthetic datasets for interval membership, comparison, and ring counting experiments."""

import json
import random
import os

def generate_comparison_data(n=2000, seed=42):
    """Generate number comparison pairs: Is A > B?"""
    random.seed(seed)
    samples = []
    for _ in range(n):
        a = random.randint(0, 99)
        b = random.randint(0, 99)
        while b == a:
            b = random.randint(0, 99)
        label = 1 if a > b else 0
        prompt = f"Is {a} greater than {b}? Answer Yes or No."
        samples.append({"a": a, "b": b, "label": label, "prompt": prompt})
    return samples

def generate_interval_data(n=2000, seed=42):
    """Generate interval membership queries: Is X in [A, B]?"""
    random.seed(seed)
    samples = []
    for _ in range(n):
        a = random.randint(0, 90)
        b = random.randint(a + 1, 100)
        x = random.randint(0, 100)
        label = 1 if a <= x <= b else 0
        width = b - a
        # distance from nearest boundary (negative if outside)
        if x < a:
            boundary_dist = x - a  # negative
        elif x > b:
            boundary_dist = x - b  # positive
        else:
            boundary_dist = min(x - a, b - x)  # positive, distance to nearest edge
        prompt = f"Is {x} in the interval [{a}, {b}]? Answer Yes or No."
        samples.append({
            "x": x, "a": a, "b": b, "label": label,
            "width": width, "boundary_dist": boundary_dist,
            "prompt": prompt
        })
    return samples

def generate_ring_data(n=2000, seed=42):
    """Generate alphabetic ring counting: What letter is N after start?"""
    random.seed(seed)
    samples = []
    for _ in range(n):
        start_idx = random.randint(0, 25)
        offset = random.randint(1, 25)
        answer_idx = (start_idx + offset) % 26
        start_letter = chr(ord('A') + start_idx)
        answer_letter = chr(ord('A') + answer_idx)
        wraps = (start_idx + offset) >= 26
        prompt = f"What letter is {offset} after {start_letter} in the alphabet (wrapping around)? The answer is"
        samples.append({
            "start_idx": start_idx, "offset": offset,
            "answer_idx": answer_idx, "start_letter": start_letter,
            "answer_letter": answer_letter, "wraps": wraps,
            "prompt": prompt
        })
    return samples

def generate_interval_by_width(n_per_width=500, widths=[2, 5, 10, 20, 50], seed=42):
    """Generate interval membership queries stratified by interval width."""
    random.seed(seed)
    samples = []
    for w in widths:
        for _ in range(n_per_width):
            a = random.randint(0, 100 - w)
            b = a + w
            x = random.randint(0, 100)
            label = 1 if a <= x <= b else 0
            prompt = f"Is {x} in the interval [{a}, {b}]? Answer Yes or No."
            samples.append({
                "x": x, "a": a, "b": b, "label": label,
                "width": w, "prompt": prompt
            })
    return samples

def save_datasets(output_dir="datasets/generated"):
    """Generate and save all datasets."""
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "comparison": generate_comparison_data(),
        "interval": generate_interval_data(),
        "ring": generate_ring_data(),
        "interval_by_width": generate_interval_by_width(),
    }

    for name, data in datasets.items():
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} samples to {path}")
        # Print label distribution
        if "label" in data[0]:
            pos = sum(1 for s in data if s["label"] == 1)
            print(f"  Label distribution: {pos}/{len(data)} positive ({100*pos/len(data):.1f}%)")

    return datasets

if __name__ == "__main__":
    save_datasets()
