"""
Core experiments: hidden state extraction, probing, behavioral evaluation, and activation patching.
Uses GPT-2 small via TransformerLens.
"""

import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from tqdm import tqdm

# ── Setup ──────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")

RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_model():
    """Load GPT-2 small via TransformerLens."""
    import transformer_lens
    try:
        print(f"TransformerLens: {transformer_lens.__version__}")
    except AttributeError:
        print("TransformerLens: (version unknown)")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    print(f"Model: gpt2 ({model.cfg.n_layers} layers, {model.cfg.d_model}d)")
    return model


# ── Experiment 1: Hidden State Extraction ──────────────────────────────────

def extract_hidden_states(model, prompts, batch_size=32):
    """Extract residual stream activations at all layers for the last token position.

    Returns: dict mapping layer_idx -> np.array of shape (n_samples, d_model)
    Layer 0 = after embedding, Layer 1-12 = after each transformer block.
    """
    n_layers = model.cfg.n_layers + 1  # embedding + 12 blocks
    all_states = {i: [] for i in range(n_layers)}

    for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting hidden states"):
        batch = prompts[start:start + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: "resid_post" in name or name == "hook_embed")

        # Get last token position for each sample
        # tokens shape: (batch, seq_len)
        seq_lens = (tokens != model.tokenizer.pad_token_id).sum(dim=1) if model.tokenizer.pad_token_id is not None else torch.full((tokens.shape[0],), tokens.shape[1], device=tokens.device)

        # Embedding layer
        embed = cache["hook_embed"]  # (batch, seq, d_model)
        for i in range(embed.shape[0]):
            pos = min(seq_lens[i].item() - 1, embed.shape[1] - 1)
            all_states[0].append(embed[i, pos].cpu().numpy())

        # Residual stream after each layer
        for layer in range(model.cfg.n_layers):
            key = f"blocks.{layer}.hook_resid_post"
            resid = cache[key]  # (batch, seq, d_model)
            for i in range(resid.shape[0]):
                pos = min(seq_lens[i].item() - 1, resid.shape[1] - 1)
                all_states[layer + 1].append(resid[i, pos].cpu().numpy())

        del cache
        torch.cuda.empty_cache()

    return {k: np.array(v) for k, v in all_states.items()}


# ── Experiment 2: Behavioral Evaluation ────────────────────────────────────

def behavioral_eval_comparison(model, data, batch_size=64):
    """Evaluate GPT-2's next-token prediction on comparison tasks."""
    correct = 0
    total = 0
    yes_id = model.to_single_token(" Yes")
    no_id = model.to_single_token(" No")

    for start in tqdm(range(0, len(data), batch_size), desc="Behavioral eval (comparison)"):
        batch = data[start:start + batch_size]
        prompts = [s["prompt"] for s in batch]
        labels = [s["label"] for s in batch]
        tokens = model.to_tokens(prompts, prepend_bos=True)

        with torch.no_grad():
            logits = model(tokens)

        # Get logits at last position
        last_logits = logits[:, -1, :]  # (batch, vocab)
        yes_logits = last_logits[:, yes_id]
        no_logits = last_logits[:, no_id]
        preds = (yes_logits > no_logits).cpu().numpy().astype(int)

        for pred, label in zip(preds, labels):
            if pred == label:
                correct += 1
            total += 1

    return correct / total


def behavioral_eval_interval(model, data, batch_size=64):
    """Evaluate GPT-2's next-token prediction on interval membership."""
    correct = 0
    total = 0
    yes_id = model.to_single_token(" Yes")
    no_id = model.to_single_token(" No")

    for start in tqdm(range(0, len(data), batch_size), desc="Behavioral eval (interval)"):
        batch = data[start:start + batch_size]
        prompts = [s["prompt"] for s in batch]
        labels = [s["label"] for s in batch]
        tokens = model.to_tokens(prompts, prepend_bos=True)

        with torch.no_grad():
            logits = model(tokens)

        last_logits = logits[:, -1, :]
        yes_logits = last_logits[:, yes_id]
        no_logits = last_logits[:, no_id]
        preds = (yes_logits > no_logits).cpu().numpy().astype(int)

        for pred, label in zip(preds, labels):
            if pred == label:
                correct += 1
            total += 1

    return correct / total


def behavioral_eval_ring(model, data, batch_size=64):
    """Evaluate GPT-2's next-token prediction on ring counting.
    Check if the model's top-1 token matches the answer letter.
    """
    correct = 0
    total = 0

    for start in tqdm(range(0, len(data), batch_size), desc="Behavioral eval (ring)"):
        batch = data[start:start + batch_size]
        prompts = [s["prompt"] for s in batch]
        answers = [s["answer_letter"] for s in batch]
        tokens = model.to_tokens(prompts, prepend_bos=True)

        with torch.no_grad():
            logits = model(tokens)

        last_logits = logits[:, -1, :]  # (batch, vocab)

        for i, answer in enumerate(answers):
            # Check if the answer letter (with or without space) is in top predictions
            answer_ids = []
            for prefix in [" ", ""]:
                try:
                    tid = model.to_single_token(prefix + answer)
                    answer_ids.append(tid)
                except:
                    pass
            pred_token = last_logits[i].argmax().item()
            if pred_token in answer_ids:
                correct += 1
            total += 1

    return correct / total


# ── Experiment 3: Linear Probing ───────────────────────────────────────────

def probe_classification(hidden_states, labels, n_splits=5):
    """Train linear probes to classify labels from hidden states at each layer.

    Returns: dict mapping layer_idx -> (mean_accuracy, std_accuracy)
    """
    results = {}
    labels = np.array(labels)

    for layer_idx in sorted(hidden_states.keys()):
        X = hidden_states[layer_idx]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=SEED)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        scores = cross_val_score(clf, X_scaled, labels, cv=cv, scoring='accuracy')
        results[layer_idx] = (scores.mean(), scores.std())
        print(f"  Layer {layer_idx:2d}: {scores.mean():.4f} ± {scores.std():.4f}")

    return results


def probe_regression(hidden_states, values, n_splits=5):
    """Train linear probes to regress numerical values from hidden states.

    Returns: dict mapping layer_idx -> (mean_r2, std_r2)
    """
    results = {}
    values = np.array(values, dtype=float)

    for layer_idx in sorted(hidden_states.keys()):
        X = hidden_states[layer_idx]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reg = Ridge(alpha=1.0)
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        scores = cross_val_score(reg, X_scaled, values, cv=cv, scoring='r2')
        results[layer_idx] = (scores.mean(), scores.std())
        print(f"  Layer {layer_idx:2d}: R²={scores.mean():.4f} ± {scores.std():.4f}")

    return results


def probe_with_shuffled_control(hidden_states, labels, n_splits=5):
    """Probe with real labels and shuffled labels (control) to validate probing results."""
    print("  Real labels:")
    real_results = probe_classification(hidden_states, labels, n_splits)

    shuffled_labels = np.array(labels).copy()
    np.random.shuffle(shuffled_labels)
    print("  Shuffled labels (control):")
    control_results = probe_classification(hidden_states, shuffled_labels, n_splits)

    return real_results, control_results


# ── Experiment 4: Activation Patching ──────────────────────────────────────

def activation_patching(model, source_prompts, target_prompts, source_labels, target_labels, layers_to_patch=None):
    """Patch residual stream activations from source to target examples.

    Measures how much patching the residual stream at each layer shifts the model's
    output from the target prediction toward the source prediction.

    Returns: dict mapping layer_idx -> mean_effect (logit difference change)
    """
    if layers_to_patch is None:
        layers_to_patch = list(range(model.cfg.n_layers))

    yes_id = model.to_single_token(" Yes")
    no_id = model.to_single_token(" No")

    n_pairs = min(len(source_prompts), len(target_prompts), 200)
    effects = {layer: [] for layer in layers_to_patch}

    for i in tqdm(range(n_pairs), desc="Activation patching"):
        src_tokens = model.to_tokens([source_prompts[i]], prepend_bos=True)
        tgt_tokens = model.to_tokens([target_prompts[i]], prepend_bos=True)

        # Get source activations
        with torch.no_grad():
            _, src_cache = model.run_with_cache(src_tokens)

        # Get clean target logits
        with torch.no_grad():
            clean_logits = model(tgt_tokens)
        clean_diff = (clean_logits[0, -1, yes_id] - clean_logits[0, -1, no_id]).item()

        for layer in layers_to_patch:
            key = f"blocks.{layer}.hook_resid_post"
            src_act = src_cache[key][0, -1, :].clone()  # last token

            def patch_hook(activation, hook, src_activation=src_act):
                activation[0, -1, :] = src_activation
                return activation

            patched_logits = model.run_with_hooks(
                tgt_tokens,
                fwd_hooks=[(key, patch_hook)]
            )
            patched_diff = (patched_logits[0, -1, yes_id] - patched_logits[0, -1, no_id]).item()

            # Effect: how much did patching shift the logit difference?
            effect = patched_diff - clean_diff
            # Normalize: positive effect means patching moved prediction toward source's label
            if source_labels[i] == 1:  # source is "Yes"
                effects[layer].append(effect)  # positive effect = moved toward Yes
            else:
                effects[layer].append(-effect)  # flip sign so positive = moved toward source

        del src_cache
        torch.cuda.empty_cache()

    return {layer: (np.mean(effs), np.std(effs)) for layer, effs in effects.items()}


# ── Main Experiment Runner ─────────────────────────────────────────────────

def run_all_experiments():
    """Run the complete experimental pipeline."""
    from data_generation import (
        generate_comparison_data,
        generate_interval_data,
        generate_ring_data,
        generate_interval_by_width,
    )

    all_results = {}
    t_start = time.time()

    # ── Load model ──
    print("=" * 60)
    print("Loading GPT-2 small...")
    model = load_model()

    # ── Generate data ──
    print("\n" + "=" * 60)
    print("Generating datasets...")
    comparison_data = generate_comparison_data(n=1000)
    interval_data = generate_interval_data(n=1000)
    ring_data = generate_ring_data(n=1000)
    width_data = generate_interval_by_width(n_per_width=300, widths=[2, 5, 10, 20, 50])

    # Print label distributions
    for name, data in [("comparison", comparison_data), ("interval", interval_data)]:
        pos = sum(1 for s in data if s["label"] == 1)
        print(f"  {name}: {pos}/{len(data)} positive ({100*pos/len(data):.1f}%)")
    wraps = sum(1 for s in ring_data if s["wraps"])
    print(f"  ring: {wraps}/{len(ring_data)} wrap-around ({100*wraps/len(ring_data):.1f}%)")

    # ── Behavioral evaluation ──
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Behavioral Evaluation")
    print("-" * 40)

    comp_acc = behavioral_eval_comparison(model, comparison_data)
    print(f"  Comparison accuracy: {comp_acc:.4f}")

    int_acc = behavioral_eval_interval(model, interval_data)
    print(f"  Interval membership accuracy: {int_acc:.4f}")

    ring_acc = behavioral_eval_ring(model, ring_data)
    print(f"  Ring counting accuracy: {ring_acc:.4f}")

    all_results["behavioral"] = {
        "comparison": comp_acc,
        "interval": int_acc,
        "ring": ring_acc,
    }

    # ── Hidden state extraction ──
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Hidden State Extraction + Probing")
    print("-" * 40)

    # Extract states for comparison task
    print("\nExtracting hidden states for COMPARISON task...")
    comp_prompts = [s["prompt"] for s in comparison_data]
    comp_states = extract_hidden_states(model, comp_prompts)

    print("\nExtracting hidden states for INTERVAL task...")
    int_prompts = [s["prompt"] for s in interval_data]
    int_states = extract_hidden_states(model, int_prompts)

    print("\nExtracting hidden states for RING task...")
    ring_prompts = [s["prompt"] for s in ring_data]
    ring_states = extract_hidden_states(model, ring_prompts)

    # ── Probing: classification ──
    print("\n" + "=" * 60)
    print("PROBING: Comparison (is A > B?)")
    comp_labels = [s["label"] for s in comparison_data]
    comp_probe_real, comp_probe_ctrl = probe_with_shuffled_control(comp_states, comp_labels)
    all_results["probe_comparison"] = {
        "real": {k: v[0] for k, v in comp_probe_real.items()},
        "control": {k: v[0] for k, v in comp_probe_ctrl.items()},
    }

    print("\nPROBING: Interval membership (is X in [A, B]?)")
    int_labels = [s["label"] for s in interval_data]
    int_probe_real, int_probe_ctrl = probe_with_shuffled_control(int_states, int_labels)
    all_results["probe_interval"] = {
        "real": {k: v[0] for k, v in int_probe_real.items()},
        "control": {k: v[0] for k, v in int_probe_ctrl.items()},
    }

    print("\nPROBING: Ring counting answer (classification into 26 letters)")
    ring_labels = [s["answer_idx"] for s in ring_data]
    ring_probe_results = {}
    for layer_idx in sorted(ring_states.keys()):
        X = ring_states[layer_idx]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=SEED, multi_class='multinomial')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = cross_val_score(clf, X_scaled, ring_labels, cv=cv, scoring='accuracy')
        ring_probe_results[layer_idx] = (scores.mean(), scores.std())
        print(f"  Layer {layer_idx:2d}: {scores.mean():.4f} ± {scores.std():.4f}")
    all_results["probe_ring"] = {k: v[0] for k, v in ring_probe_results.items()}
    # Chance level for 26-class: ~3.8%
    print(f"  Chance level: {1/26:.4f}")

    # ── Probing: regression (decode X value from interval prompts) ──
    print("\nPROBING: Decode X value from interval task hidden states")
    x_values = [s["x"] for s in interval_data]
    x_reg_results = probe_regression(int_states, x_values)
    all_results["probe_x_value"] = {k: v[0] for k, v in x_reg_results.items()}

    # ── Probing: decode offset from ring task ──
    print("\nPROBING: Decode offset N from ring task hidden states")
    offsets = [s["offset"] for s in ring_data]
    offset_reg_results = probe_regression(ring_states, offsets)
    all_results["probe_ring_offset"] = {k: v[0] for k, v in offset_reg_results.items()}

    # ── Probing: interval by width ──
    print("\n" + "=" * 60)
    print("PROBING: Interval membership by width")
    print("Extracting hidden states for width-stratified data...")
    width_prompts = [s["prompt"] for s in width_data]
    width_states = extract_hidden_states(model, width_prompts)
    width_labels = [s["label"] for s in width_data]
    widths_arr = np.array([s["width"] for s in width_data])

    width_probe_results = {}
    for w in [2, 5, 10, 20, 50]:
        mask = widths_arr == w
        subset_labels = np.array(width_labels)[mask]
        subset_states = {k: v[mask] for k, v in width_states.items()}
        print(f"\n  Width={w} (n={mask.sum()}, pos={subset_labels.sum()}/{mask.sum()})")

        # Only probe at key layers to save time
        key_layers = [0, 3, 6, 9, 12]
        width_results = {}
        for layer_idx in key_layers:
            X = subset_states[layer_idx]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=SEED)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            scores = cross_val_score(clf, X_scaled, subset_labels, cv=cv, scoring='accuracy')
            width_results[layer_idx] = scores.mean()
            print(f"    Layer {layer_idx:2d}: {scores.mean():.4f} ± {scores.std():.4f}")
        width_probe_results[w] = width_results

    all_results["probe_interval_by_width"] = width_probe_results

    # ── Probing: ring with vs. without wrap ──
    print("\n" + "=" * 60)
    print("PROBING: Ring counting (wrap vs. no-wrap)")
    wraps_arr = np.array([s["wraps"] for s in ring_data])
    for wrap_cond, label in [(True, "wrap"), (False, "no-wrap")]:
        mask = wraps_arr == wrap_cond
        subset_labels = np.array([s["answer_idx"] for s in ring_data])[mask]
        subset_states = {k: v[mask] for k, v in ring_states.items()}
        print(f"\n  {label} (n={mask.sum()})")
        key_layers = [0, 3, 6, 9, 12]
        for layer_idx in key_layers:
            X = subset_states[layer_idx]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=SEED, multi_class='multinomial')
            cv = StratifiedKFold(n_splits=min(5, len(np.unique(subset_labels))), shuffle=True, random_state=SEED)
            scores = cross_val_score(clf, X_scaled, subset_labels, cv=cv, scoring='accuracy')
            print(f"    Layer {layer_idx:2d}: {scores.mean():.4f} ± {scores.std():.4f}")

    # ── Activation Patching ──
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Activation Patching (Interval Task)")
    print("-" * 40)

    # Separate in-interval and out-of-interval examples
    in_interval = [s for s in interval_data if s["label"] == 1]
    out_interval = [s for s in interval_data if s["label"] == 0]
    n_patch = min(100, len(in_interval), len(out_interval))

    src_prompts = [s["prompt"] for s in in_interval[:n_patch]]
    tgt_prompts = [s["prompt"] for s in out_interval[:n_patch]]
    src_labels = [1] * n_patch
    tgt_labels = [0] * n_patch

    patch_results = activation_patching(
        model, src_prompts, tgt_prompts, src_labels, tgt_labels,
        layers_to_patch=list(range(12))
    )
    print("\nPatching effects (source=in-interval → target=out-of-interval):")
    for layer in sorted(patch_results.keys()):
        mean, std = patch_results[layer]
        print(f"  Layer {layer:2d}: {mean:+.4f} ± {std:.4f}")
    all_results["patching_interval"] = {k: v[0] for k, v in patch_results.items()}

    # Also patch comparison task
    print("\nActivation Patching (Comparison Task)")
    comp_pos = [s for s in comparison_data if s["label"] == 1]
    comp_neg = [s for s in comparison_data if s["label"] == 0]
    n_patch_comp = min(100, len(comp_pos), len(comp_neg))

    patch_comp = activation_patching(
        model,
        [s["prompt"] for s in comp_pos[:n_patch_comp]],
        [s["prompt"] for s in comp_neg[:n_patch_comp]],
        [1] * n_patch_comp, [0] * n_patch_comp,
        layers_to_patch=list(range(12))
    )
    print("\nPatching effects (source=A>B → target=A<B):")
    for layer in sorted(patch_comp.keys()):
        mean, std = patch_comp[layer]
        print(f"  Layer {layer:2d}: {mean:+.4f} ± {std:.4f}")
    all_results["patching_comparison"] = {k: v[0] for k, v in patch_comp.items()}

    # ── Save results ──
    elapsed = time.time() - t_start
    all_results["metadata"] = {
        "model": "gpt2",
        "device": DEVICE,
        "seed": SEED,
        "elapsed_seconds": elapsed,
        "n_comparison": len(comparison_data),
        "n_interval": len(interval_data),
        "n_ring": len(ring_data),
        "n_width": len(width_data),
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        return obj

    results_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"All results saved to {results_path}")
    print(f"Total time: {elapsed:.1f}s")

    return all_results


if __name__ == "__main__":
    results = run_all_experiments()
