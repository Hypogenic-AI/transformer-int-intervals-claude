"""
Streamlined experiment runner for interval/ring probing study.
Optimized for speed: smaller CV, faster probes, batched patching.
"""

import json
import os
import random
import sys
import time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ── Setup ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0"
RESULTS_DIR = "../results"
FIGURES_DIR = "../results/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")


# ── Data Generation ──
def gen_comparison(n=800):
    random.seed(SEED)
    samples = []
    for _ in range(n):
        a, b = random.randint(0, 99), random.randint(0, 99)
        while b == a: b = random.randint(0, 99)
        samples.append({"a": a, "b": b, "label": int(a > b),
                        "prompt": f"Is {a} greater than {b}? Answer Yes or No."})
    return samples

def gen_interval(n=800):
    random.seed(SEED + 1)
    samples = []
    for _ in range(n):
        a = random.randint(0, 90)
        b = random.randint(a + 1, 100)
        x = random.randint(0, 100)
        label = int(a <= x <= b)
        samples.append({"x": x, "a": a, "b": b, "label": label, "width": b - a,
                        "prompt": f"Is {x} in the interval [{a}, {b}]? Answer Yes or No."})
    return samples

def gen_ring(n=800):
    random.seed(SEED + 2)
    samples = []
    for _ in range(n):
        start = random.randint(0, 25)
        offset = random.randint(1, 25)
        answer = (start + offset) % 26
        sl, al = chr(65 + start), chr(65 + answer)
        wraps = (start + offset) >= 26
        samples.append({"start_idx": start, "offset": offset, "answer_idx": answer,
                        "start_letter": sl, "answer_letter": al, "wraps": wraps,
                        "prompt": f"What letter is {offset} after {sl} in the alphabet (wrapping around)? The answer is"})
    return samples

def gen_interval_by_width(n_per=200, widths=[2, 5, 10, 20, 50]):
    random.seed(SEED + 3)
    samples = []
    for w in widths:
        for _ in range(n_per):
            a = random.randint(0, 100 - w)
            b = a + w
            x = random.randint(0, 100)
            samples.append({"x": x, "a": a, "b": b, "label": int(a <= x <= b), "width": w,
                            "prompt": f"Is {x} in the interval [{a}, {b}]? Answer Yes or No."})
    return samples


# ── Model Loading ──
def load_model():
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    print(f"Model: gpt2 ({model.cfg.n_layers} layers, {model.cfg.d_model}d)")
    return model


# ── Hidden State Extraction ──
def extract_states(model, prompts, batch_size=64):
    """Extract residual stream at last token for all layers."""
    n_layers = model.cfg.n_layers + 1
    states = {i: [] for i in range(n_layers)}

    for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting"):
        batch = prompts[start:start + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: "resid_post" in name or name == "hook_embed"
            )

        # Last real token position
        for i in range(tokens.shape[0]):
            pos = tokens.shape[1] - 1  # last position

            states[0].append(cache["hook_embed"][i, pos].cpu().numpy())
            for layer in range(model.cfg.n_layers):
                states[layer + 1].append(cache[f"blocks.{layer}.hook_resid_post"][i, pos].cpu().numpy())

        del cache
        torch.cuda.empty_cache()

    return {k: np.array(v) for k, v in states.items()}


# ── Behavioral Evaluation ──
def eval_yesno(model, data, batch_size=128):
    """Evaluate Yes/No tasks."""
    yes_id = model.to_single_token(" Yes")
    no_id = model.to_single_token(" No")
    correct = 0

    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        tokens = model.to_tokens([s["prompt"] for s in batch], prepend_bos=True)
        with torch.no_grad():
            logits = model(tokens)[:, -1, :]
        preds = (logits[:, yes_id] > logits[:, no_id]).cpu().numpy().astype(int)
        correct += sum(p == s["label"] for p, s in zip(preds, batch))

    return correct / len(data)

def eval_ring(model, data, batch_size=128):
    """Evaluate ring counting (check top-5 predictions for answer letter)."""
    correct = 0
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        tokens = model.to_tokens([s["prompt"] for s in batch], prepend_bos=True)
        with torch.no_grad():
            logits = model(tokens)[:, -1, :]

        for i, s in enumerate(batch):
            top5 = logits[i].topk(5).indices.tolist()
            answer_ids = set()
            for pfx in [" ", ""]:
                try: answer_ids.add(model.to_single_token(pfx + s["answer_letter"]))
                except: pass
            if answer_ids & set(top5):
                correct += 1

    return correct / len(data)


# ── Probing ──
def probe_classify(states, labels, n_cv=3):
    """Fast linear probe classification per layer. Returns {layer: (mean_acc, std)}."""
    labels = np.array(labels)
    results = {}
    for layer in sorted(states.keys()):
        X = StandardScaler().fit_transform(states[layer])
        clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs', random_state=SEED)
        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=SEED)
        scores = cross_val_score(clf, X, labels, cv=cv, scoring='accuracy', n_jobs=-1)
        results[layer] = (float(scores.mean()), float(scores.std()))
    return results

def probe_multiclass(states, labels, n_cv=3):
    """Multiclass probe for ring counting."""
    labels = np.array(labels)
    results = {}
    for layer in sorted(states.keys()):
        X = StandardScaler().fit_transform(states[layer])
        clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs', random_state=SEED)
        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=SEED)
        scores = cross_val_score(clf, X, labels, cv=cv, scoring='accuracy', n_jobs=-1)
        results[layer] = (float(scores.mean()), float(scores.std()))
    return results

def probe_regression(states, values, n_cv=3):
    """Ridge regression probe per layer. Returns {layer: (mean_r2, std)}."""
    values = np.array(values, dtype=float)
    results = {}
    for layer in sorted(states.keys()):
        X = StandardScaler().fit_transform(states[layer])
        reg = Ridge(alpha=1.0)
        cv = KFold(n_splits=n_cv, shuffle=True, random_state=SEED)
        scores = cross_val_score(reg, X, values, cv=cv, scoring='r2', n_jobs=-1)
        results[layer] = (float(scores.mean()), float(scores.std()))
    return results


# ── Activation Patching ──
def activation_patching(model, src_data, tgt_data, n_pairs=80):
    """Patch residual stream from source to target, measure logit shift."""
    yes_id = model.to_single_token(" Yes")
    no_id = model.to_single_token(" No")

    effects = {l: [] for l in range(model.cfg.n_layers)}

    for i in tqdm(range(n_pairs), desc="Patching"):
        src_tok = model.to_tokens([src_data[i]["prompt"]], prepend_bos=True)
        tgt_tok = model.to_tokens([tgt_data[i]["prompt"]], prepend_bos=True)

        with torch.no_grad():
            _, src_cache = model.run_with_cache(src_tok)
            clean_logits = model(tgt_tok)

        clean_diff = (clean_logits[0, -1, yes_id] - clean_logits[0, -1, no_id]).item()

        for layer in range(model.cfg.n_layers):
            key = f"blocks.{layer}.hook_resid_post"
            src_act = src_cache[key][0, -1, :].clone()

            def hook_fn(act, hook, s=src_act):
                act[0, -1, :] = s
                return act

            patched_logits = model.run_with_hooks(tgt_tok, fwd_hooks=[(key, hook_fn)])
            patched_diff = (patched_logits[0, -1, yes_id] - patched_logits[0, -1, no_id]).item()

            eff = patched_diff - clean_diff
            if src_data[i]["label"] == 0:
                eff = -eff
            effects[layer].append(eff)

        del src_cache
        torch.cuda.empty_cache()

    return {l: (float(np.mean(e)), float(np.std(e))) for l, e in effects.items()}


# ── Main ──
def main():
    t0 = time.time()
    results = {}

    # Load
    model = load_model()

    # Generate data
    print("\n=== Generating Data ===")
    comp = gen_comparison()
    intv = gen_interval()
    ring = gen_ring()
    wdata = gen_interval_by_width()

    for name, d in [("comparison", comp), ("interval", intv)]:
        pos = sum(s["label"] for s in d)
        print(f"  {name}: {pos}/{len(d)} positive ({100*pos/len(d):.1f}%)")

    # Behavioral
    print("\n=== Behavioral Evaluation ===")
    results["behavioral"] = {}
    results["behavioral"]["comparison"] = eval_yesno(model, comp)
    print(f"  Comparison: {results['behavioral']['comparison']:.4f}")
    results["behavioral"]["interval"] = eval_yesno(model, intv)
    print(f"  Interval: {results['behavioral']['interval']:.4f}")
    results["behavioral"]["ring"] = eval_ring(model, ring)
    print(f"  Ring (top-5): {results['behavioral']['ring']:.4f}")

    # Extract hidden states
    print("\n=== Extracting Hidden States ===")
    comp_states = extract_states(model, [s["prompt"] for s in comp])
    int_states = extract_states(model, [s["prompt"] for s in intv])
    ring_states = extract_states(model, [s["prompt"] for s in ring])

    # Probing: comparison
    print("\n=== Probing: Comparison ===")
    print("  Real labels:")
    comp_real = probe_classify(comp_states, [s["label"] for s in comp])
    for l, (m, s) in sorted(comp_real.items()):
        print(f"    Layer {l:2d}: {m:.4f} ± {s:.4f}")

    print("  Shuffled control:")
    shuf_labels = [s["label"] for s in comp]
    np.random.shuffle(shuf_labels)
    comp_ctrl = probe_classify(comp_states, shuf_labels)
    for l, (m, s) in sorted(comp_ctrl.items()):
        print(f"    Layer {l:2d}: {m:.4f} ± {s:.4f}")

    results["probe_comparison"] = {"real": {k: v[0] for k, v in comp_real.items()},
                                    "control": {k: v[0] for k, v in comp_ctrl.items()}}

    # Probing: interval
    print("\n=== Probing: Interval Membership ===")
    print("  Real labels:")
    int_real = probe_classify(int_states, [s["label"] for s in intv])
    for l, (m, s) in sorted(int_real.items()):
        print(f"    Layer {l:2d}: {m:.4f} ± {s:.4f}")

    print("  Shuffled control:")
    shuf_int = [s["label"] for s in intv]
    np.random.shuffle(shuf_int)
    int_ctrl = probe_classify(int_states, shuf_int)
    for l, (m, s) in sorted(int_ctrl.items()):
        print(f"    Layer {l:2d}: {m:.4f} ± {s:.4f}")

    results["probe_interval"] = {"real": {k: v[0] for k, v in int_real.items()},
                                  "control": {k: v[0] for k, v in int_ctrl.items()}}

    # Probing: ring (26-class)
    print("\n=== Probing: Ring Counting (26-class) ===")
    ring_probe = probe_multiclass(ring_states, [s["answer_idx"] for s in ring])
    for l, (m, s) in sorted(ring_probe.items()):
        print(f"    Layer {l:2d}: {m:.4f} ± {s:.4f}")
    print(f"    Chance: {1/26:.4f}")
    results["probe_ring"] = {k: v[0] for k, v in ring_probe.items()}

    # Probing: regression (decode X value)
    print("\n=== Probing: Decode X value (regression) ===")
    x_reg = probe_regression(int_states, [s["x"] for s in intv])
    for l, (m, s) in sorted(x_reg.items()):
        print(f"    Layer {l:2d}: R²={m:.4f} ± {s:.4f}")
    results["probe_x_value"] = {k: v[0] for k, v in x_reg.items()}

    # Probing: decode offset
    print("\n=== Probing: Decode offset N (regression) ===")
    off_reg = probe_regression(ring_states, [s["offset"] for s in ring])
    for l, (m, s) in sorted(off_reg.items()):
        print(f"    Layer {l:2d}: R²={m:.4f} ± {s:.4f}")
    results["probe_ring_offset"] = {k: v[0] for k, v in off_reg.items()}

    # Probing by width
    print("\n=== Probing: Interval by Width ===")
    w_states = extract_states(model, [s["prompt"] for s in wdata])
    w_labels = np.array([s["label"] for s in wdata])
    w_widths = np.array([s["width"] for s in wdata])

    width_results = {}
    key_layers = [0, 3, 6, 9, 12]
    for w in [2, 5, 10, 20, 50]:
        mask = w_widths == w
        sub_labels = w_labels[mask]
        sub_states = {k: v[mask] for k, v in w_states.items()}
        pos = sub_labels.sum()
        print(f"\n  Width={w} (n={mask.sum()}, pos={pos}/{mask.sum()})")
        wr = {}
        for l in key_layers:
            X = StandardScaler().fit_transform(sub_states[l])
            clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs', random_state=SEED)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            scores = cross_val_score(clf, X, sub_labels, cv=cv, scoring='accuracy', n_jobs=-1)
            wr[l] = float(scores.mean())
            print(f"    Layer {l:2d}: {scores.mean():.4f} ± {scores.std():.4f}")
        width_results[w] = wr
    results["probe_interval_by_width"] = width_results

    # Ring: wrap vs no-wrap
    print("\n=== Probing: Ring wrap vs no-wrap ===")
    wraps_arr = np.array([s["wraps"] for s in ring])
    ring_wrap_results = {}
    for cond, label in [(True, "wrap"), (False, "no-wrap")]:
        mask = wraps_arr == cond
        sub_labels = np.array([s["answer_idx"] for s in ring])[mask]
        sub_states = {k: v[mask] for k, v in ring_states.items()}
        print(f"\n  {label} (n={mask.sum()})")
        wr = {}
        for l in key_layers:
            X = StandardScaler().fit_transform(sub_states[l])
            clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs', random_state=SEED)
            n_unique = len(np.unique(sub_labels))
            cv = StratifiedKFold(n_splits=min(3, n_unique), shuffle=True, random_state=SEED)
            scores = cross_val_score(clf, X, sub_labels, cv=cv, scoring='accuracy', n_jobs=-1)
            wr[l] = float(scores.mean())
            print(f"    Layer {l:2d}: {scores.mean():.4f} ± {scores.std():.4f}")
        ring_wrap_results[label] = wr
    results["probe_ring_wrap"] = ring_wrap_results

    # Activation patching
    print("\n=== Activation Patching: Interval ===")
    in_intv = [s for s in intv if s["label"] == 1]
    out_intv = [s for s in intv if s["label"] == 0]
    n_patch = min(80, len(in_intv), len(out_intv))
    patch_int = activation_patching(model, in_intv[:n_patch], out_intv[:n_patch], n_patch)
    for l in sorted(patch_int.keys()):
        m, s = patch_int[l]
        print(f"  Layer {l:2d}: {m:+.4f} ± {s:.4f}")
    results["patching_interval"] = {k: v[0] for k, v in patch_int.items()}

    print("\n=== Activation Patching: Comparison ===")
    comp_pos = [s for s in comp if s["label"] == 1]
    comp_neg = [s for s in comp if s["label"] == 0]
    n_patch_c = min(80, len(comp_pos), len(comp_neg))
    patch_comp = activation_patching(model, comp_pos[:n_patch_c], comp_neg[:n_patch_c], n_patch_c)
    for l in sorted(patch_comp.keys()):
        m, s = patch_comp[l]
        print(f"  Layer {l:2d}: {m:+.4f} ± {s:.4f}")
    results["patching_comparison"] = {k: v[0] for k, v in patch_comp.items()}

    # Save
    elapsed = time.time() - t0
    results["metadata"] = {
        "model": "gpt2", "device": DEVICE, "seed": SEED,
        "elapsed_seconds": elapsed,
        "n_comparison": len(comp), "n_interval": len(intv),
        "n_ring": len(ring), "n_width": len(wdata),
    }

    def to_json(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict): return {str(k): to_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [to_json(x) for x in obj]
        return obj

    with open(os.path.join(RESULTS_DIR, "experiment_results.json"), "w") as f:
        json.dump(to_json(results), f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to {RESULTS_DIR}/experiment_results.json")

    return results


if __name__ == "__main__":
    main()
