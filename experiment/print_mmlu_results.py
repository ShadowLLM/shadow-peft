import json
import sys

if len(sys.argv) < 2:
    print("Usage: python print_mmlu_results2.py <result_json> [selector_key]")
    sys.exit(1)

result_path = sys.argv[1]
selector = sys.argv[2] if len(sys.argv) > 2 else None
with open(result_path, "r") as f:
    data = json.load(f)

def _is_subject_metrics_mapping(obj: object) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    if not all(isinstance(v, dict) for v in obj.values()):
        return False
    # Any entry having "accuracy" is enough to treat this as a subject->metrics mapping.
    return any("accuracy" in v for v in obj.values())


def _unwrap_results(obj: object, selector_key: str | None = None) -> dict:
    """
    Try to unwrap nested suite results into a mapping:
        {subject_name: {loss, accuracy, samples, shadow_loss?, shadow_accuracy?}}
    """
    cur = obj
    # Walk down single-key wrappers and/or selector keys.
    for _ in range(10):
        if not isinstance(cur, dict):
            break
        if selector_key and selector_key in cur and isinstance(cur[selector_key], dict):
            cur = cur[selector_key]
            continue
        if _is_subject_metrics_mapping(cur):
            return cur
        if len(cur) == 1:
            cur = next(iter(cur.values()))
            continue
        # Common cases: pick a known branch if present.
        for k in ("shadow_peft_sft", "lora_sft", "shadow_peft", "lora"):
            if k in cur and isinstance(cur[k], dict):
                cur = cur[k]
                break
        else:
            # Can't unwrap further.
            break
    if isinstance(cur, dict) and _is_subject_metrics_mapping(cur):
        return cur
    raise ValueError(
        "Could not find subject-level metrics mapping in the provided JSON. "
        "If your JSON has multiple branches, pass a selector_key as the 2nd argument."
    )


results = _unwrap_results(data, selector_key=selector)

# Print header
print(
    f"{'Subject':<45} "
    f"{'Loss':<12} {'Accuracy':<12} {'ShadowAcc':<12} "
    f"{'Samples':<10}"
)
print("=" * 80)

# Collect metrics for averaging
total_loss = 0.0
total_accuracy = 0.0
total_shadow_accuracy = 0.0
total_samples = 0
weighted_accuracy_sum = 0.0
weighted_shadow_accuracy_sum = 0.0
weighted_samples = 0
weighted_shadow_samples = 0
num_subjects = 0
num_shadow_subjects = 0

# Print each subject's results
for subject, metrics in sorted(results.items()):
    loss = float(metrics.get("loss", 0.0))
    accuracy = float(metrics.get("accuracy", 0.0))
    shadow_accuracy = metrics.get("shadow_accuracy", None)
    samples = int(metrics.get("samples", 0))
    
    shadow_acc_str = "n/a"
    if shadow_accuracy is not None:
        shadow_accuracy = float(shadow_accuracy)
        shadow_acc_str = f"{shadow_accuracy:<12.4f}"
    print(
        f"{subject:<45} "
        f"{loss:<12.4f} {accuracy:<12.4f} {shadow_acc_str} "
        f"{samples:<10}"
    )
    
    total_loss += loss
    total_accuracy += accuracy
    total_samples += samples
    if samples > 0:
        weighted_accuracy_sum += accuracy * samples
        weighted_samples += samples
    num_subjects += 1

    if shadow_accuracy is not None:
        total_shadow_accuracy += shadow_accuracy
        num_shadow_subjects += 1
        if samples > 0:
            weighted_shadow_accuracy_sum += shadow_accuracy * samples
            weighted_shadow_samples += samples

# Print separator and averages
print("=" * 80)
avg_loss = total_loss / num_subjects if num_subjects > 0 else 0.0
avg_accuracy = total_accuracy / num_subjects if num_subjects > 0 else 0.0
avg_shadow_accuracy = (
    total_shadow_accuracy / num_shadow_subjects if num_shadow_subjects > 0 else None
)

weighted_avg_accuracy = weighted_accuracy_sum / weighted_samples if weighted_samples > 0 else 0.0
weighted_avg_shadow_accuracy = (
    weighted_shadow_accuracy_sum / weighted_shadow_samples if weighted_shadow_samples > 0 else None
)

avg_shadow_str = f"{avg_shadow_accuracy:<12.4f}" if avg_shadow_accuracy is not None else "n/a"
print(
    f"{'MACRO_AVG':<45} "
    f"{avg_loss:<12.4f} {avg_accuracy:<12.4f} {avg_shadow_str} "
    f"{total_samples:<10}"
)
weighted_shadow_str = (
    f"{weighted_avg_shadow_accuracy:<12.4f}" if weighted_avg_shadow_accuracy is not None else "n/a"
)
print(
    f"{'WEIGHTED_AVG':<45} "
    f"{'':<12} {weighted_avg_accuracy:<12.4f} {weighted_shadow_str} "
    f"{total_samples:<10}"
)
print(f"\nTotal subjects: {num_subjects}")
print(f"Subjects with shadow_accuracy: {num_shadow_subjects}")
print(f"Total samples: {total_samples}")