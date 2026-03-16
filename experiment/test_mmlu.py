"""
Quick test script to verify MMLU dataset loading and formatting.
"""
from transformers import AutoTokenizer
from data_utils import build_mmlu_datasets

def test_mmlu_loading():
    print("Testing MMLU dataset loading...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test loading a small subset
    print("\nLoading abstract_algebra subset...")
    datasets = build_mmlu_datasets(
        tokenizer=tokenizer,
        max_length=512,
        eval_subsets=["abstract_algebra"],
    )
    
    print(f"Train dataset size: {len(datasets.train_dataset)}")
    print(f"Eval dataset size: {len(datasets.eval_datasets['abstract_algebra'])}")
    
    # Check first example
    print("\nFirst training example:")
    example = datasets.train_dataset[0]
    print(f"Keys: {example.keys()}")
    print(f"Input IDs length: {len(example['input_ids'])}")
    print(f"Answer letter: {example['answer_letter']}")
    print(f"Answer index: {example['answer_idx']}")
    
    # Decode and print
    decoded = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
    print(f"\nDecoded text:\n{decoded}")
    
    print("\n✓ MMLU dataset loading test passed!")
    
    # Test a few more subsets
    test_subsets = ["anatomy", "computer_security", "machine_learning"]
    print(f"\nTesting additional subsets: {test_subsets}")
    for subset in test_subsets:
        ds = build_mmlu_datasets(
            tokenizer=tokenizer,
            max_length=512,
            eval_subsets=[subset],
        )
        print(f"  {subset}: train={len(ds.train_dataset)}, test={len(ds.eval_datasets[subset])}")
    
    print("\n✓ All MMLU tests passed!")

if __name__ == "__main__":
    test_mmlu_loading()

