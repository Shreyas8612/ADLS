from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_tokenized_dataset, get_trainer
from pathlib import Path
import matplotlib.pyplot as plt
import torch

checkpoint_path = Path(__file__).parent.parent / "Tutorial_3" / "Tutorial_3_QAT"

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

def test_pruning(sparsity, method, run_finetuning=True):
    # Load the 8-bit base model fresh
    mg = MaseGraph.from_checkpoint(str(checkpoint_path))

    # Configure pruning
    pruning_config = {
        "weight": {
            "sparsity": sparsity,
            "method": method,
            "scope": "local",
        },
        "activation": {
            "sparsity": sparsity,
            "method": method,
            "scope": "local",
        },
    }
    
    # Apply pruning
    print(f"Applying {sparsity*100:.0f}% pruning with {method}:")
    mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)
    
    # Evaluate immediately after pruning
    trainer = get_trainer(
        model=mg.model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
        num_train_epochs=3,
    )
    pruned_results = trainer.evaluate()
    pruned_accuracy = pruned_results["eval_accuracy"]
    print(f"Pruned Accuracy: {pruned_accuracy:.4f}")
    
    # Fine-tune to recover
    if run_finetuning:
        print("Fine-tuning pruned model:")
        trainer.train()
        finetuned_results = trainer.evaluate()
        finetuned_accuracy = finetuned_results["eval_accuracy"]
        print(f"Fine-tuned Accuracy: {finetuned_accuracy:.4f}")
    else:
        finetuned_accuracy = None
    
    return pruned_accuracy, finetuned_accuracy

sparsities = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]

# Storage for results
random_accuracies = []
l1norm_accuracies = []

# Test with Random method
print("\n\nTESTING RANDOM PRUNING:")
for sparsity in sparsities:
    _, acc = test_pruning(sparsity, method="random", run_finetuning=True)
    random_accuracies.append(acc)

# Test with L1-Norm method
print("\n\nTESTING L1-NORM PRUNING:")
for sparsity in sparsities:
    _, acc = test_pruning(sparsity, method="l1-norm", run_finetuning=True)
    l1norm_accuracies.append(acc)

print("\n\nRESULTS SUMMARY")
print("Sparsity | Random | L1-Norm")
print("-" * 35)
for i, sp in enumerate(sparsities):
    print(f"{sp*100:5.0f}%   | {random_accuracies[i]:.4f} | {l1norm_accuracies[i]:.4f}")

#RESULTS SUMMARY
#Sparsity | Random | L1-Norm
#-----------------------------------
#   10%   | 0.8208 | 0.8408
#   20%   | 0.7990 | 0.8358
#   40%   | 0.7280 | 0.8204
#   60%   | 0.5092 | 0.7982
#   80%   | 0.5077 | 0.6323
#   90%   | 0.5006 | 0.5124

plt.figure(figsize=(10, 6))

# Convert sparsity to percentage for better readability
sparsity_pct = [s * 100 for s in sparsities]

plt.plot(sparsity_pct, random_accuracies, marker='o', label='Random Pruning', linewidth=2)
plt.plot(sparsity_pct, l1norm_accuracies, marker='s', label='L1-Norm Pruning', linewidth=2)

plt.xlabel('Sparsity (%)', fontsize=12)
plt.ylabel('Accuracy (After Fine-tuning)', fontsize=12)
plt.title('Pruning on 8-bit Model: Sparsity vs Accuracy', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Add a horizontal line showing the baseline (8-bit QAT from Task 1)
plt.axhline(y=0.813, color='r', linestyle='--', alpha=0.5, label='8-bit Baseline (No Pruning)')
plt.legend()

# Save the figure
save_path = Path(__file__).parent / "Lab_1_Pruning.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()