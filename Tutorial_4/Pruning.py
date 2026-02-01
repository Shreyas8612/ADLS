from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_tokenized_dataset, get_trainer
from pathlib import Path

checkpoint_path = Path(__file__).parent.parent / "Tutorial_3" / "Tutorial_3_QAT"

try:
    mg = MaseGraph.from_checkpoint(str(checkpoint_path))
except FileNotFoundError:
    print("Checkpoint not found! Please ensure you ran Tutorial 3.")
    exit()

dataset_name = "imdb"
tokenizer_checkpoint = "DeepWokLab/bert-tiny" 

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

pruning_config = {
    "weight": {
        "sparsity": 0.5,     # Remove 50% of the weak weights
        "method": "l1-norm", # Absolute values close to zero are weak
        "scope": "local", # Locally
    },
    "activation": {
        "sparsity": 0.5,
        "method": "l1-norm",
        "scope": "local",
    },
}

print("\nPruning:")
mg, _ = passes.prune_transform_pass(mg, pass_args=pruning_config)

trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
)
pruned_results = trainer.evaluate()
print(f"Pruned Accuracy: {pruned_results['eval_accuracy']:.4f}")

# Pruned Accuracy: 0.7069

print(f"Pruning for 5 epoch:")
trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
    num_train_epochs=5, 
)
trainer.train()

final_results = trainer.evaluate()
print(f"Final Accuracy: {final_results['eval_accuracy']:.4f}")

#Final Accuracy: 0.8132