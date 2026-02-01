# Quantised Aware Training
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_tokenized_dataset, get_trainer
from pathlib import Path

checkpoint_path = Path(__file__).parent / "Tutorial_3_PTQ"

try:
    mg = MaseGraph.from_checkpoint(str(checkpoint_path))
except FileNotFoundError:
    print("Checkpoint not found! Please ensure you ran PTQ")
    exit()

print("\nStarting QAT (Quantisation-Aware Training)")
dataset_name = "imdb"
tokenizer_checkpoint = "DeepWokLab/bert-tiny" 

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
)

# The model is already quantised so PyTorch simulates quantised Training
trainer.train()

print("\nPost-QAT Evaluation:")
qat_results = trainer.evaluate()
print(f"QAT Final Accuracy: {qat_results['eval_accuracy']:.4f}")

# QAT Final Accuracy: 0.8139

save_path = Path(__file__).parent / "Tutorial_3_QAT"
mg.export(str(save_path))