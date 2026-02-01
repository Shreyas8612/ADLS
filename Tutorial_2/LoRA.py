from chop import MaseGraph
import chop.passes as passes
from chop.passes.module import report_trainable_parameters_analysis_pass
from transformers import AutoModelForSequenceClassification
from chop.tools import get_trainer
from pathlib import Path
from chop.tools import get_tokenized_dataset

checkpoint = "DeepWokLab/bert-tiny"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.config.problem_type = "single_label_classification"

mg = MaseGraph(
    model,
    hf_input_names=["input_ids", "attention_mask", "labels"],
)

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(mg)

# This pass finds "Linear" layers and replaces them with "LoRALinear" layers
mg, _ = passes.insert_lora_adapter_transform_pass(
    mg,
    pass_args={
        "rank": 6,       # Size of the r for A and B matrices
        "alpha": 1.0,    # Scaling factor (strength of the adapter)
        "dropout": 0.5,  # Randomly ignore some connections (prevents overfitting)
    },
)

# Freeze Embedding
for param in mg.model.bert.embeddings.parameters():
    param.requires_grad = False  # "False" means "Do not train"

# Check the number of parameters trainable compared to SFT
print("\nLoRA Parameter Count:")
_, _ = report_trainable_parameters_analysis_pass(mg.model)

# With SFT including Embedding
# Total Trainable Parameters: 14480258

# With LoRA including Embedding
# Total Trainable Parameters: 15088408

# With SFT without Embedding
# Total Trainable Parameters: 2561666

# With LoRA without Embedding
# Total Trainable Parameters: 3169816

tokenizer_checkpoint = "DeepWokLab/bert-tiny"
dataset_name = "imdb"

# Load and Tokenize
# This utility wraps HuggingFace's load_dataset and AutoTokenizer
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,  # Points to the specific pre-trained model on HuggingFace
    return_tokenizer=True,
)

trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
    num_train_epochs=1,
)

print("\nLoRA Training for one Epoch")
trainer.train()

eval_results = trainer.evaluate()
print(f"LoRA Final Accuracy: {eval_results['eval_accuracy']:.4f}")

# LoRA Final Accuracy: 0.7790

# Replacing each LoRALinear with an nn.Linear where AB product is added to the original weight matrix W
print("\n--- Fusing LoRA Weights ---")
mg, _ = passes.fuse_lora_weights_transform_pass(mg)

# The math ensures the behavior doesn't change, only the structure does.
eval_results = trainer.evaluate()
print(f"Fused Model Accuracy: {eval_results['eval_accuracy']:.4f}")

save_path = Path(__file__).parent / "Tutorial_2_LoRA_Model"
mg.export(str(save_path))
print("LoRA Model Saved!")