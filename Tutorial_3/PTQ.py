# Post Training Quantisation
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_tokenized_dataset, get_trainer
from pathlib import Path

# Load the model from Tutorial_2
checkpoint_path = Path(__file__).parent.parent / "Tutorial_2" / "Tutorial_2_LoRA_Model"

try:
    mg = MaseGraph.from_checkpoint(str(checkpoint_path))
except FileNotFoundError:
    print("Checkpoint not found! Please ensure you ran Tutorial 2.")
    exit()

dataset_name = "imdb"
tokenizer_checkpoint = "DeepWokLab/bert-tiny" 

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

# We tell Mase: "Hey! Turn all Linear layers into 8-bit Integers"
quantization_config = {
    "by": "type",  # Apply quantisation based on mase_op
    "default": {
        "config": {"name": None} # Don't touch other layers
    },
    "linear": {
        "config": {
            "name": "integer",
            # We give the model a budget of 8 bits per number
            # 4 bits for the whole number, 4 bits for the decimal part
            "data_in_width": 8, "data_in_frac_width": 4, 
            "weight_width": 8, "weight_frac_width": 4,
            "bias_width": 8, "bias_frac_width": 4,
        }
    },
}

# 5. Run the Quantization Pass
print("\nRunning Quantization Pass (PTQ):")
mg, _ = passes.quantize_transform_pass(
    mg,
    pass_args=quantization_config,
)

print("\nQuantized Evaluation (8-bit)")
trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,
    tokenizer=tokenizer,
    evaluate_metric="accuracy",
)
q_results = trainer.evaluate()
print(f"Quantized Accuracy: {q_results['eval_accuracy']:.4f}")

# Quantized Accuracy: 0.7174

save_path = Path(__file__).parent / "Tutorial_3_PTQ"
mg.export(str(save_path))