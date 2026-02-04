from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_tokenized_dataset, get_trainer
from pathlib import Path
import matplotlib.pyplot as plt

# Load your LoRA fine-tuned model from Tutorial 2
checkpoint_path = Path(__file__).parent.parent / "Tutorial_2" / "Tutorial_2_LoRA_Model"

try:
    base_mg = MaseGraph.from_checkpoint(str(checkpoint_path))
except FileNotFoundError:
    print("LoRA Fine-Tuned Model Not Found")
    exit()

checkpoint = "prajjwal1/bert-tiny"
tokenizer_checkpoint = "bert-base-uncased"
dataset_name = "imdb"

dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,
    return_tokenizer=True,
)

def test_quantization(bit_width, run_qat=True):
    print(f"Testing {bit_width}-bit Quantization")
    mg = MaseGraph.from_checkpoint(str(checkpoint_path))
    
    # Configure quantization
    # HEURISTIC: weights are small (max ~1.3), activations are larger (max ~8.0)
    # 8-bit: w_frac=6 (range 2.0), data_frac=4 (range 8.0)
    # 16-bit: w_frac=14, data_frac=12
    
    w_frac = bit_width - 2
    d_frac = bit_width - 4 if bit_width > 4 else 1 # Avoid 0 or negative for 4-bit
    
    quantization_config = {
        "by": "type",
        "default": {"config": {"name": None}},
        "linear": {
            "config": {
                "name": "integer",
                "data_in_width": bit_width, "data_in_frac_width": max(1, d_frac),
                "weight_width": bit_width, "weight_frac_width": max(1, w_frac),
                "bias_width": bit_width, "bias_frac_width": max(1, w_frac),
            }
        },
    }
    
    # Apply quantization
    print(f"Applying {bit_width}-bit quantization:")
    mg, _ = passes.quantize_transform_pass(mg, pass_args=quantization_config)
    
    # PTQ Evaluation
    print("Evaluating PTQ (Post-Training Quantization):")
    trainer = get_trainer(
        model=mg.model,
        tokenized_dataset=dataset,
        tokenizer=tokenizer,
        evaluate_metric="accuracy",
    )
    ptq_results = trainer.evaluate()
    ptq_accuracy = ptq_results["eval_accuracy"]
    print(f"PTQ Accuracy: {ptq_accuracy:.4f}")
    
    # QAT (Training)
    if run_qat:
        print("Running QAT (Quantization-Aware Training):")
        trainer.train()
        qat_results = trainer.evaluate()
        qat_accuracy = qat_results["eval_accuracy"]
        print(f"QAT Accuracy: {qat_accuracy:.4f}")
    else:
        qat_accuracy = None
    
    return ptq_accuracy, qat_accuracy

bit_widths = [4, 8, 16, 32]

# Storage for results
ptq_accuracies = []
qat_accuracies = []

for width in bit_widths:
    ptq_acc, qat_acc = test_quantization(width, run_qat=True)
    ptq_accuracies.append(ptq_acc)
    qat_accuracies.append(qat_acc)

print("\n\nRESULTS SUMMARY:")
for i, width in enumerate(bit_widths):
    print(f"{width}-bit: PTQ={ptq_accuracies[i]:.4f}, QAT={qat_accuracies[i]:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(bit_widths, ptq_accuracies, marker='o', label='PTQ', linewidth=2)
plt.plot(bit_widths, qat_accuracies, marker='s', label='QAT', linewidth=2)

plt.xlabel('Bit Width', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Quantization: Bit-Width vs Accuracy', fontsize=14)
plt.xticks(bit_widths)
plt.ylim(0.45, 0.85)
plt.legend()
plt.grid(True, alpha=0.3)

# Save the figure
save_path = Path(__file__).parent / "Lab_1_Quantisation.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# One Epoch

# 4-bit: PTQ=0.5000, QAT=0.5000
# 8-bit: PTQ=0.7758, QAT=0.8170
# 16-bit: PTQ=0.7798, QAT=0.8172
# 32-bit: PTQ=0.7797, QAT=0.8173