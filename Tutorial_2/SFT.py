from chop.passes.module import report_trainable_parameters_analysis_pass
from transformers import AutoModelForSequenceClassification
from chop import MaseGraph
import chop.passes as passes
from chop.tools import get_trainer
from pathlib import Path
from chop.tools import get_tokenized_dataset

checkpoint = "DeepWokLab/bert-tiny" 
tokenizer_checkpoint = "DeepWokLab/bert-tiny"
dataset_name = "imdb"

# Load and Tokenize
# This utility wraps HuggingFace's load_dataset and AutoTokenizer
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,  # Points to the specific pre-trained model on HuggingFace
    return_tokenizer=True,
)

# Load the "Raw" Model
# We tell it "problem_type" so it knows to stick a classification head on top
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
model.config.problem_type = "single_label_classification"

# Create the MaseGraph
# We explicitly list 'labels' so the graph includes the loss calculation layers
mg = MaseGraph(
    model,
    hf_input_names=[
        "input_ids",      # The words (as numbers -> Tokens)
        "attention_mask", # Tells model which words are real vs padding
        "labels",         # The correct answer (Positive/Negative)
    ],
)

# This calculates shapes for every layer
mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(mg)

print("\nInitial Parameter Count: ")
# This prints a table of every layer and how many parameters it has
_, _ = report_trainable_parameters_analysis_pass(mg.model)

# Freeze the "Embeddings"
# Informing PyTorch to "NOT" update these embedding weights
print("\n--- Freezing Embeddings ---")
for param in mg.model.bert.embeddings.parameters():
    param.requires_grad = False  # "False" means "Do not train"

print("After Freezing")
# You should see the 'bert.embeddings' count go to 0
_, pass_info = report_trainable_parameters_analysis_pass(mg.model)

# Before Freezing
# Total Trainable Parameters: 14480258

# After Freezing
# Total Trainable Parameters: 2561666

# The trainer handles all the complex math of updating weights
trainer = get_trainer(
    model=mg.model,
    tokenized_dataset=dataset,     # Tokenuzed IMDb dataset
    tokenizer=tokenizer,           # The translator to numbers
    evaluate_metric="accuracy",
)

print("\nBaseline Evaluation:")
# Since the model is yet to learn it is likely to have accuracy of 50%
eval_results = trainer.evaluate()
print(f"Baseline Accuracy: {eval_results['eval_accuracy']:.4f}")

# Baseline Accuracy: 0.5090

print("\nTrain for Single Epoch")
trainer.train()

print("\nPost-Training Evaluation:")
eval_results = trainer.evaluate()
print(f"Accuracy after One Epoch: {eval_results['eval_accuracy']:.4f}")

# Accuracy after one epoch: 0.7732

# We save this so we don't have to train it again later.
save_path = Path(__file__).parent / "Tutorial_2_SFT_Model"
mg.export(str(save_path))
print("SFT Model Saved!")