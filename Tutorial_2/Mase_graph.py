from transformers import AutoModelForSequenceClassification
from chop import MaseGraph
import chop.passes as passes

# Load the "Raw" Model
# We tell it "problem_type" so it knows to stick a classification head on top
checkpoint = "DeepWokLab/bert-tiny"
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

print(f"Graph Nodes: {len(mg.fx_graph.nodes)}")

mg = MaseGraph(model)
mg.draw("Tutorial_2/inp_att_label.svg")
# For the below graph comment out 'attention_mask' and 'labels' from hf_input_names
# mg.draw("Tutorial_2/Only_Input.svg")

# Graph Nodes - 115 for Only_Input and no Loss implemented
# Graph Nodes - 117 for inp_att_label and Loss implemented

