import torch
import torch.fx as fx
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from chop import MaseGraph
from chop.tools import get_logger
import chop.passes as passes
from pathlib import Path

logger = get_logger("mase_logger")
logger.setLevel("INFO")

def count_dropout_analysis_pass(mg, pass_args={}):
    dropout_modules = 0
    
    logger.info("--- Starting Dropout Count ---")
    for node in mg.fx_graph.nodes:
        # Check if the node is a module call and contains 'dropout' in its name
        if node.op == "call_module" and "dropout" in node.target:
            logger.info(f"Found dropout module: {node.target}")
            dropout_modules += 1
            
    return mg, {"dropout_count": dropout_modules}

def remove_dropout_transform_pass(mg, pass_args={}):
    logger.info("--- Starting Dropout Removal ---")
    
    for node in list(mg.fx_graph.nodes):
        if node.op == "call_module" and "dropout" in node.target:
            logger.info(f"Removing dropout module: {node.target}")

            # Find who is feeding data into dropout (the parent)
            # node.args[0] is the input to this node
            parent_node = node.args[0]
            
            # Connect the parent directly to anyone who was listening to the dropout node
            node.replace_all_uses_with(parent_node) # Important

            # Now it is safe to erase
            mg.fx_graph.erase_node(node)

    return mg, {}

model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
mg = MaseGraph(model)

# Count Dropouts (Should be non-zero)
mg, pass_out = count_dropout_analysis_pass(mg)
initial_count = pass_out['dropout_count']
print(f"Initial Dropout Count: {initial_count}")

# Remove Dropouts
mg, _ = remove_dropout_transform_pass(mg)

# Verify (Should be zero)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dummy_input = tokenizer(
    ["AI may take over the world one day", "This is why you should learn ADLS"],
    return_tensors="pt",
)

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(
    mg, pass_args={"dummy_in": dummy_input, "add_value": False}
)

mg, pass_out = count_dropout_analysis_pass(mg)
final_count = pass_out['dropout_count']
print(f"Final Dropout Count: {final_count}")

assert final_count == 0, "Error: Dropouts were not fully removed!"
print("Verification Successful: All dropouts removed.")

# Export
save_path = Path(__file__).parent / "Tutorial_1_checkpoint"
mg.export(str(save_path))

print("Loading graph back from checkpoint...")
new_mg = MaseGraph.from_checkpoint(str(save_path))
print("Graph loaded successfully. Tutorial 1 Complete!")