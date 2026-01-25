import chop.passes as passes
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from chop import MaseGraph
import torch

model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
mg = MaseGraph(model)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dummy_input = tokenizer(
    ["AI may take over the world one day", "This is why you should learn ADLS"],
    return_tensors="pt",
)

mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(
    mg, pass_args={"dummy_in": dummy_input, "add_value": False}
)

for i, node in enumerate(mg.fx_graph.nodes):
    if i >= 10:
        break
    print(node.op, node.target, "mase" in node.meta)