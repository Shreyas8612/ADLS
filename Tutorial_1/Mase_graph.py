from transformers import AutoModelForSequenceClassification
from chop import MaseGraph

model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")

mg = MaseGraph(model)
mg.draw("Tutorial_1/bert_tiny.svg")