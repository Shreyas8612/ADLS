from chop.tools import get_tokenized_dataset

checkpoint = "DeepWokLab/bert-tiny" 
tokenizer_checkpoint = "DeepWokLab/bert-tiny"
dataset_name = "imdb"

print(f"Loading and tokenizing {dataset_name} using {tokenizer_checkpoint}...")

# Load and Tokenize
# This utility wraps HuggingFace's load_dataset and AutoTokenizer
dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=tokenizer_checkpoint,  # Points to the specific pre-trained model on HuggingFace
    return_tokenizer=True,
)

print("\nDataset Info: ")
print(dataset)
print("\nExample Tokenization: ")
print(f"Keys: {dataset['train'][0].keys()}")

# Output is commented Below
"""
Dataset Info: 
DatasetDict({
    train: Dataset({
        features: ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 50000
    })
})

Example Tokenization: 
Keys: dict_keys(['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask'])
"""