from transformers import AutoTokenizer

# Breaking text into tokens
sample_sentence = "Brevity is the soul of wit."

## YOUR SOLUTION HERE ##
tokenzed_array = sample_sentence.split()
print(tokenzed_array)


# Wordpiece tokenization
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

## YOUR SOLUTION HERE ##
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_array_wordpiece = tokenizer.tokenize(sample_sentence)
print(tokenized_array_wordpiece)

# Encoding
sequence = "I've been waiting for a HuggingFace course my whole life."

## YOUR SOLUTION HERE ##
input_tokens = tokenizer(sequence)
print(input_tokens)

# Decoding
## YOUR SOLUTION HERE ##
decoded_sequence = tokenizer.decode(input_tokens['input_ids'])
print(decoded_sequence)

# Batching
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

##YOUR SOLUTION HERE
model_inputs_batched = tokenizer(sequences)
print(model_inputs_batched)

# Padding
## Padding
# Will pad the sequences up to the maximum sequence length
model_inputs_padded_1 = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs_padded_2 = tokenizer(sequences, padding="max_length")


## YOUR SOLUTION HERE ##
# Pad the sequences up to the specified max length
model_inputs_padded_3 = tokenizer(sequences, padding="max_length", max_length=8)
print(model_inputs_padded_3)

## Truncation
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs_truncated_1 = tokenizer(sequences, truncation=True)
print(model_inputs_truncated_1)

## YOUR SOLUTION HERE ##
# Truncate the sequences that are longer than the specified max length
model_inputs_truncated_2 = tokenizer(sequences, max_length=8, truncation=True)
print(model_inputs_truncated_2)