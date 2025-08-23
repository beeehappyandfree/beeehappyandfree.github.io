# Importing the packages

# We have successfully reproduced the three steps of the pipeline: preprocessing with tokenizers, passing the inputs through the model, and postprocessing. Now let’s take some time to dive deeper into each of those steps.


from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Setting the checkpoint
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# Initializing the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Raw inputs
raw_inputs = ["I've been waiting to learn about transformers my whole life.",
               "I hate this so much!"]

## YOUR SOLUTION HERE ##
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
# Initializing the model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

## YOUR SOLUTION HERE ##
outputs = model(**inputs)

print(outputs.logits.shape)
print(outputs.logits)

import torch

# Converting the tensor output to a probability distribution
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

## YOUR SOLUTION HERE ##
print(predictions)

