# ## Importing pipeline from transformers package
# import torch
# from transformers import pipeline

# ## Check if MPS (Metal Performance Shaders) is available for M2 chip
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"Using device: {device}")

# ## Setting up a sentiment analysis classifier with M2 GPU support
# classifier = pipeline(
#     task="sentiment-analysis", 
#     model="distilbert-base-uncased-finetuned-sst-2-english",
#     device=device
# )

# print(classifier)

# ## Example usage
# text = "I love using my M2 Mac for machine learning!"
# result = classifier(text)
# print(f"Text: {text}")
# print(f"Sentiment: {result}")


from transformers import pipeline
classifier = pipeline(task = "sentiment-analysis", 
                     model = "distilbert-base-uncased-finetuned-sst-2-english")  
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
print(classifier(raw_inputs))
