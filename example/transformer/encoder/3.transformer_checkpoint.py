from transformers import BertConfig, BertModel

# Create a model,effectively loaded the architecture of BERT and randomly initialized its weights. It is, however, fairly useless for inference at the moment as it hasn’t been trained yet! Training a model from scratch is a time-consuming and computationally expensive task so it’s often best to choose a pretrained model that is suitable for the problem we’d like to solve.
config  = BertConfig()
model = BertModel(config)

# Load a pretrained model from the Hugging Face model hub. This model has been trained on a large corpus of text and is therefore more suitable for inference.
model_trained = BertModel.from_pretrained("bert-base-cased")
