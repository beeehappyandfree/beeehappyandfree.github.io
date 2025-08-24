from transformers import BertConfig, BertModel
config  = BertConfig()
model = BertModel(config)


model_trained = BertModel.from_pretrained("bert-base-cased")
