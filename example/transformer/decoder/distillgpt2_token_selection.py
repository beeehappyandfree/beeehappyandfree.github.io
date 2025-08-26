# Important: Don't run this cell more than once.
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")


from transformers import set_seed
set_seed(10) # Don't change this value.

prompt = "The capital of France is"
inputs = tokenizer.encode(prompt, return_tensors="pt")

greedy_outputs = model.generate(inputs, max_length=50, pad_token_id=tokenizer.eos_token_id) #(Don't mind the `pad_token_id` bit there; it's just included to avoid a warning message.)

print(tokenizer.decode(greedy_outputs[0]))

prompt = "The capital of France is"
inputs = tokenizer.encode(prompt, return_tensors="pt")

## YOUR SOLUTION HERE ##
beam_outputs = model.generate(
    inputs, 
    max_length=50,
    early_stopping=True,
    pad_token_id=tokenizer.eos_token_id,
    num_beams = 5
)

print(tokenizer.decode(beam_outputs[0]))

prompt = "The capital of France is"
inputs = tokenizer.encode(prompt, return_tensors="pt")

## YOUR SOLUTION HERE ##
ngram_outputs = model.generate(
    inputs, 
    max_length=50,
    num_beams=5, 
    early_stopping=True,
    pad_token_id=tokenizer.eos_token_id,
    no_repeat_ngram_size = 2    
)


print(tokenizer.decode(ngram_outputs[0])) 

prompt = "The capital of France is"
inputs = tokenizer.encode(prompt, return_tensors="pt")


set_seed(10) # Because sampling involves random number generation, we need to set the seed an additional time.

## YOUR SOLUTION HERE ##
sample_outputs = model.generate(
    inputs,
    no_repeat_ngram_size=2,
    max_new_tokens=40,
    pad_token_id=tokenizer.eos_token_id,
    do_sample = True,
    temperature = 0.6,
    top_k = 50
)

print(tokenizer.decode(sample_outputs[0]))

