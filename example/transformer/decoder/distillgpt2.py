from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Setup
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

text = "Let's turn this string into a PyTorch tensor of tokens."

## YOUR SOLUTION HERE ##
pt_tokens = tokenizer(text, return_tensors="pt")
print(pt_tokens)

list_tokens = [1532, 345, 821, 3555, 428, 11, 345, 875, 9043, 502, 0]

## YOUR SOLUTION HERE ##
decoded_tokens = tokenizer.decode(list_tokens)
print(decoded_tokens)

prompt = "Hello, my name is"

inputs = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(inputs, max_length=75, num_beams = 1, do_sample = True, pad_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(output[0]))

from transformers import set_seed
set_seed(10)

prompt = "Artificial intelligence is"

def generate_text(prompt, temperature):
    input = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input, max_length=75,num_return_sequences=1, do_sample=True, temperature=temperature, pad_token_id=tokenizer.eos_token_id)
    return f"\n---\n{tokenizer.decode(output[0]).strip()}\n---\n"

## YOUR SOLUTION HERE ##
high_temp = 1.2
print(high_temp)
print(generate_text(prompt, high_temp))

low_temp = 0.3
print(low_temp)
print(generate_text(prompt, low_temp))