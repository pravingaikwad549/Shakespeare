# d_model in our case is 1629 and seq length is 100
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from model import model_v, Embedding, LayerNorm, MyPositionalEncoding, MultiheadAttnBlock, feed_forward, final_layer
# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

seq_length = 8
d_model = 64
h = 4
hidden_dim = 256
mask =  torch.tril(torch.ones(seq_length, seq_length))
model = model_v(Embedding, LayerNorm, MyPositionalEncoding, MultiheadAttnBlock,feed_forward, d_model, final_layer, seq_length, hidden_dim, h=4, vocab_size = vocab_size, mask = mask)

model.train()
optimizer = torch.optim.SGD(params = model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    epoch_loss = 0.0
    xb, yb = get_batch('train')
    optimizer.zero_grad()
    output = model(xb)
    loss = criterion(output.view(-1, vocab_size), yb.view(-1))
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    print("Epoch:", epoch + 1, "Loss:", epoch_loss)

def reshape_tensor(input_tensor, target_shape):
    num_elements = input_tensor.numel()
    flattened_tensor = input_tensor.view(-1)
    if num_elements < target_shape[1]:
        padded_tensor = torch.zeros(target_shape[1])
        padded_tensor[-num_elements:] = flattened_tensor
    else:
        padded_tensor = flattened_tensor[-target_shape[1]:]    
    reshaped_tensor = padded_tensor.view(target_shape)
    return reshaped_tensor.type(torch.long)

model.eval()
inputs = "Once upon a time,"
lst = []
for i in range(100):
    inputs_t = torch.tensor(encode(inputs), dtype=torch.long)
    inputs_t = inputs_t.unsqueeze(0)
    inputs_t = reshape_tensor(inputs_t, (1, seq_length))
    lst.append(decode(torch.argmax(model(inputs_t)[:,-1,:]).view(-1).tolist()))
    inputs += decode(torch.argmax(model(inputs_t)[:,-1,:]).view(-1).tolist())
"".join(lst)
print(inputs)