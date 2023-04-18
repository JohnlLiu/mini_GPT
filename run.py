import torch
import torch.nn as nn
from torch.nn import functional as F
from model import BigramLanguageModel
from model import encode
from model import decode


#load saved and trained model
saved_model = BigramLanguageModel()
saved_model.load_state_dict(torch.load('GPT_model.pt'))

new_tokens = 500

#run with input text
while True:
    text = input()
    if text == 'quit':
        break

    context = torch.tensor([encode(text)], dtype = torch.long)
    #context = torch.tensor([[1,2,3,4,5],[12,3,23,15,8]])
    print(decode(saved_model.generate(context, max_new_tokens = new_tokens)[0].tolist()))
    print('\n')