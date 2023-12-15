import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, predictions, labels):
        mse = F.mse_loss(predictions.squeeze(1), labels)
        rmse = torch.sqrt(mse)
        return rmse
    
def pen_l2(model):
    l2_reg=0
    for param in model.parameters():
        l2_reg += torch.norm(param)

    return l2_reg

"""
from encoder_dir.LLM_encoder import LLM_encoder
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import LongformerTokenizer, LongformerModel
import functools
from Data_loader.LLM_loder import CustomDataset,collate_fn


# Initialize the tokenizer for your model (replace 'bert-base-uncased' with your model)
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerModel.from_pretrained('allenai/longformer-base-4096')


path = "../data/dataset.csv"
dataset = CustomDataset(path)

partial_collate_fn = functools.partial(collate_fn, LLM_tokenizer=tokenizer)
test_loader = DataLoader(dataset, collate_fn=partial_collate_fn, batch_size=1)
model=LLM_encoder(model)


def fun_cls(outputs1,outputs2):
    cls_output1 = outputs1.last_hidden_state[:, 0, :]
    cls_output2 = outputs2.last_hidden_state[:, 0, :]

    return cls_output1, cls_output2

for prompts_data, prompt_masks, reps_data, reps_masks, score_list in test_loader:
    #print(prompts_data.size(), prompt_masks.size(), reps_data.size(), reps_masks.size(), score_list.size())
    output1, output2 = model(prompts_data, prompt_masks, reps_data, reps_masks,fun_cls)
    print(output1.size(), output2.size())

"""