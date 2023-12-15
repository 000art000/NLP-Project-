import torch.nn as nn
import torch


class LLM_encoder(nn.Module):
    def __init__(self, LLM_model):
        super().__init__()

        self.LLM = LLM_model
        
        # Geler les paramètres du modèle
        for param in self.LLM.parameters():
            param.requires_grad = False

    def forward(self, input_1, attention_mask_1,input_2, attention_mask_2,fun_cls):
        """
            input1/attention_mask_1 : shape B,T1
            input2/attention_mask_2 : shape B,T2
            fun_cls : fonction to Extracting cls 
        """
        outputs1 = self.LLM(input_ids=input_1, attention_mask=attention_mask_1)
        outputs2 = self.LLM(input_ids=input_2, attention_mask=attention_mask_2)

        # Extracting the [CLS] token's representation from each output
        cls_output1,cls_output2 =fun_cls(outputs1,outputs2)

        return cls_output1, cls_output2
