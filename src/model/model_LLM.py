import torch.nn as nn
from encoder_dir.LLM_encoder import LLM_encoder
from decoder_dir.decoder_cat import decoder_cat
import torch
import pytorch_lightning as pl
from loss import pen_l2

class encodeur_decodeur_cat(pl.LightningModule):

    def __init__(self, LLM_model, loss, num_layers_dec, learning_rate=1e-3):
        super().__init__()

        self.enocder = LLM_encoder(LLM_model)

        self.learning_rate = learning_rate
        self.loss = loss
        self.decoder=decoder_cat((LLM_model.config.hidden_size)*2, num_layers_dec)


    def __init__(self, LLM_model,fun_cls):
        super().__init__()

        self.enocder = LLM_model

        # Assigner les paramètres non utilisés à None
        self.learning_rate = None
        self.loss = None
        self.num_layers_dec=None
        self.fun_cls=fun_cls


    def forward(self,input1,mask1,input2,mask2):
        output1_enc,output2_enc = self.enocder(input1,mask1,input2,mask2,self.fun_cls)
        input_dec= torch.cat((output1_enc,output2_enc), dim=1) #shape B,2*EMB
        output= self.decoder(input_dec) #shape B,1

        return output
    
    def training_step(self, batch, batch_idx):
        prompts_data, prompt_masks, reps_data, reps_masks, score_list= batch
        logits = self(prompts_data, prompt_masks, reps_data, reps_masks)
        loss = self.loss(logits, score_list) - self.lambda_l * pen_l2(self.decoder)
        return loss.float()
    
    def validation_step(self, batch, batch_idx):
        prompts_data, prompt_masks, reps_data, reps_masks, score_list= batch
        logits = self(prompts_data, prompt_masks, reps_data, reps_masks)
        loss = self.loss(logits, score_list)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def complet(self, grid):
        # Instanciation du modèle avec les paramètres de la grille
        self.learning_rate = grid.get('learning_rate')
        self.loss = grid.get('loss')
        self.num_layers_dec= grid.get('num_layers_dec')
        self.lambda_l= grid.get('lambda_l')

        #complet
        self.decoder=decoder_cat((self.enocder.config.hidden_size)*2, self.num_layers_dec)
        self.enocder = LLM_encoder(self.enocder)

    
