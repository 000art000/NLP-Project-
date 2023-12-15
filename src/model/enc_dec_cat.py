import torch.nn as nn
from encoder_dir.encoder_scratch import encoder
from decoder_dir.decoder_cat import decoder_cat
import torch
import pytorch_lightning as pl
from loss import pen_l2


class Model(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers_enc, num_layers_dec, dropout=0.1):
        super(Model, self).__init__()
        self.encoder=encoder(vocab_size, embedding_dim, num_heads, num_layers_enc,dropout)
        self.decoder=decoder_cat((embedding_dim)*2, num_layers_dec)

    def forward(self ,input1, input2):
        #shape of input(1|2) : T,B

        output1_enc,output2_enc=self.encoder(input1,input2) # shape of output(1|2) : B,EMB

        input_dec= torch.cat((output1_enc,output2_enc), dim=1) #shape B,2*EMB

        output= self.decoder(input_dec) #shape B,1

        return output

class encodeur_decodeur_cat(pl.LightningModule):

    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers_enc, num_layers_dec,loss, dropout=0.1, learning_rate=1e-3,lambda_l=0.01):
        super().__init__()
        self.model = Model(vocab_size, embedding_dim, num_heads, num_layers_enc, num_layers_dec, dropout)
        self.learning_rate = learning_rate
        self.loss = loss
        self.lambda_l=lambda_l


    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size=vocab_size

        # Assigner les paramètres non utilisés à None
        self.model = None
        self.learning_rate = None
        self.loss = None
        self.embedding_dim = None 
        self.num_heads = None 
        self.num_layers_enc = None 
        self.num_layers_dec = None 
        self.dropout = None 
        self.lambda_l=None


    def forward(self,input1,input2):
        x = self.model(input1,input2)
        return x
    
    def training_step(self, batch, batch_idx):
        prompt,rps,score = batch
        logits = self(prompt,rps)
        loss = self.loss(logits, score) - self.lambda_l * pen_l2(self)
        return loss
    
    def validation_step(self, batch, batch_idx):
        prompt,rps,score = batch
        logits = self(prompt,rps)
        loss = self.loss(logits, score)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def complet(self, grid):
        # Instanciation du modèle avec les paramètres de la grille
        embedding_dim = grid.get('embedding_dim')
        num_heads = grid.get('num_heads')
        num_layers_enc = grid.get('num_layers_enc')
        num_layers_dec = grid.get('num_layers_dec')
        dropout = grid.get('dropout')

        self.learning_rate = grid.get('learning_rate')
        self.loss = grid.get('loss')
        self.lambda_l = grid.get('lambda_l')
  
        # Réinstanciation du modèle avec les nouveaux paramètres
        self.model = Model(self.vocab_size, embedding_dim, num_heads, num_layers_enc, num_layers_dec, dropout)

    
