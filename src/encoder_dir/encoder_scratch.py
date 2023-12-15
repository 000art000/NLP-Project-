import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dropout=0.1):
        super(encoder, self).__init__()

        # Couche d'embedding
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Positional Encoding (nécessaire pour les modèles transformer)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout)

        # Configuration de TransformerEncoderLayer
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=embedding_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)

    def forward_once(self, x):
        # x : T,B
        x = self.embedding(x) #x : T,B,EMB 
        x = self.positional_encoding(x) #x : (T+1),B,EMB 
        x = self.transformer_encoder(x) #x: (T+1),B,EMB 
        return x[0] # B,EMB

    def forward(self, input1, input2):
        """
            j'ai 2 text en entrée
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1,output2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x : T,B,EMB 

        # Récupérer la taille de la séquence
        seq_length = x.size(0) # T

        #creer le positionel embedding
        pe = torch.zeros(seq_length, self.d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1).repeat(1, x.size(1), 1)  # Adapter à la taille du batch (T,B,EMBS)
       
        pe = pe.to(x.device)       
        # Ajouter le positionnel embedding
        x+=pe

        # Créer un tensor "cls" initialisé avec des 1
        cls_token = torch.ones(1, x.size(1), self.d_model) # 1,B,EMB

        # Concaténer le tensor "cls" au début de chaque séquence du batch
        # pour stocké l'info global de mon text dans ce embedding de faire une moyenne de mes embedding de chaque mot
        cls_token = cls_token.to(x.device)
        x = torch.cat((cls_token, x), dim=0) # x : (T+1),B,EMB 

        return self.dropout(x)