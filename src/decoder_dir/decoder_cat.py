import torch.nn as nn
import torch

"""
class res_bloc(nn.Module):
    def __init__(self, input_size, dropout=0.1):

        self.dense=nn.dense(input_size,input_size)
        self.bn=nn.BatchNorm1d(input_size)

    def forward(self,x):
        # Appliquer la couche dense
        out = self.dense(x)
        
        # Appliquer la normalisation par lots
        out = self.bn(out)
        
        # Appliquer une fonction d'activation (par exemple, ReLU)
        out = torch.relu(out)

        # Ajouter la connexion résiduelle
        out = x + out

        return out
"""

class decoder_cat(nn.Module):
    """
        decoder qui concaten les embedding suivie d'un NN simple
    """

    def __init__(self, input_size, num_layers):
        super(decoder_cat, self).__init__()
        assert num_layers > 0 , "error num_layers < 0 choose value for num_layers > 0"
        assert input_size//(2**num_layers) > 1, "you have to much layers compared to your embedding size"

        self.layers=[]

        for _ in range(num_layers):
            self.layers.append(nn.Linear(input_size,input_size//2))
            input_size=input_size//2
            self.layers.append(nn.BatchNorm1d(input_size))

        self.last_dns = nn.Linear(input_size, 1) 
        self.last_bn = nn.BatchNorm1d(1)
        
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()

    def forward(self, x):
        for i,layer in enumerate(self.layers):
            # i=2*k donc j'applique une dense layer
            # i=2*k+1 donc j'applique une batchnorm
            layer=layer.to(x.device)
            x = layer(x)

            # si je suis en train d'appliquer une batchnorm elle sera suivi d'une activation
            if i % 2 == 1:
                x = self.relu(x)
        
        x = self.last_dns(x)
        x = self.last_bn(x)
        x = self.tanh(x)

        return x