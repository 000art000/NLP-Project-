from typing import List
from torch.utils.data import Dataset
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import torch

class Vocabulary :
    
    BLANK=0
    START_TEXT=1
    END_TEXT=2
    START_QST=3
    END_QST=4

    def __init__(self):
        self.id2word = [ "BLANK", 'start_text', 'end_text', 'start_question', 'end_question']

        self.word2id = { "BLANK" : Vocabulary.BLANK, "start_text" : Vocabulary.START_TEXT,
                         "end_text" : Vocabulary.END_TEXT, "start_question" : Vocabulary.START_QST, 'end_question': Vocabulary.END_QST
                        }

    def __getitem__(self, word: str):
        return self.word2id[word]
        
    def get(self, word: str):
        try:
            return self.word2id[word]
        except KeyError:
            #si le mot n'existe pas en le rajoute au vocabulaire
            wordid = len(self.id2word)
            self.word2id[word] = wordid
            self.id2word.append(word)
            return wordid

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        #convertir un idx a un mot
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        #convertir une list d'idx a une liste de mots
        return [self.getword(i) for i in idx]



########################################################################################################################################################## 
                                            ### Dataset ###
class CustomDataset(Dataset):
    def __init__(self, path_file, words: Vocabulary):

        self.data = []

        data = pd.read_csv(path_file)

        for _, row in data.iterrows():
            prompt = row['prompt']
            content = row['content']
            rep = row['text']

            # Convertir en indices de vocabulaire
            prompt_indices = [words.get(token) for token in prompt.split(' ')]
            rep_indices = [words.get(token) for token in rep.split(' ')]

            self.data.append((prompt_indices, rep_indices, content))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ix):
        return self.data[ix]
    


########################################################################################################################################################
                                        ### LightningDataModule ###    

class MyDataModule(LightningDataModule):
    def __init__(self, path_file, batch_size=32, num_workers=1):
        super().__init__()

        #param pour CustomDataset
        self.path_file = path_file
        self.words = Vocabulary()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_paquet=None
        self.idx_eval=None
        self.full_dataset=CustomDataset(self.path_file, self.words)

    def get_size_voc(self):
        return len(self.words)

    def set_paquet(self,nbr_paquet):
        self.k_paquet=nbr_paquet

    def set_idx_eval(self,idx):
        self.idx_eval=idx
        
    def get_nbr_paquet(self):
        return self.k_paquet

    def setup(self, stage=None):
        assert self.k_paquet is not None and self.idx_eval is not None, 'vous devez donner nombre de paquet et l"index du paquet d"eval'

        assert self.idx_eval<self.k_paquet, "idx_eval doit etre < k_paquet"

        # Taille de chaque paquet pour la validation croisée
        fold_size = len(self.full_dataset) // self.k_paquet

        # Division séquentielle des indices en k paquets
        folds = [range(i * fold_size, (i + 1) * fold_size) for i in range(self.k_paquet)]

        # Si les données ne se divisent pas exactement, ajoutez les données restantes au dernier paquet
        if len(self.full_dataset) % self.k_paquet != 0:
            folds[-1] = range((self.k_paquet - 1) * fold_size, len(self.full_dataset))

        # Sélectionner le paquet de test et les indices d'entraînement
        test_indices = folds[self.idx_eval]
        train_indices = [idx for i, fold in enumerate(folds) if i != self.idx_eval for idx in fold]

        # Créer les ensembles de données d'entraînement et de test
        self.train_dataset = Subset(self.full_dataset, train_indices)
        self.val_dataset = Subset(self.full_dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,collate_fn=collate_fn ,shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,collate_fn=collate_fn ,num_workers=self.num_workers)



###################################################################################################################################################

def collate_fn(batch):
    src, trg, content = zip(*batch)

    # Convertir les listes en tenseurs
    src = [torch.tensor(s) for s in src]
    trg = [torch.tensor(t) for t in trg]

    # Padding des séquences
    src = pad_sequence(src, padding_value=Vocabulary.BLANK)
    trg = pad_sequence(trg, padding_value=Vocabulary.BLANK)

    # Convertir content en tenseur
    content_tensor = torch.tensor(content, dtype=torch.float) 

    return (src, trg, content_tensor)

#pour checker
"""
words = Vocabulary()
path="../data/dataset.csv"

dataset=CustomDataset(path,words)

test_loader = DataLoader(dataset, collate_fn=collate_fn,batch_size=35)

for src,trg,score in test_loader :
    print(src.shape,trg.shape)
    print(src[:,0])
    print(trg[:,0])
    break
    
"""