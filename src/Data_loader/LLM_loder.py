import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import functools
from pytorch_lightning import LightningDataModule

class CustomDataset(Dataset):
    def __init__(self, path_file):
        self.data = pd.read_csv(path_file)  # schema text, content, prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

def collate_fn(batch, LLM_tokenizer):

    # Extract prompt, rep, score from the batch
    prompts = [row['prompt'] for row in batch]
    reps = [row['text'] for row in batch]
    score_list = [row['content'] for row in batch]

    max_length= LLM_tokenizer.model_max_length #taille max de sequence que le modele peux supporté
    max_prompts = min(max(list(map(len, prompts))),max_length)
    max_reps = min(max(list(map(len, reps))),max_length)

    prompts_data = []
    prompt_masks = []
    reps_data = []
    reps_masks = []

    for i in range(len(prompts)):
        
        # Tokenize and pad prompts to the maximum length
        encoded = LLM_tokenizer(prompts[i], return_tensors="pt", max_length=max_prompts, truncation=True, padding='max_length')
        input_ids = encoded['input_ids']
        mask = encoded['attention_mask']
        prompts_data.append(input_ids)
        prompt_masks.append(mask)

        # Tokenize and pad answers to the maximum length
        encoded = LLM_tokenizer(reps[i], return_tensors='pt', padding='max_length', max_length=max_reps, truncation=True)
        input_ids = encoded['input_ids']
        mask = encoded['attention_mask']
        reps_data.append(input_ids)
        reps_masks.append(mask)

    prompts_data = torch.stack(prompts_data).squeeze(1) #shape B,T 
    prompt_masks = torch.stack(prompt_masks).squeeze(1) #shape B,T 
    reps_data = torch.stack(reps_data).squeeze(1) #shape B,T 
    reps_masks = torch.stack(reps_masks).squeeze(1) #shape B,T 
    score_list = torch.tensor(score_list) #shape B

    return prompts_data, prompt_masks, reps_data, reps_masks, score_list


class LLM_Loader(LightningDataModule):
    def __init__(self, path_file, LLM_tokenizer, batch_size=1, num_workers=0):
        super().__init__()

        #param pour CustomDataset
        self.path_file = path_file

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_paquet=None
        self.idx_eval=None
        self.LLM_tokenizer=LLM_tokenizer
        self.full_dataset=CustomDataset(self.path_file)

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
        partial_collate_fn = functools.partial(collate_fn, LLM_tokenizer=self.LLM_tokenizer)
        return DataLoader(self.train_dataset, batch_size=self.batch_size,collate_fn=partial_collate_fn ,shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        partial_collate_fn = functools.partial(collate_fn, LLM_tokenizer=self.LLM_tokenizer)
        return DataLoader(self.val_dataset, batch_size=self.batch_size,collate_fn=partial_collate_fn ,num_workers=self.num_workers)




"""

# Initialize the tokenizer for your model (replace 'bert-base-uncased' with your model)
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

path = "../data/dataset.csv"
dataset = CustomDataset(path, tokenizer)

partial_collate_fn = functools.partial(collate_fn, LLM_tokenizer=tokenizer)
test_loader = DataLoader(dataset, collate_fn=partial_collate_fn, batch_size=35)

for prompts_data, prompt_masks, reps_data, reps_masks, score_list in test_loader:
    print(prompts_data.size(), prompt_masks.size(), reps_data.size(), reps_masks.size(), score_list.size())

"""