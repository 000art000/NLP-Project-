from model.enc_dec_cat import encodeur_decodeur_cat
from loss import RMSELoss
from Data_loader.transform_data import MyDataModule
import functools
from outils import objective_cv,calcul_hyp_com,save_best_model_params
import optuna
import json


path_file="../data/dataset.csv"
batch_size= 32

grid={
    'learning_rate' : [1e-3],
    'embedding_dim' :  [64],
    'num_layers_enc' : [8],
    'num_heads' : [8],
    'num_layers_dec' : [2,4],
    'dropout' : [0.2],
    'loss' : RMSELoss(),
    'lambda_l' : [0.001]
}



nbr_paquet=5

data=MyDataModule(path_file, batch_size)
data.set_paquet(nbr_paquet)

vocab_size=data.get_size_voc()
model=encodeur_decodeur_cat(vocab_size)

objective_partial = functools.partial(objective_cv,model=model, grid=grid ,data_module=data)
# Create an Optuna study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective_partial, n_trials=calcul_hyp_com(grid))

# Après la fin de l'étude Optuna
best_trial = study.best_trial
best_score = best_trial.value
best_params =  best_trial.params

filename=f'best_model/model_{1}'
save_best_model_params(best_score, best_params,filename)

# Analyze the results
print("Best trial:", best_params)
print("Best loss:", best_score)


"""
from model.model_LLM import encodeur_decodeur_cat
from loss import RMSELoss
from Data_loader.LLM_loder import LLM_Loader
import functools
from outils import objective_cv,calcul_hyp_com
import optuna
from transformers import LongformerTokenizer, LongformerModel

path_file="../data/dataset.csv"
batch_size=2

grid={
    'learning_rate' : [1e-3],
    'num_layers_dec' : [2,4,6],
    'loss' : RMSELoss(),
    'lambda_l' :[1e-2,1e-3]
}


nbr_paquet=5

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
LLM_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

def fun_cls(outputs1,outputs2):
    cls_output1 = outputs1.last_hidden_state[:, 0, :]
    cls_output2 = outputs2.last_hidden_state[:, 0, :]

    return cls_output1, cls_output2

model=encodeur_decodeur_cat(LLM_model,fun_cls)

data = LLM_Loader(path_file, tokenizer, batch_size=batch_size)
data.set_paquet(nbr_paquet)

objective_partial = functools.partial(objective_cv,model=model, grid=grid ,data_module=data)
# Create an Optuna study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective_partial, n_trials=calcul_hyp_com(grid))

# Analyze the results
print("Best trial:", study.best_trial.params)


"""