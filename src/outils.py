import optuna
import pytorch_lightning as pl
import numpy as np
import json

def config(model,grid):
    """
        function pour complementer le model
    """
    model.complet(grid)

def optimize_hyperparameters(trial, hyperparameters):
    """
    Fonction pour optimiser les hyperparamètres en utilisant Optuna.

    Args:
        trial (optuna.trial.Trial): Objet d'Optuna utilisé pour suggérer des valeurs d'hyperparamètres.
        hyperparameters (dict): Dictionnaire des spécifications d'hyperparamètres à optimiser.

    Returns:
        dict: Dictionnaire mis à jour avec les valeurs d'hyperparamètres optimisées.
    """
    updated_hyperparameters = {}
    
    for key, value in hyperparameters.items():
        if isinstance(value, list):
            # Si la valeur est une liste, cela signifie que nous utilisons trial.suggest_int pour obtenir une valeur
            updated_value = trial.suggest_categorical(key, value)  
        else:
            # Sinon, nous conservons la valeur telle quelle
            updated_value = value
        
        updated_hyperparameters[key] = updated_value

    return updated_hyperparameters

#train sur un seul train_set et val_set
def single_fold_objective(model, data_module):
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=10)

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Evaluate the model on the validation set
    val_loss = trainer.callback_metrics['val_loss']
    return val_loss

def objective_cv(trial,model, grid ,data_module):

    k_folds=data_module.get_nbr_paquet()

    # Define hyperparameters
    grid = optimize_hyperparameters(trial,grid)

    # complet the model
    config(model,grid)

    scores = []
    for fold_idx in range(k_folds):
        # Initialize data module for the current fold
        data_module.set_idx_eval(fold_idx)
        data_module.setup()

        # Evaluate the model on the current fold
        fold_score = single_fold_objective(model, data_module)
        scores.append(fold_score)

    # Return the average score across all folds
    return np.mean(scores)


def calcul_hyp_com(grid):
    res=1
    for v in grid.values():
        if isinstance(v, list):
            res*=len(v)
    
    return res


def load_best_model_params(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        score = data['score']
        params = data['params']
        return score, params


def save_best_model_params(score, params, filename="test"):
    with open(filename+'.json', 'w') as file:
        json.dump({'score': score, 'params': params}, file, indent=4)