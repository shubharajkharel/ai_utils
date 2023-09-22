import logging
import os
from typing import Union, Sequence
import pickle

import optuna


def trial_instance_generator(hparams_dict: dict):
    def hparams_generator(trial):
        hparams = {}
        for param_name, param_func in hparams_dict.items():
            hparams[param_name] = param_func(trial)
        return hparams

    return hparams_generator


def optimize(
    objective,
    study_name: str = "example-study",
    n_trials: int = 20,
    direction: Union[
        str, Sequence[Union[str, optuna.study.StudyDirection]]
    ] = "minimize",  # "maximize", ["minimize", "maximize", ...]
    load_if_exists=True,  # load study and sampler if exists
    sampler: Union[optuna.samplers.BaseSampler, None] = None,
    save_study: bool = True,
    save_sampler: bool = True,
    timeout: Union[float, None] = None,
    n_jobs: int = 10,
    
):
    # file names for saving and loading
    if load_if_exists or save_study or save_sampler:
        study_db_name = study_name + ".db"
        study_db_path = "sqlite:///" + study_db_name
        sampler_name = sampler.__class__.__name__
        sampler_path = study_name + "_" + sampler_name + ".pkl"

    if load_if_exists:
        if os.path.exists(sampler_path):
            with open(sampler_path, "rb") as fin:
                sampler = pickle.load(fin)
        else:
            logging.getLogger().warning("Sampler file not found. Creating new sampler.")
        if not os.path.exists(study_db_name):
            logging.getLogger().warning("Study file not found. Creating new study.")

    if save_study and os.path.exists(study_db_name):
        logging.getLogger().warning("Study file already exists. Overwriting.")
        os.remove(study_db_name)

    if save_sampler and os.path.exists(sampler_path):
        logging.getLogger().warning("Sampler file already exists. Overwriting.")
        os.remove(sampler_path)

    study = optuna.create_study(
        study_name=study_name,
        storage=study_db_path if save_study else None,
        direction=None if isinstance(direction, list) else direction,
        directions=direction if isinstance(direction, list) else None,
        load_if_exists=load_if_exists,
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

    if save_sampler:
        with open(sampler_path, "wb") as fout:
            pickle.dump(study.sampler, fout)
