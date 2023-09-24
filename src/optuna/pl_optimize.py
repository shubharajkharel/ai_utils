import pytorch_lightning as pl
from fvcore.nn import FlopCountAnalysis

from ..lightning.pl_data_module import PlDataModule
from ..lightning.pl_module import PLModule
from ..optuna.optimize import optimize, trial_instance_generator
from ..torch.sample_torch_dataset import SampleTorchDataset
from ..torch.simple_torch_models import SimpleTorchFCModel


# TODO: refactor
def pl_objective(trial, trial_objs_generator):
    objs = trial_objs_generator(trial)
    objs["trainer"].fit(objs["model"], objs["data_module"])
    val_loss = objs["trainer"].callback_metrics["val_loss"].item()
    # ==================== #
    # FLOPS
    # sample_input = (
    #     objs["data_module"]
    #     .train_dataloader()
    #     .dataset[0][0]
    #     .unsqueeze(0)
    #     .to(objs["model"].device)
    # )
    # flops = FlopCountAnalysis(objs["model"].model, (sample_input,)).total()
    # ==================== #
    return val_loss


if __name__ == "__main__":
    model_params_generator = trial_instance_generator(
        {"hidden_size": lambda t: t.suggest_int("hidden_size", 10, 100)}
    )

    trainer_params_generator = trial_instance_generator(
        {
            # TODO: fix why this do not work for multi-objective optimization.
            # prolly coz monitor needs two values or not..
            # "callbacks": lambda t: [
            #     PyTorchLightningPruningCallback(t, monitor="val_loss")
            # ],
        }
    )

    trial_objs_generator = trial_instance_generator(
        {
            "model": lambda t: PLModule(
                SimpleTorchFCModel(**model_params_generator(t))
            ),
            "trainer": lambda t: pl.Trainer(
                max_epochs=10, **trainer_params_generator(t)
            ),
            "data_module": lambda t: PlDataModule(
                SampleTorchDataset(SimpleTorchFCModel()), random_seed=42
            ),
        }
    )

    optimize(
        pl_objective(trial_objs_generator=trial_objs_generator),
        study_name="example-study",
        n_trials=10,
        load_if_exists=False,
        save_study=True,
        save_sampler=True,
        direction=["minimize", "minimize"],
    )
