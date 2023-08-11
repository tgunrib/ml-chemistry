import optuna
import torch
from lightning_fabric import seed_everything
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional.regression import mean_absolute_error

from GCN_Files.Attentive_FP_files.pytorch_dataset import train_dataset, val_dataset, test_dataset
# This was used to tune attentive_fp model to measure if it improved the performance.

class AttentiveFPModel(pl.LightningModule):
    def __init__(self, lr, num_layers, hidden_dim, dropout, num_timesteps):
        super(AttentiveFPModel, self).__init__()
        self.lr = lr
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_timesteps = num_timesteps

        self.model = AttentiveFP(
            in_channels=39, hidden_channels=hidden_dim, out_channels=1,
            edge_dim=10, num_layers=num_layers, num_timesteps=num_timesteps,
            dropout=dropout
        )

        self.loss = torch.nn.L1Loss()

    def forward(self, data):
        return self.model(data.x, data.edge_index, data.edge_attr)

    def training_step(self, batch, batch_idx):
        y_pred = self(batch)
        loss = self.loss(y_pred.view(-1), batch.y.view(-1))
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred = self(batch)
        loss = self.loss(y_pred.view(-1), batch.y.view(-1))
        self.log("val_loss", loss, on_epoch=True)
        self.log("mae", mean_absolute_error(y_pred.view(-1), batch.y.view(-1)), on_epoch=True)

    def test_step(self, batch, batch_idx):
        y_pred = self(batch)
        self.log("test_mae", mean_absolute_error(y_pred.view(-1), batch.y.view(-1)), on_epoch=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


seed_everything(42)


def objective(trial: optuna.trial.Trial) -> float:
    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_layers = trial.suggest_int("num_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    lr = trial.suggest_loguniform('lr', 1e-1, 1e-5)
    num_timesteps = trial.suggest_int('num_timesteps',2,6)

    logger = TensorBoardLogger(save_dir="logs", name="attentivefp")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best_model",
        save_top_k=1,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=10, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=200,
        gpus=torch.cuda.device_count(),
    )

    model = AttentiveFPModel(lr=lr, num_layers=n_layers, hidden_dim=hidden_dim, dropout=dropout, num_timesteps=num_timesteps)

    hyperparameters = dict(num_layers=n_layers, dropout=dropout, lr=lr, hidden_dim=hidden_dim, num_timesteps=num_timesteps)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model)

    return trainer.callback_metrics["val_acc"].item()


pruner = optuna.pruners.MedianPruner
study = optuna.create_study(direction="minimize", pruner=pruner, study_name='Attentive_FP')
study.optimize(objective, n_trials=50)
tuning = study.trials_dataframe()
tuning.to_csv('Attentivr_FP with Tuning')


print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
