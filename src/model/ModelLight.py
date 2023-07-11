import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch import Tensor
from pytorch_optimizer import AdaBelief
from torch.optim.lr_scheduler import CyclicLR
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_f1_score,
)


class MultilabelClassifier(nn.Module):
    def __init__(self, resnet_type: str, freeze: bool = False):
        super().__init__()
        if resnet_type == "resnet18":
            self.model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

        if resnet_type == "resnet50":
            self.model = torchvision.models.resnet50(weights="IMAGENET1K_V1")

        if resnet_type == "resnet101":
            self.model = torchvision.models.resnet101(weights="IMAGENET1K_V1")

        self.model.fc = nn.Identity()

        if freeze:
            self.freeze_model(self.model)

        self.example_input_array = torch.rand(1, 3, 720, 720)
        len_result = self.model(self.example_input_array).shape[1]

        self.list_name_classes = []

        self.sofa_clf = self.make_seq("sofa", len_result)
        self.wardrobe_clf = self.make_seq("wardrobe", len_result)
        self.chair_clf = self.make_seq("chair", len_result)
        self.armchair_clf = self.make_seq("armchair", len_result)
        self.table_clf = self.make_seq("table", len_result)
        self.commode_clf = self.make_seq("commode", len_result)
        self.bed_clf = self.make_seq("bed", len_result)

    def make_seq(self, name: str, len_result: int):
        self.list_name_classes.append(name)
        return nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=len_result, out_features=int(len_result / 2)),
            nn.Linear(in_features=int(len_result / 2), out_features=1),
        )

    def freeze_model(self, model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)

        return {
            "sofa": self.sofa_clf(x),
            "wardrobe": self.wardrobe_clf(x),
            "chair": self.chair_clf(x),
            "armchair": self.armchair_clf(x),
            "table": self.table_clf(x),
            "commode": self.commode_clf(x),
            "bed": self.bed_clf(x),
        }


class ModelLight(pl.LightningModule):
    def __init__(self, resnet_type: str, lr: float = 0.003):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.model = MultilabelClassifier(resnet_type)
        self.example_input_array = torch.rand(1, 3, 720, 720)

        self.test_metric = None

        self.make_val_test_list()

    def forward(self, x: Tensor):
        return self.model(x)

    def make_val_test_list(self):
        self.predict_test = {}
        self.target_test = {}
        self.predict_proba_test = {}
        self.predict_val_label = {}
        self.target_val_label = {}

        for name in self.model.list_name_classes:
            self.predict_val_label[name] = []
            self.target_val_label[name] = []

            self.predict_test[name] = []
            self.target_test[name] = []
            self.predict_proba_test[name] = []

    def configure_optimizers(self):
        optimizer = AdaBelief(self.model.parameters(), lr=self.lr)

        sch = CyclicLR(
            optimizer,
            base_lr=self.lr / 10,
            max_lr=self.lr * 10,
            step_size_up=4000,
            mode="exp_range",
            cycle_momentum=False,
            gamma=0.8,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sch},
        }

    def criterion(self, pred: Tensor, y: Tensor):
        loss = 0
        for key in pred.keys():
            loss += F.binary_cross_entropy_with_logits(pred[key].squeeze_(), y[key])
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch["img_x"]
        y = batch["labels"]

        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        metrics = self._shared_eval_epoch()

        self.log_dict(metrics, on_epoch=True)

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    def on_test_end(self):
        mectrics = self._shared_eval_epoch(save_predict=True)
        self.test_metric = mectrics

    def _shared_eval_step(self, batch, batch_idx):
        with torch.no_grad():
            x = batch["img_x"]
            y = batch["labels"]
            pred = self(x)

            for key in pred.keys():
                self.predict_val_label[key].append(
                    F.sigmoid(pred[key]).squeeze_().cpu()
                )
                self.target_val_label[key].append(y[key].cpu())

    def _shared_eval_epoch(self, save_predict=False):
        metric_dict = {}
        metric_dict["val_loss"] = 0

        with torch.no_grad():
            global_predict = []
            global_target = []
            for key in self.predict_val_label.keys():
                predict = torch.cat(self.predict_val_label[key])
                target = torch.cat(self.target_val_label[key])

                target = target.to(dtype=torch.float)

                if save_predict:
                    self.predict_proba_test[key].extend(predict.tolist())

                loss = F.binary_cross_entropy(predict, target)

                predict = predict.round_().long()
                target = target.long()

                if save_predict:
                    self.predict_test[key].extend(predict.tolist())
                    self.target_test[key].extend(target.tolist())

                accuracy = binary_accuracy(predict, target)
                f1_score = binary_f1_score(predict, target)

                metric_dict["val_loss"] += loss.float()
                metric_dict[f"{key}_loss"] = loss.float()
                metric_dict[f"{key}_accuracy"] = accuracy.float()
                metric_dict[f"{key}_f1"] = f1_score.float()

                global_predict.append(predict)
                global_target.append(target)

                self.predict_val_label[key] = []
                self.target_val_label[key] = []

        predict = torch.cat(global_predict)
        target = torch.cat(global_target)
        global_accuracy = binary_accuracy(predict, target).float()
        global_f1 = binary_f1_score(predict, target).float()
        metric_dict["global_accuracy"] = global_accuracy
        metric_dict["global_f1"] = global_f1

        if save_predict:
            self.test_metrics = metric_dict.copy()

        return metric_dict
