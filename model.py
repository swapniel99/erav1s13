import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from yolov3 import YOLOv3
from dataset import YOLODataset
from loss import YoloLoss
from torch import optim
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader

import config
from utils import ResizeDataLoader


class Model(LightningModule):
    def __init__(self, in_channels=3, batch_size=config.BATCH_SIZE, learning_rate=config.LEARNING_RATE,
                 num_epochs=config.NUM_EPOCHS, enable_gc='batch', dws=False, lambda_noobj=5, lambda_box=10,
                 print_step=False, print_batch=False, device_count=config.DEVICE_COUNT):
        super(Model, self).__init__()
        self.network = YOLOv3(in_channels, config.NUM_CLASSES, dws=dws)
        self.criterion = YoloLoss(config.SCALED_ANCHORS, lambda_noobj=lambda_noobj, lambda_box=lambda_box)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.enable_gc = enable_gc
        self.num_epochs = num_epochs
        self.print_step = print_step
        self.print_batch = print_batch
        self.device_count = device_count
        self.my_train_loss = MeanMetric()
        self.my_val_loss = MeanMetric()
        self.save_hyperparameters()

    def forward(self, x):
        return self.network(x)

    def common_step(self, batch, metric):
        x, y = batch
        out = self.forward(x)
        loss = self.criterion(out, y)
        metric.update(loss, x.shape[0])
        del x, y, out
        return loss

    def get_current_lrs(self):
        return ', '.join(str(param_group['lr']) for param_group in self.optimizers().optimizer.param_groups)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, self.my_train_loss)
        if self.print_step:
            print(f"Global Step: {self.global_step}, Current LRs: {self.get_current_lrs()}, Loss: {loss}")
        self.log(f"train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, self.my_val_loss)
        self.log(f"val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, (tuple, list)):
            x, _ = batch
        else:
            x = batch
        return self.forward(x)

    def configure_optimizers(self):
        # Effective LR and batch size are different in DDP
        effective_lr = self.learning_rate * self.device_count

        optimizer = optim.SGD(self.parameters(), lr=effective_lr/100, momentum=0.9)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=effective_lr,
            steps_per_epoch=(len(self.train_dataloader()) + 1) // self.device_count,
            epochs=self.num_epochs,
            pct_start=0.2,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy='linear'
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    def train_dataloader(self):
        train_dataset = YOLODataset(
            config.DATASET + '/train.csv',
            transform=config.train_transforms,
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
            mosaic=0.5
        )

        train_loader = ResizeDataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=True,
            resolutions=config.MULTIRES,
            cum_weights=config.CUM_PROBS
        )

        return train_loader

    def val_dataloader(self):
        train_eval_dataset = YOLODataset(
            config.DATASET + '/test.csv',
            transform=config.test_transforms,
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
            mosaic=0
        )

        train_eval_loader = DataLoader(
            dataset=train_eval_dataset,
            batch_size=self.batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False
        )

        return train_eval_loader

    def predict_dataloader(self):
        return self.val_dataloader()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.enable_gc == 'batch':
            garbage_collection_cuda()
        if self.print_batch:
            print(f"Global Step: {self.global_step}, Current LRs: {self.get_current_lrs()}")

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if self.enable_gc == 'batch':
            garbage_collection_cuda()

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if self.enable_gc == 'batch':
            garbage_collection_cuda()

    def on_train_epoch_end(self):
        if self.enable_gc == 'epoch':
            garbage_collection_cuda()
        print(f"Epoch: {self.current_epoch}, Global Steps: {self.global_step}, "
              f"Train Loss: {self.my_train_loss.compute()}, LR: {self.get_current_lrs()}")
        self.my_train_loss.reset()

    def on_validation_epoch_end(self):
        if self.enable_gc == 'epoch':
            garbage_collection_cuda()
        print(f"Epoch: {self.current_epoch}, Global Steps: {self.global_step}, "
              f"Val Loss: {self.my_val_loss.compute()}, LR: {self.get_current_lrs()}")
        self.my_val_loss.reset()

    def on_predict_epoch_end(self):
        if self.enable_gc == 'epoch':
            garbage_collection_cuda()


def main():
    num_classes = 20
    IMAGE_SIZE = 416
    INPUT_SIZE = IMAGE_SIZE  # * 2
    model = Model()
    from torchinfo import summary
    print(summary(model, input_size=(2, 3, INPUT_SIZE, INPUT_SIZE)))
    inp = torch.randn((2, 3, INPUT_SIZE, INPUT_SIZE))
    out = model(inp)
    assert out[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert out[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert out[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")


if __name__ == "__main__":
    main()
