import torch
from pytorch_lightning import LightningModule
from yolov3 import YOLOv3


class Model(LightningModule):
    def __init__(self, in_channels=3, num_classes=20):
        super(Model, self).__init__()
        self.network = YOLOv3(in_channels, num_classes)

    def forward(self, x):
        return self.network(x)


def main():
    num_classes = 20
    IMAGE_SIZE = 416
    INPUT_SIZE = IMAGE_SIZE * 2
    model = Model(num_classes=num_classes)
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
