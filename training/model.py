"""Модели для регрессии блендшейпов рта.

Вход  : 1×H×W, оттенки серого в [0,1].
Выход : K значений в [0,1] (сигмоида).

Две архитектуры:
  tiny — маленькая CNN с нуля. Быстрая, но на малых реальных датасетах слабая.
  mnv3 — MobileNetV3-small (torchvision). С --pretrained берёт веса ImageNet: при
         сотнях размеченных кадров предобученный бэкбон сильно выигрывает у сети с
         нуля (так же сделано в трекинге языка Rulez777).

Обе заканчиваются глобальным average-pool, поэтому не зависят от точного H×W.
"""
import torch
import torch.nn as nn


def conv_block(cin, cout, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class MouthNet(nn.Module):
    """tiny: компактная CNN с нуля."""
    arch = "tiny"

    def __init__(self, num_outputs: int, width: int = 16):
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c1, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            conv_block(c1, c2, 2),
            conv_block(c2, c3, 2),
            conv_block(c3, c4, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.1), nn.Linear(c4, num_outputs))

    def forward(self, x):
        return torch.sigmoid(self.head(self.pool(self.encoder(x))))


class MouthNetMNV3(nn.Module):
    """mnv3: MobileNetV3-small + регрессионная голова."""
    arch = "mnv3"

    def __init__(self, num_outputs: int, pretrained: bool = False):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_small(weights=weights).features      # → 576 каналов
        if not pretrained:
            # В torchvision у MobileNetV3 momentum BN = 0.01: за сотни шагов скользящая
            # статистика не успевает сойтись, и в eval сеть выдаёт константу (проверено:
            # train MSE 0.006, а выход в eval не зависит от кадра). При обучении с нуля —
            # обычный 0.1. С --pretrained BN заморожен на статистике ImageNet.
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = 0.1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, num_outputs),
        )
        # Нормализация ImageNet: серый кадр повторяется в 3 канала.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        return torch.sigmoid(self.head(self.pool(self.encoder(x))))


def freeze_encoder_bn(net: nn.Module):
    """Держать статистику BatchNorm энкодера замороженной (для предобученного бэкбона:
    на маленьком батче своих кадров она бы разъехалась). Вызывать после net.train()."""
    for m in net.encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def build_model(arch: str, num_outputs: int, width: int = 16, pretrained: bool = False) -> nn.Module:
    if arch == "tiny":
        return MouthNet(num_outputs, width=width)
    if arch == "mnv3":
        return MouthNetMNV3(num_outputs, pretrained=pretrained)
    raise ValueError(f"неизвестная архитектура: {arch}")


def model_from_checkpoint(ck: dict) -> nn.Module:
    """Восстановить модель из чекпойнта train.py (без скачивания предобученных весов)."""
    net = build_model(ck.get("arch", "tiny"), len(ck["channels"]), width=ck.get("width_mult", 16))
    net.load_state_dict(ck["state"])
    return net.eval()
