"""
DINOv2 ViT-B/14 feature extractor via timm.

Archivo nuevo — no modifica ningún archivo existente.
Pesos descargados automáticamente de HuggingFace la primera vez.

Referencia: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

_EMBED_DIM = 768


class DINOv2Features(nn.Module):
    """
    DINOv2 ViT-B/14 wrapper compatible con la interfaz de data4robotics.

    BaseAgent requiere tres cosas del encoder de imagen:
      - embed_dim  : dimensión del token de salida (768 para ViT-B)
      - n_tokens   : número de tokens devueltos por imagen (1 — sólo CLS)
      - forward(x) : devuelve (B, n_tokens, embed_dim)

    Acepta imágenes de cualquier resolución y las redimensiona internamente
    a img_size (múltiplo de 14). La normalización ImageNet del transform
    pipeline es compatible con DINOv2.
    """

    def __init__(self, img_size: int = 252, freeze: bool = False):
        super().__init__()
        assert img_size % 14 == 0, f"img_size debe ser múltiplo de 14, recibido {img_size}"
        self.img_size = img_size

        # timm descarga los pesos DINOv2 de HuggingFace automáticamente
        self.model = timm.create_model(
            "vit_base_patch14_dinov2",
            pretrained=True,
            num_classes=0,       # elimina cabeza de clasificación → devuelve CLS
            img_size=img_size,
        )

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    @property
    def embed_dim(self) -> int:
        return _EMBED_DIM

    @property
    def n_tokens(self) -> int:
        return 1  # sólo el token CLS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — puede venir en cualquier resolución
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(
                x, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )
        cls = self.model(x)          # (B, 768)
        return cls[:, None, :]       # (B, 1, 768) — formato BaseAgent


def load_dinov2(img_size: int = 252, freeze: bool = False, restore_path: str = "") -> DINOv2Features:
    """
    Factory function compatible con la convención _target_ de Hydra.

    restore_path: ignorado (pesos vienen de HuggingFace via timm).
                  Se mantiene por coherencia con los demás feature configs.
    """
    model = DINOv2Features(img_size=img_size, freeze=freeze)
    if restore_path:
        print(f"[DINOv2] AVISO: restore_path='{restore_path}' ignorado "
              f"(los pesos DINOv2 se cargan desde HuggingFace via timm)")
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[DINOv2] ViT-B/14  img_size={img_size}  params={n:.0f}M  "
          f"embed_dim={model.embed_dim}  n_tokens={model.n_tokens}  freeze={freeze}")
    return model
