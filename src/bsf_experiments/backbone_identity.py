"""Immutable identity shared by DINOv3 runtime and release workflows.

Hugging Face supports full commit hashes through the ``revision`` parameter.
Using the release commit here keeps new activations compatible with published
BSF checkpoints even if the model repository's default branch later changes:
https://huggingface.co/docs/transformers/main_classes/model#from_pretrained.
"""

from __future__ import annotations


DINO_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"


__all__ = ["DINO_MODEL_ID", "DINO_REVISION"]
