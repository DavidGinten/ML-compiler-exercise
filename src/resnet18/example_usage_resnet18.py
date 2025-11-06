# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys
from pathlib import Path

import torch
import torch.utils._pytree as pytree
import torchvision.models as models
from torch_mlir import fx
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
from torch_mlir_e2e_test.configs.utils import (
    recursively_convert_to_numpy,
)

sys.path.append(str(Path(__file__).absolute().parent))
from PIL import Image
import requests

import torch
from torchvision import transforms


DEFAULT_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"
)
DEFAULT_LABEL_URL = (
    "https://raw.githubusercontent.com/cathyzhyi/ml-data/main/imagenet-classes.txt"
)


def load_and_preprocess_image(url: str = DEFAULT_IMAGE_URL):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
    }
    img = Image.open(requests.get(url, headers=headers, stream=True).raw).convert("RGB")
    # preprocessing pipeline
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)


def load_labels(url: str = DEFAULT_LABEL_URL):
    classes_text = requests.get(
        url=url,
        stream=True,
    ).text
    labels = [line.strip() for line in classes_text.splitlines()]
    return labels


def top3_possibilities(res, labels):
    _, indexes = torch.sort(res, descending=True)
    percentage = torch.nn.functional.softmax(res, dim=1)[0] * 100
    top3 = [(labels[idx], percentage[idx].item()) for idx in indexes[0][:3]]
    return top3


print("load image from " + DEFAULT_IMAGE_URL, file=sys.stderr)
img = load_and_preprocess_image(DEFAULT_IMAGE_URL)
labels = load_labels()

resnet18 = models.resnet18(pretrained=True).eval()
module = fx.export_and_import(
    resnet18,
    torch.ones(1, 3, 224, 224),
    output_type="linalg-on-tensors",
    func_name=resnet18.__class__.__name__,
)
backend = refbackend.RefBackendLinalgOnTensorsBackend()
compiled = backend.compile(module)
fx_module = backend.load(compiled)

params = {
    **dict(resnet18.named_buffers(remove_duplicate=False)),
}
params_flat, params_spec = pytree.tree_flatten(params)
params_flat = list(params_flat)
with torch.no_grad():
    numpy_inputs = recursively_convert_to_numpy(params_flat + [img])

golden_prediction = top3_possibilities(resnet18.forward(img), labels)
print("PyTorch prediction")
print(golden_prediction)
print(*numpy_inputs)


prediction = top3_possibilities(
    torch.from_numpy(getattr(fx_module, resnet18.__class__.__name__)(*numpy_inputs)),
    labels,
)
print("torch-mlir prediction")
with open("resnet18_input.mlir", "w") as f:
	f.write(str(numpy_inputs))
