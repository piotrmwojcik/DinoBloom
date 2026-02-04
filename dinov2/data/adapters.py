# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

from torch.utils.data import Dataset


class DatasetWithEnumeratedTargets(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def get_image_data(self, index: int) -> bytes:
        return self._dataset.get_image_data(index)

    def get_target(self, index: int) -> Tuple[Any, int]:
        target = self._dataset.get_target(index)
        return (index, target)

    def __getitem__(self, index: int) -> Tuple[Any, Tuple[Any, int]]:
        out = self._dataset[index]

        # Underlying dataset might return:
        #   (image, target) or (image, target, filepath) or more
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            image, target = out[0], out[1]
        else:
            raise ValueError(
                f"Underlying dataset must return at least (image, target). Got: {type(out)} {out}"
            )

        # DINOv2 convention: return (image, (index, target))
        # If target is None, fall back to index (keeps old behavior)
        target = index if target is None else target
        return image, (index, target)

    def __len__(self) -> int:
        return len(self._dataset)
