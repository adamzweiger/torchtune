# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets._text_completion import TextCompletionDataset
from typing import Any, Dict, List, Mapping


class MaskedTextCompletionDataset(TextCompletionDataset):
    """
    A subclass of TextCompletionDataset that masks all tokens before and
    including the last occurrence of "A:" in the raw text. We do not rely on
    sublist matching for "A:" but rather do a prefix-based approach.
    """

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        prompt = sample[self._column]

        # find last "A:"
        idx = prompt.rfind("A:")
        if idx == -1:
            # fallback: if no "A:" found, mask everything
            tokens = self._tokenizer.encode(prompt, add_bos=True, add_eos=True)
            labels = [CROSS_ENTROPY_IGNORE_IDX] * len(tokens)
            return {"tokens": tokens, "labels": labels}

        # otherwise, split raw text
        prefix = prompt[: idx + len("A:")]  # e.g. "Q: Who won?\nA:"
        answer = prompt[idx + len("A:") :]  # e.g. " The Lakers"

        # tokenize separately
        prefix_ids = self._tokenizer.encode(prefix, add_bos=True, add_eos=False)
        answer_ids = self._tokenizer.encode(answer, add_bos=False, add_eos=False)

        # create labels
        prefix_labels = [CROSS_ENTROPY_IGNORE_IDX] * len(prefix_ids)
        answer_labels = answer_ids[:]  # copy them, we want to predict this portion

        tokens = prefix_ids + answer_ids
        labels = prefix_labels + answer_labels
        return {"tokens": tokens, "labels": labels}


def masked_text_completion_dataset(
    tokenizer,
    source: str,
    column: str = "text",
    add_eos: bool = True,
    packed: bool = False,
    split_across_pack: bool = True,
    split: str = "train",
    filter_fn=None,
    **load_dataset_kwargs
):
    ds = MaskedTextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        column=column,
        add_eos=add_eos,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs
    )

    if packed:
        # If you need packing, you can wrap ds in PackedDataset here
        from torchtune.datasets._packed import PackedDataset
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(
            ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=split_across_pack
        )

    return ds
