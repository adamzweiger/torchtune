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
    including the last occurrence of "A:\n" in the raw text. We do not rely on
    sublist matching for "A:\n" but rather do a prefix-based approach.
    """

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        # Use the parent to get tokens and labels
        result = super()._prepare_sample(sample)
        tokens = result["tokens"]
        labels = result["labels"]  # same as tokens from parent

        # The entire raw text from which tokens were generated
        prompt = sample[self._column]

        # 1) Locate the last occurrence of "A:" in the raw text
        last_a_pos = prompt.rfind("A:")
        if last_a_pos == -1:
            # If no "A:" found, mask everything
            labels = [CROSS_ENTROPY_IGNORE_IDX] * len(labels)
        else:
            # 2) Truncate the text up to that occurrence
            #    (including the length of "A:")
            prefix_text = prompt[: last_a_pos + len("A:")]

            # 3) Tokenize just the truncated prefix
            #    (we assume our parent class has a self._tokenizer)
            prefix_tokens = self._tokenizer.encode(prefix_text, add_bos=True, add_eos=self.add_eos)
            
            # 4) Everything in prefix_tokens is masked, i.e. ignore its loss
            mask_end = len(prefix_tokens)
            if mask_end > len(labels):
                # edge case: if something caused prefix_tokens to be longer than the
                # final tokens. Typically won't happen, but just to be safe:
                mask_end = len(labels)
            for i in range(mask_end):
                labels[i] = CROSS_ENTROPY_IGNORE_IDX

        result["labels"] = labels
        return result


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
