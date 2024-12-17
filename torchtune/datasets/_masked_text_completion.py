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
    A subclass of TextCompletionDataset that masks all tokens before and including the last "A:" marker
    so that the model only computes loss on the portion after the last "A:".
    """

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        # Use the parent to get tokens and labels
        result = super()._prepare_sample(sample)
        tokens = result["tokens"]
        labels = result["labels"]  # same as tokens from parent

        # Original prompt text used to find "A:" substring
        prompt = sample[self._column]

        # Find the last occurrence of "A:"
        last_a_pos = prompt.rfind("A:")
        if last_a_pos == -1:
            # If no "A:" found, mask everything
            labels = [CROSS_ENTROPY_IGNORE_IDX] * len(labels)
        else:
            # Tokenize "A:" separately to find its position in tokens
            a_tokens = self._tokenizer.encode("A:", add_bos=False, add_eos=False)

            # Find the last occurrence of a_tokens in tokens
            def find_last_sublist(main_list, sub_list):
                last_found = -1
                for i in range(len(main_list)-len(sub_list)+1):
                    if main_list[i:i+len(sub_list)] == sub_list:
                        last_found = i
                return last_found

            start_idx = find_last_sublist(tokens, a_tokens)
            if start_idx == -1:
                # Can't find "A:" in tokens; fallback - mask all
                labels = [CROSS_ENTROPY_IGNORE_IDX] * len(labels)
            else:
                # Mask everything up to and including the "A:" tokens
                mask_end = start_idx + len(a_tokens)
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
