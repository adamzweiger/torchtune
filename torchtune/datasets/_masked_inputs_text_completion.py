# _masked_inputs_text_completion.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any, Dict, List, Mapping, Optional, Callable

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._utils import truncate
from torchtune.datasets._text_completion import TextCompletionDataset
from torchtune.modules.tokenizers import ModelTokenizer

class MaskedInputsTextCompletionDataset(TextCompletionDataset):
    """
    A subclass of TextCompletionDataset that masks all question blocks
    ("Q: ...\\nA:") and keeps the rest (the answer portion) for training.

    We assume each sample is structured like:
        {prefix}Q: <question>\\nA: <answer>\\n\\nQ: <question>...
    
    For each block of text from "Q: ...\\nA:" up to the next "Q:" (or end),
    we apply the cross-entropy loss only to the tokens after the "A:" marker.
    Hence, tokens in "Q: <question>\\nA:" are assigned `CROSS_ENTROPY_IGNORE_IDX`,
    while tokens in " <answer>\\n\\n" are normal trainable labels.

    Additionally, if there is any prefix text before the first "Q: ...\\nA:"
    block, we also mask that prefix.
    """

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        prompt = sample[self._column]

        # This pattern finds each block:
        #  Group(1) => "Q: ...\nA:"
        #  Group(2) => everything after "A:" up to the next \nQ: or the end
        pattern = re.compile(r"(Q:.*?\nA:)(.*?)(?=\nQ:|$)", re.DOTALL)
        matches = list(pattern.finditer(prompt))

        all_tokens: List[int] = []
        all_labels: List[int] = []

        # Track where the previous match ended, to handle "prefix" text
        prev_end = 0

        # If no match is found, we will mask the entire text at the end
        if not matches:
            tokens = self._tokenizer.encode(prompt, add_bos=True, add_eos=self.add_eos)
            labels = [CROSS_ENTROPY_IGNORE_IDX] * len(tokens)
            return {"tokens": tokens, "labels": labels}

        # Iterate over each Q:...A: block
        for i, match in enumerate(matches):
            # Mask any "prefix" before this match, if it exists
            if match.start() > prev_end:
                prefix_text = prompt[prev_end:match.start()]
                prefix_ids = self._tokenizer.encode(
                    prefix_text,
                    add_bos=(i == 0),  # BOS only for the very first chunk
                    add_eos=False,
                )
                prefix_labels = [CROSS_ENTROPY_IGNORE_IDX] * len(prefix_ids)
                all_tokens.extend(prefix_ids)
                all_labels.extend(prefix_labels)

            # Group 1 => "Q: <question>\nA:"
            q_a_portion = match.group(1)
            # Group 2 => The answer portion " <answer>\n\n..."
            answer_portion = match.group(2)

            # For the Q: ...\nA: portion, we mask
            # BOS only if this is the first chunk and no prefix was matched
            add_bos_here = (i == 0 and match.start() == 0)
            q_a_ids = self._tokenizer.encode(
                q_a_portion,
                add_bos=add_bos_here,
                add_eos=False,
            )
            q_a_labels = [CROSS_ENTROPY_IGNORE_IDX] * len(q_a_ids)
            all_tokens.extend(q_a_ids)
            all_labels.extend(q_a_labels)

            # Now the answer portion is trainable
            suffix_ids = self._tokenizer.encode(answer_portion, add_bos=False, add_eos=False)
            suffix_labels = suffix_ids[:]
            all_tokens.extend(suffix_ids)
            all_labels.extend(suffix_labels)

            prev_end = match.end()

        # If there's leftover text after the last match, mask it
        if prev_end < len(prompt):
            leftover_text = prompt[prev_end:]
            leftover_ids = self._tokenizer.encode(
                leftover_text,
                add_bos=False,
                add_eos=self.add_eos,  # optionally add EOS here
            )
            leftover_labels = [CROSS_ENTROPY_IGNORE_IDX] * len(leftover_ids)
            all_tokens.extend(leftover_ids)
            all_labels.extend(leftover_labels)
        else:
            # If we ended exactly at the end of the prompt and want to add EOS
            if self.add_eos:
                eos_id = self._tokenizer.eos_id
                all_tokens.append(eos_id)
                # Decide if you want the EOS predicted or masked
                all_labels.append(CROSS_ENTROPY_IGNORE_IDX)

        # Optional truncation
        if self._tokenizer.max_seq_len is not None and len(all_tokens) > self._tokenizer.max_seq_len:
            all_tokens = truncate(all_tokens, self._tokenizer.max_seq_len)
            all_labels = truncate(all_labels, self._tokenizer.max_seq_len)

        return {"tokens": all_tokens, "labels": all_labels}


def masked_inputs_text_completion_dataset(
    tokenizer: ModelTokenizer,
    source: str,
    column: str = "text",
    add_eos: bool = True,
    packed: bool = False,
    split_across_pack: bool = True,
    split: str = "train",
    filter_fn: Optional[Callable] = None,
    **load_dataset_kwargs: Any,
) -> MaskedInputsTextCompletionDataset:
    """
    Build a dataset that masks each question prompt in the sample
    (including any prefix text before "Q:") and trains on everything
    after the 'A:' marker. This is useful for scenarios where questions
    should be hidden (e.g. not generating the question) but the answer
    portion should be generated.

    Args:
        tokenizer (ModelTokenizer): A tokenizer implementing the usual encode method.
        source (str): Path to dataset repository on Hugging Face or local data format (json, csv, text, etc.).
        column (str): Name of the column containing the text. Default: "text".
        add_eos (bool): Whether to add an EOS token at the end. Default: True.
        packed (bool): Whether to pack sequences to `max_seq_len` using `PackedDataset`. Default: False.
        split_across_pack (bool): If True, splits last sample across to next pack if it doesn't fit fully.
            Otherwise it moves that entire sample to the next pack. Ignored if `packed=False`.
            Default: True.
        split (str): Split argument for `datasets.load_dataset`, e.g. "train" or "train[:10%]". Default: "train".
        filter_fn (Optional[Callable]): Callable for filtering the dataset prior to pre-processing.
        **load_dataset_kwargs (Any): Additional kwargs passed to `load_dataset`.

    Returns:
        MaskedInputsTextCompletionDataset or PackedDataset:
            A dataset masking question blocks plus any prefix,
            optionally wrapped in a `PackedDataset`.

    Raises:
        ValueError: If `packed=True` and `tokenizer.max_seq_len` is not set.
    """
    ds = MaskedInputsTextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        column=column,
        add_eos=add_eos,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs
    )

    if packed:
        from torchtune.datasets._packed import PackedDataset
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(
            ds,
            max_seq_len=tokenizer.max_seq_len,
            split_across_pack=split_across_pack
        )

    return ds
