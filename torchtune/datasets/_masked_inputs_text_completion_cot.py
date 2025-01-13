# _masked_inputs_text_completion_cot.py
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

class MaskedInputsTextCompletionCoTDataset(TextCompletionDataset):
    """
    A subclass of TextCompletionDataset that masks all question blocks
    ("Q: ...\\nR:") and keeps the rest (reasoning, answer, etc.) for training.

    We assume each sample is structured like:
        {prefix}Q: <question>\\nR: <reasoning>\\nA: <answer>\\n\\nQ: <question>...
    
    For each block of text from "Q: ...\\nR:" up to the next "Q:" (or end),
    we apply the cross-entropy loss only to the tokens after the "R:" marker.
    Hence, tokens in "Q: <question>\\nR:" are assigned `CROSS_ENTROPY_IGNORE_IDX`,
    while tokens in "<reasoning>\\nA: <answer>\\n\\n" are normal trainable labels.
    
    Additionally, any prefix text before the first "Q: ...\\nR:" is masked,
    as is any leftover text after the last match.
    """

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        prompt = sample[self._column]
        
        # Regex explanation:
        # (Q:.*?\nR:) captures "Q:" up through "R:" (including "R:").
        # (.*?) lazily matches the remainder (the "reasoning/answer" portion),
        # until we hit either a newline followed by Q: OR the end of the string.
        pattern = re.compile(r"(Q:.*?\nR:)(.*?)(?=\nQ:|$)", re.DOTALL)
        matches = list(pattern.finditer(prompt))

        all_tokens: List[int] = []
        all_labels: List[int] = []

        # If no match is found, fallback: mask the entire text
        if not matches:
            tokens = self._tokenizer.encode(prompt, add_bos=True, add_eos=self.add_eos)
            labels = [CROSS_ENTROPY_IGNORE_IDX] * len(tokens)
            return {"tokens": tokens, "labels": labels}

        prev_end = 0

        for i, match in enumerate(matches):
            # 1) Mask any prefix before this match, if it exists
            if match.start() > prev_end:
                prefix_text = prompt[prev_end:match.start()]
                prefix_ids = self._tokenizer.encode(
                    prefix_text,
                    add_bos=(i == 0),  # Add BOS only if this is the very first chunk
                    add_eos=False
                )
                prefix_labels = [CROSS_ENTROPY_IGNORE_IDX] * len(prefix_ids)
                all_tokens.extend(prefix_ids)
                all_labels.extend(prefix_labels)

            # 2) Group 1: "Q: <question>\\nR:"
            q_r_portion = match.group(1)
            # 3) Group 2: "<reasoning>\\nA: <answer>\\n\\n" (up to next Q: or end)
            reasoning_ans_portion = match.group(2)

            # For the Q: ...\\nR: portion, we mask
            add_bos_here = (i == 0 and match.start() == 0)
            prefix_ids = self._tokenizer.encode(
                q_r_portion,
                add_bos=add_bos_here,
                add_eos=False
            )
            prefix_labels = [CROSS_ENTROPY_IGNORE_IDX] * len(prefix_ids)

            # The reasoning/answer portion is trainable
            suffix_ids = self._tokenizer.encode(
                reasoning_ans_portion, 
                add_bos=False, 
                add_eos=False
            )
            suffix_labels = suffix_ids[:]  # identical copy (trainable)

            # Combine
            all_tokens.extend(prefix_ids + suffix_ids)
            all_labels.extend(prefix_labels + suffix_labels)

            prev_end = match.end()

        # 4) If there's leftover text after the last match, mask it
        if prev_end < len(prompt):
            leftover_text = prompt[prev_end:]
            leftover_ids = self._tokenizer.encode(
                leftover_text,
                add_bos=False,
                add_eos=self.add_eos,  # Optionally add EOS here
            )
            leftover_labels = [CROSS_ENTROPY_IGNORE_IDX] * len(leftover_ids)
            all_tokens.extend(leftover_ids)
            all_labels.extend(leftover_labels)
        else:
            # If we ended exactly at the end of the prompt
            # and want to add an EOS:
            if self.add_eos:
                eos_id = self._tokenizer.eos_id
                all_tokens.append(eos_id)
                # Decide if you want EOS to be masked or predicted:
                all_labels.append(CROSS_ENTROPY_IGNORE_IDX)
                # all_labels.append(eos_id)  # if you want to train on the EOS

        # Optional truncation
        if (self._tokenizer.max_seq_len is not None) and (len(all_tokens) > self._tokenizer.max_seq_len):
            all_tokens = truncate(all_tokens, self._tokenizer.max_seq_len)
            all_labels = truncate(all_labels, self._tokenizer.max_seq_len)

        return {"tokens": all_tokens, "labels": all_labels}


def masked_inputs_text_completion_cot_dataset(
    tokenizer: ModelTokenizer,
    source: str,
    column: str = "text",
    add_eos: bool = True,
    packed: bool = False,
    split_across_pack: bool = True,
    split: str = "train",
    filter_fn: Optional[Callable] = None,
    **load_dataset_kwargs: Any,
) -> MaskedInputsTextCompletionCoTDataset:
    """
    Build a dataset that masks each question prompt in the sample (including
    any prefix text before "Q: ...\\nR:") and trains on everything after
    the 'R:' marker. This is useful for scenarios where questions should
    be hidden (e.g. not generating the question) but reasoning and answers
    should be generated.

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
        MaskedInputsTextCompletionCoTDataset or PackedDataset: 
            A dataset masking question blocks and any prefix, optionally wrapped in a `PackedDataset`.

    Raises:
        ValueError: If `packed=True` and `tokenizer.max_seq_len` is not set.
    """
    ds = MaskedInputsTextCompletionCoTDataset(
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
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=split_across_pack)

    return ds
