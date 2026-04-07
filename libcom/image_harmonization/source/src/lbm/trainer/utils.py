import logging
import os
import re
import time
from typing import Dict, List, Literal, Optional, Tuple

import torch


class StateDictAdapter:
    """
    StateDictAdapter for adapting the state dict of a model to a checkpoint state dict.

    This class will iterate over all keys in the checkpoint state dict and filter them by a list of regex keys.
    For each matching key, the class will adapt the checkpoint state dict to the model state dict.
    Depending on the target size, the class will add missing blocks or cut the block.
    When adding missing blocks, the class will use a strategy to fill the missing blocks: either adding zeros or normal random values.

    Example:

    ```
    adapter = StateDictAdapter()
    new_state_dict = adapter(
        model_state_dict=model.state_dict(),
        checkpoint_state_dict=state_dict,
        regex_keys=[
            r"class_embedding.linear_1.weight",
            r"conv_in.weight",
            r"(down_blocks|up_blocks)\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
            r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight"
        ]
    )
    ```

    Args:
        model_state_dict (Dict[str, torch.Tensor]): The model state dict.
        checkpoint_state_dict (Dict[str, torch.Tensor]): The checkpoint state dict.
        regex_keys (Optional[List[str]]): A list of regex keys to adapt the checkpoint state dict. Defaults to None.
            Passing a list of regex will drastically reduce the latency.
            If None, all keys in the checkpoint state dict will be adapted.
        strategy (Literal["zeros", "normal"], optional): The strategy to fill the missing blocks. Defaults to "normal".

    """

    def _create_block(
        self,
        shape: List[int],
        strategy: Literal["zeros", "normal"],
        input: torch.Tensor = None,
    ):
        if strategy == "zeros":
            return torch.zeros(shape)
        elif strategy == "normal":
            if input is not None:
                mean = input.mean().item()
                std = input.std().item()
                return torch.randn(shape) * std + mean
            else:
                return torch.randn(shape)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

    def __call__(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        checkpoint_state_dict: Dict[str, torch.Tensor],
        regex_keys: Optional[List[str]] = None,
        strategy: Literal["zeros", "normal"] = "normal",
    ):
        start = time.perf_counter()
        # if no regex keys are provided, we use all keys in the model state dict
        if regex_keys is None:
            regex_keys = list(model_state_dict.keys())

        # iterate over all keys in the checkpoint state dict
        for checkpoint_key in list(checkpoint_state_dict.keys()):
            # iterate over all regex keys
            for regex_key in regex_keys:
                if re.match(regex_key, checkpoint_key):
                    dst_shape = model_state_dict[checkpoint_key].shape
                    src_shape = checkpoint_state_dict[checkpoint_key].shape

                    ## Sizes adapter
                    # if length of shapes are different, we need to unsqueeze or squeeze the tensor
                    if len(dst_shape) != len(src_shape):
                        # in the case [a] vs [a, b] -> unsqueeze [a, 1]
                        if len(src_shape) == 1:
                            checkpoint_state_dict[checkpoint_key] = (
                                checkpoint_state_dict[checkpoint_key].unsqueeze(1)
                            )
                            logging.info(
                                f"Unsqueeze {checkpoint_key}: {src_shape} -> {checkpoint_state_dict[checkpoint_key].shape}"
                            )
                        # in the case [a, b] vs [a] -> squeeze [a]
                        elif len(dst_shape) == 1:
                            checkpoint_state_dict[checkpoint_key] = (
                                checkpoint_state_dict[checkpoint_key][:, 0]
                            )
                            logging.info(
                                f"Squeeze {checkpoint_key}: {src_shape} -> {checkpoint_state_dict[checkpoint_key].shape}"
                            )
                        # in the other cases, raise an error
                        else:
                            raise ValueError(
                                f"Shapes of {checkpoint_key} are different: {dst_shape} != {src_shape}"
                            )

                        # update the shapes
                        dst_shape = model_state_dict[checkpoint_key].shape
                        src_shape = checkpoint_state_dict[checkpoint_key].shape
                        assert len(dst_shape) == len(
                            src_shape
                        ), f"Shapes of {checkpoint_key} are different: {dst_shape} != {src_shape}"

                    ## Shapes adapter
                    # modify the checkpoint state dict only if the shapes are different
                    if dst_shape != src_shape:
                        # create a copy of the tensor
                        tmp = torch.clone(checkpoint_state_dict[checkpoint_key])

                        # iterate over all dimensions
                        for i in range(len(dst_shape)):
                            if dst_shape[i] != src_shape[i]:
                                diff = dst_shape[i] - src_shape[i]

                                # if the difference is greater than 0, we need to add missing blocks
                                if diff > 0:
                                    missing_shape = list(tmp.shape)
                                    missing_shape[i] = diff
                                    missing = self._create_block(
                                        shape=missing_shape,
                                        strategy=strategy,
                                        input=tmp,
                                    )
                                    tmp = torch.cat((tmp, missing), dim=i)
                                    logging.info(
                                        f"Adapting {checkpoint_key} with strategy:{strategy} from shape {src_shape} to {dst_shape}"
                                    )
                                # if the difference is less than 0, we need to cut the block
                                else:
                                    tmp = tmp.narrow(i, 0, dst_shape[i])
                                    logging.info(
                                        f"Adapting {checkpoint_key} by narrowing from shape {src_shape} to {dst_shape}"
                                    )

                        checkpoint_state_dict[checkpoint_key] = tmp
        end = time.perf_counter()
        logging.info(f"StateDictAdapter took {end-start:.2f} seconds")
        return checkpoint_state_dict


class StateDictRenamer:
    """
    StateDictRenamer for renaming keys in a checkpoint state dict.
    This class will iterate over all keys in the checkpoint state dict and rename them according to a rename dict.

    Example:

        ```
        renamer = StateDictRenamer()
        new_state_dict = renamer(
            checkpoint_state_dict=state_dict,
            rename_dict={
                "add_embedding.linear_1.weight": "class_embedding.linear_1.weight",
                "add_embedding.linear_1.bias": "class_embedding.linear_1.bias",
                "add_embedding.linear_2.weight": "class_embedding.linear_2.weight",
                "add_embedding.linear_2.bias": "class_embedding.linear_2.bias",
            }
        )
        ```

    Args:

        checkpoint_state_dict (Dict[str, torch.Tensor]): The checkpoint state dict.
        rename_dict (Dict[str, str]): The dictionary mapping the old keys to new keys
    """

    def __call__(
        self,
        checkpoint_state_dict: Dict[str, torch.Tensor],
        rename_dict: Dict[str, str],
    ) -> Dict[str, torch.Tensor]:
        for old_key, new_key in rename_dict.items():
            if old_key not in checkpoint_state_dict:
                logging.warning(f"Key {old_key} not found in checkpoint state dict")
                continue
            else:
                assert (
                    new_key not in checkpoint_state_dict
                ), f"Key {new_key} already exists in checkpoint state dict"
                checkpoint_state_dict[new_key] = checkpoint_state_dict.pop(old_key)
                logging.info(f"Renaming {old_key} to {new_key}")
        return checkpoint_state_dict
