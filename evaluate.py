from typing import List
import yaml
import os
import json

import torch
import torch.distributed as dist

import pydantic
from omegaconf import OmegaConf
from pretrain import PretrainConfig, init_train_state, create_dataloader, TrainState
from puzzle_dataset import PuzzleDatasetMetadata
#OMP_NUM_THREADS=8 torchrun --nproc-per-node 1 evaluate.py checkpoint=checkpoints/HRM-checkpoint-sudoku-extreme/checkpoint data_path=data/sudoku-extreme-1k-aug-1000

class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    
    save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"]

def evaluate_custom(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    with torch.inference_mode():
        outputs = []
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        
        all_preds = {}

        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]
        
        carry = None
        for set_name, batch, global_batch_size in eval_loader:
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            # Collect per-step outputs
            _step_probs_list = []
            _step_pred_ids_list = []
            while True:
                carry, _, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=config.eval_save_outputs)
                # Capture per-step predictions/probabilities
                if "logits" in preds:
                    _logits_step = preds["logits"].detach().to(torch.float32)
                    _probs_step = torch.softmax(_logits_step, dim=-1)
                    _pred_ids_step = torch.argmax(_probs_step, dim=-1)
                    _step_probs_list.append(_probs_step)
                    _step_pred_ids_list.append(_pred_ids_step)
                
                if all_finish:
                    break

            if ("labels" in batch) and ("logits" in preds):
                labels = batch["labels"].detach()
                inputs_tensor = batch.get("inputs", None)
                if inputs_tensor is not None:
                    inputs_tensor = inputs_tensor.detach()
                logits = preds["logits"].detach()
                probs = torch.softmax(logits, dim=-1)
                pred_ids = torch.argmax(probs, dim=-1)
                puzzle_ids = batch.get("puzzle_identifiers", None)
                batch_size = labels.shape[0]
                for i in range(batch_size):
                    step_preds = [t[i].cpu().tolist() for t in _step_pred_ids_list] if len(_step_pred_ids_list) else []
                    step_logits = [t[i].cpu().tolist() for t in _step_probs_list] if len(_step_probs_list) else []
                    valid_mask = labels[i] != -100
                    if len(_step_pred_ids_list) and valid_mask.any():
                        for s, _pred_ids_step in enumerate(_step_pred_ids_list):
                            if (_pred_ids_step[i][valid_mask] == labels[i][valid_mask]).all().item():
                                step_preds = step_preds[: s + 1]
                                step_logits = step_logits[: s + 1]
                                break
                    if valid_mask.any():
                        acc = (pred_ids[i][valid_mask] == labels[i][valid_mask]).to(torch.float32).mean().item()
                    else:
                        acc = 0.0
                    item = {
                        "input": inputs_tensor[i].cpu().tolist() if inputs_tensor is not None else None,
                        "target": labels[i].cpu().tolist(),
                        "preds": pred_ids[i].cpu().tolist(),
                        "logits": probs[i].cpu().tolist(),
                        "step_preds": step_preds,
                        "step_logits": step_logits,
                        "accuracy": round(acc, 4),
                    }
                    if puzzle_ids is not None:
                        item["puzzle_id"] = int(puzzle_ids[i].detach().cpu().item())
                    outputs.append(item)

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory
                        
            del carry, preds, batch, all_finish

            # Aggregate
            set_id = set_ids[set_name]
            
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")
                
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))

        # Logging
        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)
            
            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {set_name: {metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
                                   for set_id, set_name in enumerate(set_ids)}
                
                # Postprocess
                for set_name, metrics in reduced_metrics.items():
                    count = metrics.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}

                print(f'length of outputs: {len(outputs)}')
                with open("outputs.json", "w") as f:
                    json.dump(outputs[:25], f)

                return reduced_metrics


def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore
    
    RANK = 0
    WORLD_SIZE = 1
    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    with open(os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))

        config.eval_save_outputs = eval_cfg.save_outputs
        config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)

    # Dataloader
    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    MAX_EVAL_EXAMPLES = 5
    def _limit_loader_by_examples(loader, max_examples):
        def _gen():
            seen = 0
            for set_name, batch, global_batch_size in loader:
                if seen >= max_examples:
                    break
                seen += global_batch_size
                yield set_name, batch, global_batch_size
        return _gen()

    eval_loader = _limit_loader_by_examples(eval_loader, MAX_EVAL_EXAMPLES)

    # Models
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    # Try unwrap torch.compile
    try:
        train_state.model.load_state_dict(torch.load(eval_cfg.checkpoint, map_location="cuda"), assign=True)
    except:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in torch.load(eval_cfg.checkpoint, map_location="cuda").items()}, assign=True)
    
    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("step_"):
        train_state.step = int(ckpt_filename.removeprefix("step_"))

    # Evaluate
    print ("Starting evaluation")
    
    train_state.model.eval()
    metrics = evaluate_custom(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

    if metrics is not None:
        print (metrics)


if __name__ == "__main__":
    launch()
