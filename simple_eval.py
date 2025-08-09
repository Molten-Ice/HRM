import os
import yaml
import json

import torch

from pretrain import PretrainConfig, ArchConfig, LossConfig, init_train_state, create_dataloader


EXAMPLE_INDEX = 0


def load_arch_config() -> dict:
    arch_cfg_path = os.path.join(os.path.dirname(__file__), "config", "arch", "hrm_v1.yaml")
    with open(arch_cfg_path, "r") as f:
        arch_cfg = yaml.safe_load(f)
    if isinstance(arch_cfg.get("puzzle_emb_ndim"), str) and "hidden_size" in arch_cfg:
        arch_cfg["puzzle_emb_ndim"] = arch_cfg["hidden_size"]
    return arch_cfg


def build_config() -> PretrainConfig:
    arch_cfg = load_arch_config()

    arch = ArchConfig(
        name=arch_cfg["name"],
        loss=LossConfig(name=arch_cfg["loss"]["name"]),
        **{k: v for k, v in arch_cfg.items() if k not in ("name", "loss")},
    )
    arch.loss.__pydantic_extra__ = {"loss_type": arch_cfg["loss"]["loss_type"]}  # type: ignore[attr-defined]

    cfg = PretrainConfig(
        arch=arch,
        data_path=os.path.join(os.path.dirname(__file__), "data", "sudoku-extreme-1k-aug-1000"),
        global_batch_size=1,
        epochs=1,
        lr=1e-4,
        lr_min_ratio=1.0,
        lr_warmup_steps=0,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        puzzle_emb_lr=1e-2,
        puzzle_emb_weight_decay=0.0,
        seed=0,
        checkpoint_every_eval=False,
        eval_interval=None,
        eval_save_outputs=["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"],
        project_name=None,
        run_name=None,
        checkpoint_path=None,
    )
    return cfg


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def chunk_rows(seq_list, row_len):
    return [seq_list[i:i + row_len] for i in range(0, len(seq_list), row_len)]


def main() -> None:
    os.environ["DISABLE_COMPILE"] = "1"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this quick smoke test.")
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    config = build_config()

    train_loader, train_metadata = create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1
    )
    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=0, world_size=1
    )

    train_state = init_train_state(config, train_metadata, world_size=1)
    train_state.model.eval()

    seen = 0
    for set_name, batch, global_batch_size in eval_loader:
        if seen + global_batch_size <= EXAMPLE_INDEX:
            seen += global_batch_size
            continue

        batch = to_device(batch, device)

        with torch.inference_mode():
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore[attr-defined]

            last_preds = None
            steps_taken = 0
            while True:
                carry, _loss, _metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=config.eval_save_outputs
                )
                steps_taken += 1
                last_preds = preds
                if all_finish:
                    break

        b0_inputs = batch["inputs"][0].detach().cpu().tolist()
        b0_labels = batch["labels"][0].detach().cpu().tolist()

        logits = last_preds.get("logits") if last_preds is not None else None
        if logits is not None:
            b0_pred_tokens = torch.argmax(logits[0], dim=-1).detach().cpu().tolist()
            b0_logits = logits[0].detach().cpu().tolist()
        else:
            b0_pred_tokens = None
            b0_logits = None

        q_halt = last_preds.get("q_halt_logits")[0].item() if last_preds and "q_halt_logits" in last_preds else None
        q_cont = last_preds.get("q_continue_logits")[0].item() if last_preds and "q_continue_logits" in last_preds else None

        print("Set:", set_name)
        print("Steps taken:", steps_taken)
        print("Inputs[0]:", b0_inputs)
        print("Labels[0]:", b0_labels)
        print("PredTokens[0]:", b0_pred_tokens)
        print("Q_logits (halt, continue):", (q_halt, q_cont))

        seq_len = eval_metadata.seq_len
        side = int(seq_len ** 0.5)
        if side * side == seq_len:
            grid_pred = chunk_rows(b0_pred_tokens, side) if b0_pred_tokens is not None else None
            grid_target = chunk_rows(b0_labels, side)
            grid_logits = chunk_rows(b0_logits, side) if b0_logits is not None else None
        else:
            grid_pred = [b0_pred_tokens] if b0_pred_tokens is not None else None
            grid_target = [b0_labels]
            grid_logits = [b0_logits] if b0_logits is not None else None

        result_obj = {
            "pred": grid_pred,
            "target": grid_target,
            "logits": grid_logits,
        }

        print(json.dumps(result_obj))

        break


if __name__ == "__main__":
    main()
