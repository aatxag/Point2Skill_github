#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys

DIT_DIR    = "/home/labiiwa/Point2Skill_github/dit-policy"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RN_WEIGHTS = f"{DIT_DIR}/visual_features/resnet18/IN_1M_resnet18.pth"

# img_chunk se define por run — no va en COMMON
COMMON = [
    "hydra/launcher=basic",
    f"agent.features.restore_path={RN_WEIGHTS}",
    "agent/features=resnet_gn",
    "trainer.schedule_builder.schedule_kwargs.num_warmup_steps=500",
    "ac_chunk=8",
    "batch_size=128",
    "num_workers=2",
    "train_transform=preproc",
    "max_iterations=10000",
    "eval_freq=500",
    "save_freq=500",
    "devices=1",
    "wandb.entity=aatxag-mondragon-university",
    "wandb.project=dit-policy",
]

_BUF_CLOSE = f"{DIT_DIR}/data_robobuf/close/buf.pkl"
_BUF_PICK  = f"{DIT_DIR}/data_robobuf/pick_coffee/buf.pkl"

_EXTRA_CONTACT = [
    "agent=diffusion_contact",
    "task=franka_2cam_contact",
    "trainer=bc_contact",
    "task.train_buffer.n_test_trans=800",
]

_EXTRA_DIT = [
    "agent=diffusion",
    "task=franka_2cam",
    "trainer=bc_cos_sched",
    "task.train_buffer.n_test_trans=800",
]

RUNS = [
    # ── diffusion_contact ─────────────────────────────────────────────────────
       {
        "script":      "finetune_contact.py",
        "exp_name":    "pick_coffee",
        "buffer_path": _BUF_PICK,
        "extra":       _EXTRA_CONTACT + ["img_chunk=1"],
    },

    {
        "script":      "finetune_contact.py",
        "exp_name":    "close",
        "buffer_path": _BUF_CLOSE,
        "extra":       _EXTRA_CONTACT + ["img_chunk=1"],
    },

]


def build_cmd(run: dict) -> list[str]:
    script = os.path.join(_SCRIPT_DIR, run["script"])
    return (
        ["python3", script]
        + COMMON
        + run["extra"]
        + [
            f"exp_name={run['exp_name']}",
            f"buffer_path={run['buffer_path']}",
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Muestra los comandos sin ejecutarlos")
    args = parser.parse_args()

    total = len(RUNS)
    print(f"\n{'='*60}")
    print(f"  Prueba training — {total} runs en cola")
    print(f"{'='*60}")
    for i, run in enumerate(RUNS, 1):
        print(f"  {i}. {run['exp_name']}  [{run['script']}]")
    print()

    for i, run in enumerate(RUNS, 1):
        cmd = build_cmd(run)

        print(f"\n{'='*60}")
        print(f"Run {i}/{total}: {run['exp_name']}")
        print(f"{'='*60}\n")

        if args.dry_run:
            print(" \\\n  ".join(cmd))
            continue

        result = subprocess.run(cmd, cwd=DIT_DIR)
        if result.returncode != 0:
            print(f"\n[ERROR] '{run['exp_name']}' falló (código {result.returncode}). Abortando.")
            sys.exit(result.returncode)

    if not args.dry_run:
        print(f"\n¡{total} runs completados con éxito!")


if __name__ == "__main__":
    main()
