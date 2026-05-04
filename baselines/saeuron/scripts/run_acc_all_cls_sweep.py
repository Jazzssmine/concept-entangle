import os
import subprocess
import sys

import fire

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from UnlearnCanvas_resources.const import class_available


def run_scripts_sequentially(
    classes_to_unlearn,
    multiplier,
    percentile,
    input_dir_base,
    output_dir_base,
    class_ckpt,
    batch_size,
    seed,
):
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        PROJECT_ROOT
        if not existing_pythonpath
        else f"{PROJECT_ROOT}:{existing_pythonpath}"
    )

    for cls in classes_to_unlearn:
        command = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "accuracy_unlearncanvas_cls_sweep_fast.py"),
            "--input_dir",
            f"{input_dir_base}/percentile_{percentile}_multiplier_{multiplier}/",
            "--output_dir",
            f"{output_dir_base}/percentile_{percentile}_multiplier_{multiplier}/",
            "--class_ckpt",
            class_ckpt,
            "--cls",
            cls,
            "--batch_size",
            str(batch_size),
            "--seed",
            f"[{seed}]",
        ]
        print(f"Running command: {' '.join(command)}")
        process = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
        if process.returncode != 0:
            print(
                f"Error: Script failed with return code {process.returncode} for cls '{cls}'"
            )
            break
        else:
            print(f"Successfully completed script for cls '{cls}'")


def main(
    multipliers,
    percentiles,
    input_dir_base,
    output_dir_base,
    class_ckpt,
    batch_size,
    seed,
):
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        PROJECT_ROOT
        if not existing_pythonpath
        else f"{PROJECT_ROOT}:{existing_pythonpath}"
    )

    for multiplier in multipliers:
        for percentile in percentiles:
            run_scripts_sequentially(
                class_available,
                multiplier,
                percentile,
                input_dir_base,
                output_dir_base,
                class_ckpt,
                batch_size,
                seed,
            )
            process = subprocess.run(
                [
                    sys.executable,
                    os.path.join(PROJECT_ROOT, "scripts", "avg_accuracy_cls_sweep.py"),
                    f"{output_dir_base}/percentile_{percentile}_multiplier_{multiplier}/",
                ],
                cwd=PROJECT_ROOT,
                env=env,
            )
            if process.returncode != 0:
                print("Error: Failed to run average accuracy calculation")


if __name__ == "__main__":
    fire.Fire(main)
