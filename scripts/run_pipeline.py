import argparse
import jax
import os
from mpssba.config import load_config_from_json, setup_experiment_dir
from mpssba.batch_runner import BatchRunner
from mpssba.post_processor import PostProcessor

jax.config.update("jax_enable_x64", True)

def parse_train_args():
    parser = argparse.ArgumentParser(description="MPS Pipeline")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config JSON"
    )
    args = parser.parse_args()

    config = load_config_from_json(args.config)
    
    return config

def main():
    config = parse_train_args()
    setup_experiment_dir(config)

    batcher = BatchRunner(config)

    if config.batch.continue_run:
        orig_config = load_config_from_json(os.path.join(config.batch.previous_run_path, "config.json"))
        batcher.execute_continued_runs(orig_config)
        return

    if config.batch.runs > 1:
        batcher.execute_batch_runs()
        return
    
    batcher.execute_single_run()

    print("\n=== Starting Post-Processing  ===")
    processor = PostProcessor(config)
    processor.process()
    

if __name__ == "__main__":
    main()