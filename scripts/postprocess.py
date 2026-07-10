import argparse
import os
import json
from mpssba.config import Config
from mpssba.post_processor import PostProcessor

def main():
    parser = argparse.ArgumentParser(description="Recalculate Post-Processing for runs")
    parser.add_argument("--runs", nargs="*", help="List of run directories to process")
    parser.add_argument("--all", action="store_true", help="Process all runs in results/runs/")
    args = parser.parse_args()

    if not args.runs and not args.all:
        print("Please specify --runs <run_dirs> or --all")
        return

    runs = []
    if args.all:
        runs_dir = os.path.join("results", "runs")
        if os.path.exists(runs_dir):
            runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        else:
            print(f"Directory {runs_dir} not found.")
            return
    else:
        runs = args.runs

    for run_dir in runs:
        print(f"\nProcessing run: {run_dir}")
        config_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"Skipping {run_dir}: config.json not found.")
            continue

        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        config = Config.from_dict(config_dict)

        config.output.raw_dir = os.path.join(run_dir, "raw")
        config.output.processed_dir = os.path.join(run_dir, "processed")
        
        os.makedirs(config.output.processed_dir, exist_ok=True)

        runs_count = config.batch.runs
        postfix_fmt = config.batch.postfix

        processor = PostProcessor(config)

        if runs_count > 1:
            for run_idx in range(runs_count):
                postfix = postfix_fmt.format(run_idx=run_idx)
                print(f"  Post-processing batch {run_idx+1}/{runs_count} (postfix: '{postfix}')")
                try:
                    processor.process(postfix=postfix)
                except Exception as e:
                    print(f"  Error processing batch {run_idx+1}: {e}")
        else:
            print("  Post-processing single run")
            try:
                processor.process(postfix="")
            except FileNotFoundError:
                try:
                    processor.process(postfix=postfix_fmt.format(run_idx=0))
                except Exception as e:
                    print(f"  Error processing: {e}")
            except Exception as e:
                print(f"  Error processing: {e}")

if __name__ == "__main__":
    main()
