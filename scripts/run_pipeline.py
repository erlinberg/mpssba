import jax
from mpssba.config import parse_train_args
from mpssba.batch_runner import BatchRunner
from mpssba.post_processor import PostProcessor

jax.config.update("jax_enable_x64", True)

def main():
    config = parse_train_args()

    batcher = BatchRunner(config)
    if config.batch.runs > 1:
        batcher.execute_batch_runs()
    else:
        batcher.execute_single_run()

        print("\n=== Starting Post-Processing  ===")
        processor = PostProcessor(config)
        processor.process()
    

if __name__ == "__main__":
    main()