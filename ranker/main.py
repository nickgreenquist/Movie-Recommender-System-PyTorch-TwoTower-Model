"""
Ranker pipeline entry point.

Usage:
    python ranker/main.py precompute                # Stage 0: generate candidates (default PROD CG)
    python ranker/main.py precompute <cg.pth>       # Stage 0: override CG checkpoint
    python ranker/main.py train                     # Stage 1+: train MLP ranker
    python ranker/main.py evaluate <ranker.pth>     # Eval-only: load ranker checkpoint, report metrics
    python ranker/main.py canary                    # Side-by-side CG top-10 vs Ranker top-10 on canaries
    python ranker/main.py dump                      # Dump top-20 for ALL canaries → canary_results/<ckpt>.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.canary import dump_canary, run_canary
from ranker.precompute import precompute
from ranker.train import evaluate_only, train


USAGE = __doc__.strip()


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'precompute':
        ckpt = sys.argv[2] if len(sys.argv) > 2 else None
        precompute(checkpoint_path=ckpt)
    elif cmd == 'train':
        train()
    elif cmd == 'evaluate':
        if len(sys.argv) < 3:
            print("evaluate requires a checkpoint path: python ranker/main.py evaluate <ckpt.pth>")
            sys.exit(1)
        evaluate_only(sys.argv[2])
    elif cmd == 'canary':
        run_canary()
    elif cmd == 'dump':
        dump_canary()
    else:
        print(f"Unknown command: {cmd}\n")
        print(USAGE)
        sys.exit(1)


if __name__ == '__main__':
    main()
