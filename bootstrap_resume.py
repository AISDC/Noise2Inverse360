#!/usr/bin/env python
"""
Bootstrap a resume.pth from an existing best-model checkpoint.

Use this when training was interrupted before --resume support existed,
so no resume.pth is present in TrainOutput/.

The optimizer state (Adam momentum/variance) is reset to its initial values.
This causes a brief re-adaptation period of a few hundred updates but does
not significantly affect the final result when resuming near the end of training.

Note on best-model checkpoints
-------------------------------
After bootstrapping, the first resumed epoch will overwrite best_lcl_model.pth
and best_edge_model.pth (because their thresholds default to inf/0).  To
preserve the existing checkpoints, pass the actual best values via
--best-lcl-loss, --best-edge, etc. if you know them.

Usage
-----
::

    (denoise) $ python bootstrap_resume.py \\
                    --config /path/to/experiment_config.yaml \\
                    --checkpoint /path/to/TrainOutput/best_val_model.pth \\
                    --epoch 1710 \\
                    --best-val-loss 0.053 \\
                    --best-val-epoch 1705
"""

import argparse
import pathlib
from copy import deepcopy

import yaml
import torch

from denoise.model import unet_ns_gn


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', required=True,
                        help='Path to the YAML config used for training')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_val_model.pth, best_lcl_model.pth, or best_edge_model.pth')
    parser.add_argument('--epoch', type=int, required=True,
                        help='Last completed epoch — training will resume from epoch+1')
    parser.add_argument('--best-val-loss', type=float, default=float('inf'),
                        help='Best validation L1 loss seen so far (default: inf)')
    parser.add_argument('--best-val-epoch', type=int, default=0,
                        help='Epoch at which --best-val-loss was achieved (default: 0)')
    parser.add_argument('--best-lcl-loss', type=float, default=float('inf'),
                        help='Best LCL loss seen so far (default: inf → overwrite on first epoch)')
    parser.add_argument('--best-lcl-epoch', type=int, default=0)
    parser.add_argument('--best-edge', type=float, default=0.0,
                        help='Best edge score seen so far (default: 0 → overwrite on first epoch)')
    parser.add_argument('--best-edge-epoch', type=int, default=0)
    parser.add_argument('--output', default=None,
                        help='Output path (default: resume.pth in the same directory as --checkpoint)')
    args = parser.parse_args()

    with open(args.config) as f:
        params = yaml.safe_load(f)

    # Reconstruct model and optimizer to get a structurally valid optimizer state.
    n_slices = params['train']['n_slices']
    model = unet_ns_gn(ich=n_slices, start_filter_size=16, channels_per_group=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['train']['lr'])

    # Load the saved weights into the model.
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])

    out_path = args.output or str(pathlib.Path(args.checkpoint).parent / 'resume.pth')

    resume = {
        'model_state_dict':     deepcopy(model.state_dict()),
        'optimizer_state_dict': deepcopy(optimizer.state_dict()),  # fresh Adam state
        'epoch':                args.epoch,
        'model_updates':        999_999,   # well past the warmup threshold
        'best_val_loss':        args.best_val_loss,
        'best_lcl_loss':        args.best_lcl_loss,
        'best_edge':            args.best_edge,
        'best_val_epoch':       args.best_val_epoch,
        'best_lcl_epoch':       args.best_lcl_epoch,
        'best_edge_epoch':      args.best_edge_epoch,
        'train_loss':           [],
        'val_loss':             [],
        'train_lcl_loss':       [],
        'val_lcl_loss':         [],
        'edge_values':          [],
        'continue_warmup':      False,
    }

    torch.save(resume, out_path)
    print("Saved: %s" % out_path)
    print()
    print("Resume training with:")
    print("  denoise train --config %s --gpus 0,1 --resume" % args.config)


if __name__ == '__main__':
    main()
