"""
Make an inference checkpoint from a CLARA training checkpoint.

The training checkpoint (about 3 GB) holds, besides the weights and the
`args`/`config` hyper-parameters:

- the training state (optimizer states, lr schedulers, loops, callbacks),
  which inference doesn't use
- the boundary `MDPDatastore` used during training, pickled in
  `hyper_parameters["datastore_boundary"]`. It holds lazy (dask) arrays
  pointing at the training data on the training cluster, so it can only be
  unpickled with stubs, and it isn't needed for inference:
  `train_model --eval` passes the inference datastores to
  `load_from_checkpoint`

Both are dropped. The weights (float32), `args` and `config` are kept
unchanged, which is checked after writing.

Usage:
    python -m clara_inference.artifact.sanitize_checkpoint \\
        <checkpoint_in> <checkpoint_out>
"""
import argparse
import pickle
import types

import torch

HPARAMS_TO_DROP = ["datastore_boundary"]
KEYS_TO_KEEP = [
    "epoch",
    "global_step",
    "pytorch-lightning_version",
    "state_dict",
    "hparams_name",
    "hyper_parameters",
]


class _Unloadable:
    """Stand-in for classes that can't be imported (e.g. zarr v2 classes)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(f"{module}.{name}", (_Unloadable,), {})


_pickle_module = types.ModuleType("sanitize_pickle")
_pickle_module.__dict__.update(pickle.__dict__)
_pickle_module.Unpickler = _Unpickler


def _contains_unloadable(obj, _seen=None):
    _seen = set() if _seen is None else _seen
    if id(obj) in _seen:
        return False
    _seen.add(id(obj))
    if isinstance(obj, _Unloadable):
        return True
    if isinstance(obj, dict):
        return any(_contains_unloadable(v, _seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_unloadable(v, _seen) for v in obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return _contains_unloadable(vars(obj), _seen)
    return False


def load_checkpoint(fp_checkpoint):
    """
    Load a checkpoint, replacing classes that can't be imported with stubs.
    """
    return torch.load(
        fp_checkpoint,
        map_location="cpu",
        weights_only=False,
        pickle_module=_pickle_module,
    )


def sanitize_checkpoint(ckpt):
    """
    The inference checkpoint of a loaded training checkpoint: only the
    entries in KEYS_TO_KEEP, without the hyper-parameters in
    HPARAMS_TO_DROP. Checks that nothing that couldn't be unpickled is left.
    """
    dropped = sorted(set(ckpt) - set(KEYS_TO_KEEP))
    print(f"Dropping {dropped}")
    out = {k: ckpt[k] for k in KEYS_TO_KEEP if k in ckpt}
    hparams = dict(out["hyper_parameters"])
    for key in HPARAMS_TO_DROP:
        if key in hparams:
            print(f"Dropping hyper_parameters[{key!r}]")
            del hparams[key]
    for key, value in hparams.items():
        if _contains_unloadable(value):
            raise ValueError(
                f"hyper_parameters[{key!r}] still contains objects that could "
                "not be unpickled, refusing to write a checkpoint with stubs"
            )
    out["hyper_parameters"] = hparams
    return out


def check_same_weights(ckpt_a, ckpt_b):
    """Raise unless the two checkpoints have identical state dicts."""
    sd_a, sd_b = ckpt_a["state_dict"], ckpt_b["state_dict"]
    if list(sd_a) != list(sd_b):
        raise ValueError("the state dict keys differ")
    for k in sd_a:
        if sd_a[k].dtype != sd_b[k].dtype or not torch.equal(sd_a[k], sd_b[k]):
            raise ValueError(f"state dict entry {k} differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint_in")
    parser.add_argument("checkpoint_out")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint_in)
    out = sanitize_checkpoint(ckpt)
    torch.save(out, args.checkpoint_out)
    # the result loads without stubs (neural-lam's own classes aside, the
    # args are an argparse.Namespace) and has the same weights
    reloaded = torch.load(
        args.checkpoint_out, map_location="cpu", weights_only=False
    )
    check_same_weights(ckpt, reloaded)
    n_params = sum(t.numel() for t in reloaded["state_dict"].values())
    print(
        f"Wrote {args.checkpoint_out}: {len(reloaded['state_dict'])} tensors, "
        f"{n_params:,} values, identical to the input; kept "
        f"{list(reloaded)}, hyper_parameters {list(reloaded['hyper_parameters'])}"
    )


if __name__ == "__main__":
    main()
