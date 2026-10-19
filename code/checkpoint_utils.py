"""Load task tensors while allowing only explicitly omitted base tensors."""
from __future__ import annotations


def load_task_checkpoint(model, checkpoint):
    def normalize(key):
        return key.removeprefix('_orig_mod.')

    slim = 'slim_keep' in checkpoint
    state = checkpoint.get('slim_keep') if slim else checkpoint.get('model_ema')
    if state is None:
        state = checkpoint.get('model')
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint has no model tensors')
    original_count = len(state)
    state = {normalize(k): v for k, v in state.items()}
    if len(state) != original_count:
        raise ValueError('Colliding checkpoint keys after removing _orig_mod prefix')
    expected = set(model.state_dict())
    missing = expected - set(state)
    unexpected = set(state) - expected
    allowed = {normalize(k) for k in checkpoint.get('dropped_from_base', [])} if slim else set()
    invalid_missing = missing - allowed
    trainable_missing = {k for k, p in model.named_parameters() if p.requires_grad} & missing
    if invalid_missing or unexpected or trainable_missing:
        raise ValueError(
            'Checkpoint/config mismatch: '
            f'missing={sorted(invalid_missing)[:12]}, '
            f'unexpected={sorted(unexpected)[:12]}, '
            f'trainable_missing={sorted(trainable_missing)[:12]}'
        )
    return model.load_state_dict(state, strict=not slim)
