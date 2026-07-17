"""Checkpoint management utilities for Orion."""

from pathlib import Path
from typing import Dict, Any, Optional

try:  # Compatibility shim for protobuf >= 5.0
    from google.protobuf import message_factory as _message_factory

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):
        _message_factory.MessageFactory.GetPrototype = _message_factory.MessageFactory.GetMessageClass  # type: ignore[attr-defined]
except Exception:
    pass

import orbax.checkpoint as ocp
from flax import nnx


class CheckpointManager:
    """Manages saving and loading of model checkpoints."""
    
    def __init__(self, checkpoint_dir: str):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Orbax requires checkpoint directories to be absolute paths.
        self.checkpoint_dir = self.checkpoint_dir.resolve()
        self.checkpointer = ocp.StandardCheckpointer()
        
    def save_checkpoint(self, 
                       model: nnx.Module,
                       optimizer: nnx.Optimizer,
                       epoch: int,
                       best_val_loss: float,
                       global_step: int,
                       checkpoint_name: str = 'latest') -> None:
        """Save a checkpoint.
        
        Args:
            model: Model to save.
            optimizer: Optimizer state to save.
            epoch: Current epoch number.
            best_val_loss: Best validation loss so far.
            global_step: Global training step.
            checkpoint_name: Name for the checkpoint ('latest' or 'best').
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Split NNX modules into graphdef and state
        # We only save the state (parameters), the graphdef is recreated
        _, model_state = nnx.split(model)
        _, optimizer_state = nnx.split(optimizer)
        
        checkpoint_state = {
            'model': model_state,
            'optimizer': optimizer_state,
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'global_step': global_step,
        }
        self.checkpointer.save(checkpoint_path, checkpoint_state, force=True)
        self.checkpointer.wait_until_finished()
        
    def load_checkpoint(self,
                       model: nnx.Module,
                       optimizer: nnx.Optimizer,
                       checkpoint_name: str = 'latest') -> Optional[Dict[str, Any]]:
        """Load a checkpoint if it exists.
        
        Args:
            model: Model to restore weights into.
            optimizer: Optimizer to restore state into.
            checkpoint_name: Name of the checkpoint to load.
            
        Returns:
            Dictionary with restored state, or None if checkpoint doesn't exist.
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            return None
            
        try:
            # Split current modules to get state template
            model_graphdef, model_state = nnx.split(model)
            optimizer_graphdef, optimizer_state = nnx.split(optimizer)
            
            # Restore from checkpoint
            restore_target = {
                'model': model_state,
                'optimizer': optimizer_state,
                'epoch': 0,
                'best_val_loss': float('inf'),
                'global_step': 0,
            }

            restored = self.checkpointer.restore(
                checkpoint_path,
                target=restore_target,
            )
            
            # Merge restored state back into the modules
            nnx.update(model, restored['model'])
            nnx.update(optimizer, restored['optimizer'])
            
            # Return metadata
            return {
                'epoch': restored['epoch'],
                'best_val_loss': restored['best_val_loss'],
                'global_step': restored['global_step'],
            }
        except Exception as e:
            print(f"Could not restore checkpoint: {e}")
            return None

    def restore_model(self, model: nnx.Module,
                      checkpoint_name: str = 'best') -> nnx.Module:
        """Restore only the model weights (for inference).

        Reads the ``model`` subtree of the checkpoint directly with orbax,
        bypassing the optimizer entirely. This avoids needing to reconstruct
        the exact training-time optimizer (e.g. clip + adamw chain) just to
        satisfy orbax's structural equality check on ``opt_state``.

        Raises ``FileNotFoundError`` if the named checkpoint does not exist.
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Use PyTreeRestoreArgs(partial_restore=True) so orbax only restores
        # keys we provide (just `model`) and ignores `optimizer` / metadata
        # on disk. The StandardCheckpointer.restore(target=...) path is strict
        # and rejects partial targets.
        _, model_state = nnx.split(model)
        from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import (
            PyTreeCheckpointHandler, PyTreeRestoreArgs,
        )
        checkpointer = ocp.Checkpointer(PyTreeCheckpointHandler())
        restored = checkpointer.restore(
            checkpoint_path,
            args=PyTreeRestoreArgs(item={'model': model_state},
                                   partial_restore=True),
        )
        if restored is None:
            raise FileNotFoundError(f"Could not restore: {checkpoint_path}")
        nnx.update(model, restored['model'])
        return model

    def peek_metadata(self, checkpoint_name: str = 'latest') -> Optional[Dict[str, Any]]:
        """Read the scalar metadata of a checkpoint without loading weights.

        Returns a dict with ``epoch``, ``global_step``, ``best_val_loss`` (the
        scalar keys saved alongside the model/optimizer), or ``None`` if the
        named checkpoint does not exist. Used by the ``--continue-epochs`` flow
        to learn where to resume before constructing the LR schedule.
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            return None

        # Partial-restore only the scalar keys; the model/optimizer subtrees
        # on disk are ignored.
        from orbax.checkpoint._src.handlers.pytree_checkpoint_handler import (
            PyTreeCheckpointHandler, PyTreeRestoreArgs,
        )
        checkpointer = ocp.Checkpointer(PyTreeCheckpointHandler())
        target = {
            'epoch': 0,
            'global_step': 0,
            'best_val_loss': float('inf'),
        }
        restored = checkpointer.restore(
            checkpoint_path,
            args=PyTreeRestoreArgs(item=target, partial_restore=True),
        )
        if restored is None:
            return None
        return {
            'epoch': int(restored['epoch']),
            'global_step': int(restored['global_step']),
            'best_val_loss': float(restored['best_val_loss']),
        }
    
    def save_best_model(self,
                       model: nnx.Module,
                       optimizer: nnx.Optimizer,
                       epoch: int,
                       val_loss: float,
                       global_step: int) -> None:
        """Save the best model checkpoint.
        
        Args:
            model: Model to save.
            optimizer: Optimizer state.
            epoch: Current epoch.
            val_loss: Current validation loss (now the best).
            global_step: Global training step.
        """
        self.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=val_loss,
            global_step=global_step,
            checkpoint_name='best'
        )
        print(f"✓ Best model saved (val_loss: {val_loss:.6f})")


def setup_checkpointing(config: Dict[str, Any],
                       model: nnx.Module,
                       optimizer: nnx.Optimizer) -> tuple[CheckpointManager, int, float, int]:
    """Setup checkpoint manager and optionally restore from checkpoint.
    
    Args:
        config: Training configuration.
        model: Model (will be updated in-place if checkpoint exists).
        optimizer: Optimizer (will be updated in-place if checkpoint exists).
        
    Returns:
        Tuple of (checkpoint_manager, start_epoch, best_val_loss, global_step)
    """
    ckpt_manager = CheckpointManager(config['checkpoint_dir'])
    
    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0
    
    if config.get('resume_from_checkpoint', True):
        print("Checking for existing checkpoint...")
        restored = ckpt_manager.load_checkpoint(model, optimizer, 'latest')
        
        if restored is not None:
            start_epoch = restored['epoch'] + 1
            best_val_loss = restored['best_val_loss']
            global_step = restored['global_step']
            print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
        else:
            print("No checkpoint found, starting from scratch...")
    else:
        print("Starting from scratch (resume disabled)...")
    
    return ckpt_manager, start_epoch, best_val_loss, global_step

