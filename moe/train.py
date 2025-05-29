# EXPERIMENTAL MOE TRAINING SCRIPT 
# WARNING: EXPECT BUGS, CRASHES, AND QUESTIONABLE LIFE CHOICES
# This is research code - it's ugly but it works (sometimes)
# Based on distributed expert parallelism + data parallelism hybrid approach

import os
import sys
import time
import uuid
import traceback
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
# almost tried FSDP with no_shard, but DDP is faster even though older
# FSDP has better memory but expert routing breaks with sharding anyway
from model import moegpt  # our beautiful mess of a model
from torch.cuda.amp import autocast
import torch._dynamo
import wandb  # TODO: add this for logging
import json

# CUDA setup - pray this doesn't break
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CC'] = 'gcc'  # sometimes helps with compilation
assert torch.cuda.is_available(), "CUDA required - this ain't gonna work on CPU"

# DEBUG FLAGS - turn these on when things inevitably break
DEBUG_DISTRIBUTED = False
DEBUG_DATA_LOADING = False  
DEBUG_EXPERT_ROUTING = False
PROFILE_MEMORY = False

print(f"[INIT] Python version: {sys.version}")
print(f"[INIT] PyTorch version: {torch.__version__}")
print(f"[INIT] CUDA version: {torch.version.cuda}")

@dataclass
class ExperimentConfig:
    """
    Configuration for MoE training experiments
    
    Lots of hyperparameters to tune - this is where the magic happens
    Based on Rishi's findings: need 10B+ tokens for MoE to work well
    """
    # Model architecture
    hidden_dim: int = 768  # GPT-2 scale for fast iteration
    num_blocks: int = 12   # deep enough to be interesting
    vocab_size: int = 50257  # GPT-2 vocab, could extend to power of 2
    num_experts: int = 8   # distributed across GPUs
    expert_k: int = 2      # top-k routing
    
    # Training hyperparameters  
    train_files: str = "data/fineweb10B/train/*.bin"  
    val_files: str = "data/fineweb10B/val/*.bin"
    batch_size: int = 16 * 1024 * 8  # aggressive batch size - might OOM
    learning_rate: float = 3e-4      # could try different LRs for different components
    warmup_steps: int = 1000         # warmup is critical for stability
    max_steps: int = 50000           # adjust based on patience
    
    # MoE specific
    balance_loss_weight: float = 0.001  # from Switch Transformer
    aux_loss_weight: float = 0.01       # additional regularization
    
    # Sequence lengths - FlexAttention allows variable length
    train_seq_len: int = 16 * 1024 * 8  # push the limits
    val_seq_len: int = 4 * 1024         # smaller for faster validation
    
    # Validation and logging
    val_interval: int = 200    # validate frequently to catch issues
    val_tokens: int = 10_000   # enough for reasonable estimates
    log_interval: int = 10     # spam the logs
    save_interval: int = 1000  # save often, disk is cheap
    
    # Paths and IDs
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    run_name: str = None  # will be auto-generated
    
    # Optimization tricks
    gradient_clipping: float = 1.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Compilation and performance
    use_compile: bool = True  # torch.compile can be buggy but fast
    use_mixed_precision: bool = True
    
    def __post_init__(self):
        """Generate run name and validate config"""
        if self.run_name is None:
            self.run_name = f"moegpt_{self.hidden_dim}d_{self.num_experts}e_{int(time.time())}"
        
        # sanity checks
        assert self.num_experts % dist.get_world_size() == 0 if dist.is_initialized() else True
        assert self.batch_size % self.train_seq_len == 0, "batch_size should be divisible by seq_len"
        
        print(f"[CONFIG] Experiment: {self.run_name}")
        print(f"[CONFIG] Model: {self.hidden_dim}d x {self.num_blocks}L x {self.num_experts}E")
        print(f"[CONFIG] Training: {self.max_steps} steps, batch={self.batch_size}")


def setup_distributed_training():
    """
    Initialize distributed training environment
    
    This is where things usually break - distributed training is hard
    Expert parallelism + data parallelism = communication nightmare
    """
    # get distributed info from environment (torchrun sets these)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    print(f"[DIST] Rank {rank}/{world_size}, Local rank {local_rank}")
    
    # setup GPU
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # initialize process group for communication
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=torch.distributed.default_pg_timeout)
        print(f"[DIST] Process group initialized: {dist.get_backend()}")
    
    # optimize matrix multiplications
    torch.set_float32_matmul_precision('high')  # use TensorFloat-32
    
    # memory debugging
    if PROFILE_MEMORY and rank == 0:
        torch.cuda.memory._record_memory_history(True)
    
    return rank, world_size, device


def load_fineweb_shard(file_path: Path):
    """
    Load a FineWeb data shard from binary format
    
    Format: [header][tokens]
    Header: magic_number, version, num_tokens (each int32)
    Tokens: sequence of uint16 token ids
    """
    if DEBUG_DATA_LOADING:
        print(f"[DATA] Loading shard: {file_path}")
    
    # read header
    header = torch.from_file(str(file_path), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, f"Invalid magic number: {header[0]}"
    assert header[1] == 1, f"Unsupported version: {header[1]}"
    num_tokens = int(header[2])
    
    # read tokens
    with file_path.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)  # skip header
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, f"Read {nbytes} bytes, expected {2 * num_tokens}"
    
    if DEBUG_DATA_LOADING:
        print(f"[DATA] Loaded {num_tokens:,} tokens from {file_path}")
    
    return tokens


def create_data_iterator(filename_pattern: str, batch_size: int, rank: int, world_size: int, seq_len: int):
    """
    Create infinite iterator over training data
    
    Distributed data loading: each rank gets different portion of data
    Memory efficient: loads one shard at a time
    """
    files = sorted(Path.cwd().glob(filename_pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {filename_pattern}")
    
    print(f"[DATA] Found {len(files)} data shards")
    
    local_batch_size = batch_size // world_size
    file_iterator = iter(files)
    tokens, position = load_fineweb_shard(next(file_iterator)), 0
    
    step_count = 0
    while True:
        # check if we need to load next shard
        if position + batch_size + 1 >= len(tokens):
            try:
                tokens, position = load_fineweb_shard(next(file_iterator)), 0
            except StopIteration:
                # wrap around to beginning
                file_iterator = iter(files) 
                tokens, position = load_fineweb_shard(next(file_iterator)), 0
                print(f"[DATA] Wrapped around to beginning of dataset")
        
        # get data slice for this rank
        seq_start = position + rank * local_batch_size
        seq_end = seq_start + local_batch_size + 1
        sequence = tokens[seq_start:seq_end]
        
        # split into chunks of seq_len for FlexAttention
        for i in range(0, len(sequence) - 1, seq_len):
            chunk = sequence[i:i + seq_len + 1]
            if len(chunk) <= 1:
                continue
                
            # prepare inputs and targets
            inputs = chunk[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
            targets = chunk[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
            
            if DEBUG_DATA_LOADING and step_count % 100 == 0:
                print(f"[DATA] Step {step_count}: inputs shape {inputs.shape}")
            
            step_count += 1
            yield inputs, targets
            
        position += batch_size


def run_validation(model, config, rank, world_size):
    """
    Run validation and compute perplexity
    
    Important: set model to eval mode to disable dropout/batchnorm
    Use smaller sequences for faster validation
    """
    model.eval()
    
    val_iterator = create_data_iterator(
        config.val_files,
        config.batch_size,
        rank, 
        world_size,
        config.val_seq_len
    )
    
    total_loss = 0.0
    total_aux_loss = 0.0
    num_batches = 0
    tokens_processed = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        while tokens_processed < config.val_tokens:
            try:
                inputs, targets = next(val_iterator)
                
                # forward pass
                logits, aux_loss = model(inputs)
                
                # compute main loss
                main_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    targets.view(-1),
                    reduction='mean'
                )
                
                total_loss += main_loss.item()
                total_aux_loss += aux_loss.item()
                num_batches += 1
                tokens_processed += inputs.numel()
                
            except Exception as e:
                print(f"[VAL] Error during validation: {e}")
                if DEBUG_DISTRIBUTED:
                    traceback.print_exc()
                break
    
    # average across all ranks
    avg_loss = total_loss / max(num_batches, 1)
    avg_aux_loss = total_aux_loss / max(num_batches, 1)
    
    if world_size > 1:
        # synchronize losses across ranks
        loss_tensor = torch.tensor([avg_loss, avg_aux_loss], device="cuda")
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss, avg_aux_loss = loss_tensor.tolist()
    
    val_time = time.time() - start_time
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    model.train()  # back to training mode
    
    return {
        'loss': avg_loss,
        'aux_loss': avg_aux_loss, 
        'perplexity': perplexity,
        'time': val_time,
        'tokens': tokens_processed
    }


def save_model_checkpoint(model, optimizer, scaler, step, loss, config, rank):
    """
    Save model checkpoint with all the necessary state
    
    Only save from rank 0 to avoid conflicts
    Include config, optimizer state, and scaler state
    """
    if rank != 0:
        return
        
    os.makedirs(config.save_dir, exist_ok=True)
    
    # prepare checkpoint data
    checkpoint = {
        'step': step,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'loss': loss,
        'config': config,
        'timestamp': time.time()
    }
    
    # save with step number for easy identification
    checkpoint_path = os.path.join(config.save_dir, f'{config.run_name}_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # also save as latest for easy resuming
    latest_path = os.path.join(config.save_dir, f'{config.run_name}_latest.pt')
    torch.save(checkpoint, latest_path)
    
    print(f"[SAVE] Checkpoint saved: {checkpoint_path}")


def setup_logging(config, rank):
    """Setup logging infrastructure"""
    if rank == 0:
        os.makedirs(config.log_dir, exist_ok=True)
        
        # text log file
        log_file = os.path.join(config.log_dir, f"{config.run_name}.log")
        
        # JSON metrics file for easier parsing
        metrics_file = os.path.join(config.log_dir, f"{config.run_name}_metrics.jsonl")
        
        print(f"[LOG] Logging to {log_file}")
        print(f"[LOG] Metrics to {metrics_file}")
        
        return log_file, metrics_file
    return None, None


def log_metrics(metrics, step, metrics_file=None, rank=0):
    """Log training metrics"""
    if rank == 0:
        # console output
        print(f"[STEP {step:6d}] " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
        
        # JSON log for analysis
        if metrics_file:
            with open(metrics_file, 'a') as f:
                json.dump({'step': step, **metrics, 'timestamp': time.time()}, f)
                f.write('\n')


def get_learning_rate(step, config):
    """
    Learning rate schedule with warmup and decay
    
    Warmup prevents early instability
    Cosine decay helps convergence
    """
    if step < config.warmup_steps:
        # linear warmup
        return config.learning_rate * (step / config.warmup_steps)
    else:
        # cosine decay after warmup
        progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        return config.learning_rate * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())


def main():
    """
    Main training loop - where the magic happens (or everything breaks)
    """
    print("="*80)
    print("STARTING EXPERIMENTAL MOE TRAINING")
    print("="*80)
    
    # reset compilation cache to avoid weird bugs
    torch._dynamo.reset()
    
    try:
        # setup distributed training
        rank, world_size, device = setup_distributed_training()
        
        # load configuration
        config = ExperimentConfig()
        
        # setup logging  
        log_file, metrics_file = setup_logging(config, rank)
        
        print(f"[INIT] Creating model on rank {rank}")
        
        # create model
        model = moegpt(
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            vocab=config.vocab_size
        ).to(device)
        
        # compile model for speed (might break things)
        if config.use_compile:
            print(f"[INIT] Compiling model...")
            model = torch.compile(model, dynamic=False)
        
        # wrap with DDP for multi-GPU training
        if world_size > 1:
            model = DDP(model, device_ids=[device], find_unused_parameters=True)
            print(f"[INIT] Model wrapped with DDP")
        
        # setup optimizer - could try different optimizers
        # AdamW is safe choice, could experiment with Lion, AdamScale, etc.
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            eps=1e-8
        )
        
        # setup mixed precision training
        scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # create data iterator
        print(f"[INIT] Setting up data loading...")
        train_iterator = create_data_iterator(
            config.train_files,
            config.batch_size,
            rank,
            world_size, 
            config.train_seq_len
        )
        
        print(f"[INIT] Starting training loop...")
        
        # training state
        step = 0
        best_val_loss = float('inf')
        training_start_time = time.time()
        last_log_time = time.time()
        
        # main training loop
        model.train()
        
        while step < config.max_steps:
            try:
                # get batch
                inputs, targets = next(train_iterator)
                
                # update learning rate
                current_lr = get_learning_rate(step, config)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # forward pass with mixed precision
                optimizer.zero_grad()
                
                if config.use_mixed_precision:
                    with autocast():
                        logits, aux_loss = model(inputs)
                        main_loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            targets.view(-1),
                            reduction='mean'
                        )
                        total_loss = main_loss + config.balance_loss_weight * aux_loss
                else:
                    logits, aux_loss = model(inputs)
                    main_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        reduction='mean'
                    )
                    total_loss = main_loss + config.balance_loss_weight * aux_loss
                
                # backward pass
                if config.use_mixed_precision:
                    scaler.scale(total_loss).backward()
                    # gradient clipping
                    if config.gradient_clipping > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if config.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
                    optimizer.step()
                
                # logging
                if step % config.log_interval == 0:
                    current_time = time.time()
                    tokens_per_sec = config.batch_size / (current_time - last_log_time + 1e-8)
                    
                    metrics = {
                        'train_loss': main_loss.item(),
                        'aux_loss': aux_loss.item(),
                        'total_loss': total_loss.item(),
                        'lr': current_lr,
                        'tokens_per_sec': tokens_per_sec,
                    }
                    
                    log_metrics(metrics, step, metrics_file, rank)
                    last_log_time = current_time
                
                # validation
                if step % config.val_interval == 0 and step > 0:
                    print(f"[VAL] Running validation at step {step}")
                    val_metrics = run_validation(model, config, rank, world_size)
                    
                    val_metrics_prefixed = {f'val_{k}': v for k, v in val_metrics.items()}
                    log_metrics(val_metrics_prefixed, step, metrics_file, rank)
                    
                    # save best model
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        if rank == 0:
                            best_path = os.path.join(config.save_dir, f'{config.run_name}_best.pt')
                            checkpoint = {
                                'step': step,
                                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                                'val_loss': val_metrics['loss'],
                                'config': config
                            }
                            torch.save(checkpoint, best_path)
                            print(f"[SAVE] New best model saved: {best_path}")
                
                # checkpoint saving
                if step % config.save_interval == 0 and step > 0:
                    save_model_checkpoint(model, optimizer, scaler, step, total_loss.item(), config, rank)
                
                step += 1
                
            except KeyboardInterrupt:
                print(f"\n[STOP] Training interrupted at step {step}")
                break
            except Exception as e:
                print(f"[ERROR] Training error at step {step}: {e}")
                if DEBUG_DISTRIBUTED:
                    traceback.print_exc()
                # try to continue training
                step += 1
                continue
        
        # final checkpoint
        save_model_checkpoint(model, optimizer, scaler, step, total_loss.item(), config, rank)
        
        total_time = time.time() - training_start_time
        print(f"\n[DONE] Training completed in {total_time:.1f}s")
        print(f"[DONE] Best validation loss: {best_val_loss:.4f}")
        
    except Exception as e:
        print(f"[FATAL] Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # cleanup
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # entry point
    main()


# TODO LIST (things that definitely need fixing):
# - Add gradient accumulation for huge batch sizes  
# - Implement proper checkpointing for memory efficiency
# - Add expert utilization monitoring and alerting
# - Try different expert architectures (conv experts?)
# - Implement dynamic expert assignment based on loss
# - Add proper wandb integration for experiment tracking
# - Fix the inevitable memory leaks from distributed training
# - Add automatic mixed precision tuning
# - Implement expert capacity buffers to prevent overflows
# - Add curriculum learning for better MoE sample efficiency
# - Try different load balancing strategies (entropy, switch routing)
# - Add model EMA for better validation performance
# - Implement proper data resumption for checkpoint recovery
# - Add automatic hyperparameter tuning (Optuna?)
# - Fix all the race conditions in distributed expert routing
