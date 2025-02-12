import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
#I almost wanted to use FSDP with no_shard, but DDP is actually faster even though it is older
from model import moegpt
from torch.cuda.amp import autocast
import torch._dynamo


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CC'] = 'gcc'
assert torch.cuda.is_available(), "CUDA is required for training"






@dataclass
class TrainingConfig:
    # Model parameters
    hidden_dim: int = 768
    num_blocks: int = 12
    vocab_size: int = 50257 #could potentially expand vocab to nearest power of 2 for marginal efficiency gain
    
    # Training parameters
    train_files: str = "data/fineweb10B/train/*.bin"  # Path pattern to training data
    val_files: str = "data/fineweb10B/val/*.bin"      # Path pattern to validation data
    batch_size: int = 16* 1024 *8            # Batch size in tokens remember to switch back to 64 
    learning_rate: float = 3e-4 #different learning rates for different params might need to be added for efficiency
    num_epochs: int = 1
    val_interval: int = 200                # Validate every N steps
    val_tokens: int = 10_000               # Number of validation tokens to use
    max_seq_len: int = 16*1024 *8             # maximum sequence length for flex attention
    
    # Logging and checkpointing
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_interval: int = 1000              # Save checkpoint every N steps

def setup_distributed():
    """Initialize distributed training environment"""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision('high')
    
    #if world_size > 1:
        
    
    return rank, world_size, device

def load_data_shard(file: Path):
    """Load a binary data shard"""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "Invalid data file format"
    assert header[1] == 1, "Unsupported version"
    num_tokens = int(header[2])
    
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens

def get_batch_iterator(filename_pattern: str, batch_size: int, rank: int, world_size: int, seq_len: int):
    """Generate batches of data"""
    files = sorted(Path.cwd().glob(filename_pattern))
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = load_data_shard(next(file_iter)), 0
    
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = load_data_shard(next(file_iter)), 0
            
        # Get sequence for this rank
        seq_start = pos + rank * local_batch_size
        seq = tokens[seq_start:seq_start + local_batch_size + 1]
        
        # Split into chunks of seq_len
        for i in range(0, len(seq) - 1, seq_len):
            chunk = seq[i:i + seq_len + 1]
            if len(chunk) <= 1:  # Skip if too short
                continue
                
            inputs = chunk[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
            targets = chunk[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
            yield inputs, targets
            
        pos += batch_size

def validate(model, config, rank, world_size):
    """Run validation"""
    model.eval()
    val_loader = get_batch_iterator(
        config.val_files, 
        config.batch_size,
        rank,
        world_size,
        config.max_seq_len
    )
    
    total_loss = 0
    num_batches = 0
    tokens_processed = 0
    
    with torch.no_grad():
        while tokens_processed < config.val_tokens:
            inputs, targets = next(val_loader)
            logits,bal_loss = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
            tokens_processed += inputs.numel()
            
    avg_loss = total_loss / num_batches
    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss).cuda()
        dist.all_reduce(avg_loss_tensor)
        avg_loss = avg_loss_tensor.item() / world_size
        
    model.train()
    return avg_loss

def save_checkpoint(model, optimizer, step, loss, config, run_id):
    """Save a checkpoint"""
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
        
    checkpoint = {
        'step': step,
        'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    path = os.path.join(config.save_dir, f'checkpoint_{run_id}_step_{step}.pt')
    torch.save(checkpoint, path)

def main():
    # Initialize distributed setup
    
    torch._dynamo.reset() #you might want to comment this out if you are going to be doing more than 1 training run, it might cache a bugged graph, which is why I put this here
    
    rank, world_size, device = setup_distributed()
    config = TrainingConfig()
    
    # Create model and move to GPU
    model = PicoGPT(
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        vocab=config.vocab_size
    ).to(device)
    model=torch.compile(model)
    
    if world_size > 1:
        model = DDP(model, device_ids=[device])
    
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    
    run_id = str(uuid.uuid4())
    if rank == 0:
        os.makedirs(config.log_dir, exist_ok=True)
        log_file = os.path.join(config.log_dir, f"{run_id}.txt")
        print(f"Logging to {log_file}")
    
    # Training loop
    train_loader = get_batch_iterator(
        config.train_files,
        config.batch_size,
        rank,
        world_size,
        config.max_seq_len
    )
    
    model.train()
    step = 0
    best_val_loss = float('inf')
    training_start_time = time.time()
    
    while True:
        try:
            inputs, targets = next(train_loader)
            
            
            
            logits,bal_loss = model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss=loss+bal_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            
            if rank == 0 and step % 10 == 0:
                elapsed = time.time() - training_start_time
                print(f"Step {step} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")
                
            
            if step > 0 and step % config.val_interval == 0:
                val_loss = validate(model, config, rank, world_size)
                if rank == 0:
                    print(f"Step {step} | Validation Loss: {val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, optimizer, step, val_loss, config, run_id)
            
            
            if rank == 0 and step % config.save_interval == 0:
                save_checkpoint(model, optimizer, step, loss.item(), config, run_id)
            
            step += 1
            
        except StopIteration:
            if rank == 0:
                save_checkpoint(model, optimizer, step, loss.item(), config, run_id)
            
            break
            
    
    if rank == 0:
        val_loss = validate(model, config, rank, world_size)
        print(f"Final Validation Loss: {val_loss:.4f}")
        save_checkpoint(model, optimizer, step, val_loss, config, run_id)
        
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
