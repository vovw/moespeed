# EXPERIMENTAL MIXTURE OF EXPERTS IMPLEMENTATION
# WARNING: THIS IS RESEARCH CODE, EXPECT THINGS TO BREAK
# Based on DeepSeek-V2/V3 MLA + distributed expert parallelism 
# Rishi's findings: attention MoEs suck, FFN MoEs are where it's at
# Lots of hacky optimizations and questionable design choices below...

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, sqrt, pi
import math
import torch.distributed as dist
from transformers import GPT2LMHeadModel  # not used but whatever
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# TODO: make this work without dynamic=False? compilation is slow af
# create_block_mask = torch.compile(create_block_mask, dynamic=False)

# ROTARY POSITION EMBEDDING - THE GOAT OF POSITIONAL ENCODINGS
# Su et al. 2021: RoFormer - it just works, no one knows why exactly
# but lower rank shared RoPE is magic according to deepseek
class RotaryPositionEmbedding(nn.Module):
    """
    RoPE implementation - rotate query/key vectors in complex plane
    Mathematical beauty: encoding absolute position with relative bias
    
    Key insight from deepseek: you can share this across heads (!!!)
    Saves a ton of parameters and cache during inference
    """
    def __init__(self, dim: int, max_seq_len: int = 65536):
        super().__init__()
        
        assert dim % 2 == 0, "RoPE dimension must be even (complex number pairs)"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # position encoding computation - this is the core magic
        # theta_i = 10000^(-2i/d) where i ∈ [0, d/2)
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(log(10000.0) / dim)
        )
        
        # precompute sin/cos tables for efficiency
        # these are the rotation matrices in complex form
        emb = position * div_term
        self.register_buffer("sin_table", emb.sin().unsqueeze(0))  # [1, seq_len, dim//2]
        self.register_buffer("cos_table", emb.cos().unsqueeze(0))  # [1, seq_len, dim//2]
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate the second half of the last dimension
        This implements the complex multiplication: 
        (a + ib) * (cos θ + i sin θ) = (a cos θ - b sin θ) + i(a sin θ + b cos θ)
        """
        x1 = x[..., : x.shape[-1] // 2]  # real part
        x2 = x[..., x.shape[-1] // 2 :]  # imaginary part  
        return torch.cat((-x2, x1), dim=-1)  # complex rotation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding"""
        batch_size, num_heads, seq_len, dim = x.shape
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence too long: {seq_len} > {self.max_seq_len}")
            
        # get sin/cos for current sequence
        sin = self.sin_table[:, :seq_len, :]  # [1, seq_len, dim//2]
        cos = self.cos_table[:, :seq_len, :]  # [1, seq_len, dim//2]
        
        # expand to full dimension (duplicate for real/imag parts)
        sin = torch.cat([sin, sin], dim=-1)  # [1, seq_len, dim]
        cos = torch.cat([cos, cos], dim=-1)  # [1, seq_len, dim]
        
        # broadcast to match input shape
        sin = sin.unsqueeze(1).expand(batch_size, num_heads, -1, -1)
        cos = cos.unsqueeze(1).expand(batch_size, num_heads, -1, -1)
        
        # apply rotation: x' = x*cos + rotate_half(x)*sin
        return (x * cos) + (self._rotate_half(x) * sin)


# VALUE/OUTPUT MOE - EXPERIMENTAL, PROBABLY DOESN'T WORK WELL
# Rishi's insight: "even with load-balancing the model is unable to learn effectively"
# "It saturates quite early, within 2B tokens"
# but keeping it here for science...
class voMoE(nn.Module):
    """
    Value/Output Mixture of Experts
    
    Research finding: Linear MoEs in attention provide minimal benefit
    This is probably why Switch-Head paper didn't take off
    
    Expert distribution: each GPU gets num_experts//world_size experts
    Communication pattern: all-gather -> compute -> reduce-scatter
    """
    def __init__(self, num_experts=8, hidden_size=768, k=2):
        super().__init__()
        
        # distributed setup - this better be initialized already
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        self.k = k  # top-k routing
        self.num_experts = num_experts
        
        # gating network - learns which experts to use
        # simple linear layer but could try more complex routing
        self.router = nn.Linear(hidden_size, num_experts)
        
        # distributed expert assignment
        # each GPU gets a subset of experts to avoid memory explosion
        experts_per_rank = num_experts // self.world_size
        assert num_experts % self.world_size == 0, f"num_experts={num_experts} not divisible by world_size={self.world_size}"
        
        start_idx = self.rank * experts_per_rank
        end_idx = start_idx + experts_per_rank
        self.local_experts_ids = list(range(start_idx, end_idx))
        
        # local experts - just linear layers for now
        # could try more complex expert architectures
        self.local_experts = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(experts_per_rank)
        ])
        
        print(f"[voMoE] Rank {self.rank}: managing experts {self.local_experts_ids}")

    def forward(self, x):
        """Forward pass with distributed expert computation"""
        device = torch.cuda.current_device()
        b, s, h = x.shape
        
        # route to experts
        scores = self.router(x)  # (B, S, E)
        probs = F.softmax(scores, dim=-1)  # (B, S, E)
        top_probs, expert_ids = torch.topk(probs, self.k, dim=-1)  # (B, S, K)
        
        # reshape for expert computation
        top_probs = top_probs.unsqueeze(-1)  # (B, S, K, 1)
        
        # distributed computation setup
        # all_gather inputs and expert assignments across GPUs
        global_expert_ids = torch.empty(
            (self.world_size * b, s, self.k), 
            device=device, dtype=expert_ids.dtype
        )
        torch.distributed.all_gather_into_tensor(global_expert_ids, expert_ids)
        
        global_x = torch.empty(
            (self.world_size * b, s, h), 
            device=device, dtype=x.dtype
        )
        torch.distributed.all_gather_into_tensor(global_x, x)
        
        # expert computation
        output_total = torch.zeros(
            (self.world_size * b, s, self.k, h), 
            dtype=x.dtype, device=device
        )
        
        # each rank computes for its assigned experts
        for i, expert in enumerate(self.local_experts):
            local_expert_id = self.local_experts_ids[i]
            # find tokens routed to this expert
            B, S, K = torch.where(global_expert_ids == local_expert_id)
            if len(B) > 0:  # only compute if tokens are routed here
                expert_output = expert(global_x[B, S, :])
                output_total[B, S, K, :] = expert_output
        
        # gather results back to each rank
        output_local = torch.empty(
            (b, s, self.k, h), device=device, dtype=output_total.dtype
        )
        torch.distributed.reduce_scatter_tensor(output_local, output_total)
        
        # weighted combination of expert outputs
        # normalize probabilities to sum to 1 across selected experts
        top_probs = top_probs / (top_probs.sum(dim=2, keepdim=True) + 1e-9)
        output = (top_probs * output_local).sum(dim=2)
        
        return output


# FEEDFORWARD MOE - THE REAL DEAL
# Research shows this is where you want to add expressivity
# "If our only option is to modify the output of attention, and the MLPs come right 
# after, we might as well just add the MoEs and gain back the expressivity there"
class ffMoE(nn.Module):
    """
    FeedForward Mixture of Experts
    
    Key insight: expressivity bottleneck is in FFN, not attention projections
    Load balancing is critical to prevent expert collapse
    
    Sample efficiency issue: needs 10B+ tokens to work well
    """
    def __init__(self, num_experts=8, hidden_size=768, k=2):
        super().__init__()
        
        self.k = k
        self.num_experts = num_experts
        
        # distributed setup
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # routing network
        self.router = nn.Linear(hidden_size, num_experts)
        
        # expert distribution across GPUs
        experts_per_rank = num_experts // self.world_size
        assert num_experts % self.world_size == 0, "experts must be divisible by world size"
        
        start_idx = self.rank * experts_per_rank  
        end_idx = start_idx + experts_per_rank
        self.local_experts_ids = list(range(start_idx, end_idx))
        
        # expert networks - using FeedForward modules
        self.local_experts = nn.ModuleList([
            FeedForward(hidden_size) for _ in range(experts_per_rank)
        ])
        
        print(f"[ffMoE] Rank {self.rank}: experts {self.local_experts_ids}")

    def load_balance_loss(self, probs):
        """
        Load balancing loss to prevent expert collapse
        
        Auxiliary loss from Switch Transformer paper
        Encourages uniform distribution of tokens across experts
        
        Returns:
            Small auxiliary loss (0.001 * actual_loss)
        """
        # average probability of selecting each expert across batch
        avg_probs = probs.mean(dim=[0, 1])  # (E,)
        
        # frequency of top-1 selections (non-differentiable)
        b, s, e = probs.shape
        top1_indices = torch.argmax(probs, dim=-1)  # (B, S)
        expert_counts = torch.bincount(
            top1_indices.flatten(), minlength=self.num_experts
        ).float()
        expert_freqs = expert_counts / (b * s)
        
        # load balancing loss: dot product of frequencies and probabilities
        # encourages uniform usage
        return 0.001 * torch.dot(expert_freqs, avg_probs)

    def forward(self, x):
        """Forward pass with load balancing"""
        device = torch.cuda.current_device()
        b, s, h = x.shape
        
        # expert routing
        scores = self.router(x)  # (B, S, E)
        probs = F.softmax(scores, dim=-1)  # (B, S, E) 
        top_probs, expert_ids = torch.topk(probs, self.k, dim=-1)  # (B, S, K)
        
        # compute load balancing loss
        bal_loss = self.load_balance_loss(probs)
        
        top_probs = top_probs.unsqueeze(-1)  # (B, S, K, 1)
        
        # distributed computation (same pattern as voMoE)
        global_expert_ids = torch.empty(
            (self.world_size * b, s, self.k), 
            device=device, dtype=expert_ids.dtype
        )
        torch.distributed.all_gather_into_tensor(global_expert_ids, expert_ids)
        
        global_x = torch.empty(
            (self.world_size * b, s, h), 
            device=device, dtype=x.dtype
        )
        torch.distributed.all_gather_into_tensor(global_x, x)
        
        output_total = torch.zeros(
            (self.world_size * b, s, self.k, h), 
            dtype=x.dtype, device=device
        )
        
        # expert computation
        for i, expert in enumerate(self.local_experts):
            local_expert_id = self.local_experts_ids[i]
            B, S, K = torch.where(global_expert_ids == local_expert_id)
            if len(B) > 0:
                expert_output = expert(global_x[B, S, :])
                output_total[B, S, K, :] = expert_output
        
        # scatter results back
        output_local = torch.empty(
            (b, s, self.k, h), device=device, dtype=output_total.dtype
        )
        torch.distributed.reduce_scatter_tensor(output_local, output_total)
        
        # weighted combination
        top_probs = top_probs / (top_probs.sum(dim=2, keepdim=True) + 1e-9)
        output = (top_probs * output_local).sum(dim=2)
        
        return output, bal_loss


# MAIN MODEL CLASS - MOE GPT
# we do NOT use long-short sliding windows or hybridization on this version
# keeping it simple for now, can add complexity later
class moegpt(nn.Module):
    """
    MoE GPT model with Multi-Latent Attention
    
    Architecture choices:
    - RMSNorm instead of LayerNorm (following LLaMA)
    - Pre-norm instead of post-norm
    - MLA for attention efficiency
    - FFN MoE for expressivity
    """
    def __init__(self, hidden_dim=768, num_blocks=12, vocab=50257):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # token embeddings
        self.embd = nn.Embedding(vocab, hidden_dim)
        
        # transformer blocks
        self.blocks = nn.ModuleList([
            Block(hidden_dim) for _ in range(num_blocks)
        ])
        
        # experiment with parameter sharing (Universal Transformer style)
        # unique_blocks = [Block(hidden_dim) for _ in range(num_blocks//2)]
        # self.blocks = nn.ModuleList(unique_blocks * 2)
        
        # output projection to vocab
        self.out_proj = nn.Linear(hidden_dim, vocab)
        
        print(f"[moegpt] Model created: {sum(p.numel() for p in self.parameters())/1e6:.1f}M parameters")

    def forward(self, x):
        """
        Forward pass
        x: token ids (S,) - already flattened for FlexAttention
        """
        tokens = x  # keep copy for attention masks
        x = self.embd(x).unsqueeze(0)  # (1, S, H) - batch size 1 for FlexAttention
        
        # pre-norm before blocks (empirical improvement)
        x = F.rms_norm(x, (x.size(-1),))
        
        # accumulate balance losses from MoE layers
        total_bal_loss = 0
        for block in self.blocks:
            x, bal_loss = block(x, tokens)
            total_bal_loss = total_bal_loss + bal_loss
            
        # post-norm (standard)
        x = F.rms_norm(x, (x.size(-1),))
        
        # output logits
        logits = self.out_proj(x)
        
        return logits, total_bal_loss


# TRANSFORMER BLOCK
# standard pre-norm architecture with MLA + ffMoE
class Block(nn.Module):
    """Transformer block with MLA attention and ffMoE"""
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Multi-Latent Attention
        self.attention = MultiLatentAttention(hidden_dim)
        
        # could also try value/output MoE in attention
        # self.attention = SomeAttentionWithvoMoE(hidden_dim)
        
        # FeedForward MoE - this is where the magic happens
        self.ff = ffMoE(8, hidden_dim)  # 8 experts
        
        # normalization layers
        # TODO: try RMSNorm instead of LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, tokens):
        """Forward with residual connections"""
        # attention with residual
        x = x + self.attention(self.norm1(x), tokens)
        
        # ffn with residual  
        ff_out, bal_loss = self.ff(self.norm2(x))
        x = x + ff_out
        
        return x, bal_loss


# FEEDFORWARD NETWORK  
# using ReLU² activation - Primer paper shows 1-2% improvement over GELU
class FeedForward(nn.Module):
    """
    Standard FFN with ReLU² activation
    
    ReLU²(x) = ReLU(x)² 
    Paper: https://arxiv.org/abs/2109.08668v2
    NanoGPT experiments confirm ~1-2% better than GELU at scale
    """
    def __init__(self, hidden_dim, expansion_factor=4):
        super().__init__()
        
        intermediate_dim = hidden_dim * expansion_factor
        
        # split into separate layers for clarity
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        
        # could add layer norm before output (Gemma style)
        # self.pre_out_norm = nn.LayerNorm(intermediate_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x).square()  # ReLU² activation
        # x = self.pre_out_norm(x)  # optional
        x = self.fc2(x)
        return x


# MULTI-LATENT ATTENTION 
# MULTI-LATENT ATTENTION 
# MULTI-LATENT ATTENTION 
# MULTI-LATENT ATTENTION 
# Based on DeepSeek-V2/V3 architecture 
# Key :
# 1. Low-rank Q/K/V projections (parameter efficiency)
# 2. Shared RoPE across heads (cache efficiency) 
# 3. Joint KV embedding (massive inference speedup)
class MultiLatentAttention(nn.Module):
    """
    Multi-Latent Attention from DeepSeek-V2/V3
    
    Key insight: low-rank projections + shared positional encoding works!
    
    Architecture:
    - Query: down-proj -> up-proj + shared RoPE
    - Key: shared down-proj -> up-proj + shared RoPE  
    - Value: shared down-proj -> up-proj (larger dim)
    
    Potential MoE integration points:
    - Could add MoE to value/output projections
    - Theoretically QKV low-rank projections could benefit from MoE
    - But research shows attention MoEs don't work well
    """
    def __init__(self, hidden_dim=768, num_heads=12, low_rank=2, block_size=128, max_seq_len=1024):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        
        # sanity check
        assert hidden_dim % num_heads == 0, f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}"
        
        # Query projections: down -> up + RoPE
        self.qd_proj = nn.Linear(hidden_dim, hidden_dim // low_rank)  # down-projection
        self.qu_proj = nn.Linear(hidden_dim // low_rank, hidden_dim)  # up-projection
        self.qr_proj = nn.Linear(hidden_dim, self.head_dim)  # RoPE projection
        
        # Shared KV projections (DeepSeek innovation)
        self.kvd = nn.Linear(hidden_dim, hidden_dim // low_rank)  # shared down-proj
        self.k_up_proj = nn.Linear(hidden_dim // low_rank, hidden_dim)
        self.v_up_proj = nn.Linear(hidden_dim // low_rank, hidden_dim * 2)  # larger for FlexAttention
        self.kr_proj = nn.Linear(hidden_dim, self.head_dim)  # shared RoPE projection
        
        # output projection
        self.o_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # shared RoPE embedding (parameter sharing magic)
        self.rope = RotaryPositionEmbedding(self.head_dim)
        
        # attention scaling
        # using larger scale since we concatenate RoPE dims
        self.scale = (2 * self.head_dim) ** -0.5
        
        # distributed info
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        print(f"[MLA] heads={num_heads}, head_dim={self.head_dim}, low_rank={low_rank}")

    def forward(self, x, token_seq):
        """
        Multi-Latent Attention forward pass
        
        x: (B, S, H) input features
        token_seq: (S,) token sequence for mask creation
        """
        B, N, dim = x.shape
        assert B == 1, "FlexAttention requires batch_size=1"
        
        # === Query computation ===
        # low-rank query path
        qd = self.qd_proj(x)  # (B, N, H//low_rank)
        q = self.qu_proj(qd)  # (B, N, H)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, nh, N, hd)
        
        # shared RoPE query path  
        qr = self.qr_proj(x).unsqueeze(2)  # (B, N, 1, hd)
        qr = qr.expand(-1, -1, self.num_heads, -1).permute(0, 2, 1, 3)  # (B, nh, N, hd)
        qr = self.rope(qr)  # apply rotary embedding
        
        # concatenate query components
        q = torch.cat((q, qr), dim=-1)  # (B, nh, N, 2*hd)
        
        # === Key computation ===
        # shared low-rank KV embedding
        low_rank_kv = self.kvd(x)  # (B, N, H//low_rank)
        
        # key path
        k = self.k_up_proj(low_rank_kv)  # (B, N, H)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # shared RoPE key path
        kr = self.kr_proj(x).unsqueeze(2)
        kr = kr.expand(-1, -1, self.num_heads, -1).permute(0, 2, 1, 3)
        kr = self.rope(kr)
        
        # concatenate key components  
        k = torch.cat((k, kr), dim=-1)  # (B, nh, N, 2*hd)
        
        # === Value computation ===
        # larger value projection for better expressivity
        v = self.v_up_proj(low_rank_kv)  # (B, N, 2*H)
        v = v.reshape(B, N, self.num_heads, self.head_dim * 2).permute(0, 2, 1, 3)
        
        # === FlexAttention with block masks ===
        # document-aware causal attention with sliding window
        docs = (token_seq == 50256).cumsum(0)  # document boundaries (GPT-2 <|endoftext|>)
        
        def document_causal_mask(b, h, q_idx, kv_idx):
            """Custom attention mask function"""
            causal_mask = q_idx >= kv_idx  # causal constraint
            document_mask = docs[q_idx] == docs[kv_idx]  # same document
            window_mask = q_idx - kv_idx < 1024  # sliding window
            return causal_mask & document_mask & window_mask
        
        S = len(token_seq)
        block_mask = create_block_mask(
            document_causal_mask, None, None, S, S, 
            device="cuda", _compile=True
        )
        
        # optional: QK normalization (helps stability)
        # q = F.rms_norm(q, (q.size(-1),))
        # k = F.rms_norm(k, (k.size(-1),))
        
        # FlexAttention computation
        attn_out = flex_attention(q, k, v, block_mask=block_mask, scale=self.scale)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, -1)  # (B, N, 2*H)
        
        # output projection
        output = self.o_proj(attn_out)
        
        return output


# TODO: experiment with different expert architectures
# TODO: try mixture of attention patterns (dense + sparse)
# TODO: implement gradient accumulation for large batch sizes
# TODO: add checkpointing for memory efficiency
# TODO: try different load balancing strategies
# TODO: experiment with expert capacity buffers
# TODO: implement dynamic expert assignment

if __name__ == "__main__":
    # quick test
    model = moegpt(hidden_dim=768, num_blocks=12)
    x = torch.randint(0, 50257, (1024,))
    logits, loss = model(x)
    print(f"Output shape: {logits.shape}, Balance loss: {loss.item():.6f}")
