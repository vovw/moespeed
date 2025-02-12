import torch
import torch.nn as nn
import torch.nn.functional as F
from math import inf
import math
import torch.distributed as dist
from transformers import GPT2LMHeadModel
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
#create_block_mask = torch.compile(create_block_mask, dynamic=False)
#RoPE 
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 65536):
        """
        Initialize Rotary Position Embedding
        
        Args:
            dim: Dimension of the embedding (must be divisible by 2)
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError("Dimension must be divisible by 2")
            
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create position indices tensor
        position = torch.arange(max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        
        # Create dimension indices tensor for half the dimension
        # Since we'll rotate half the dimensions, we only need dim/2
        div_term = torch.exp(
            torch.arange(0, dim//2) * -(math.log(10000.0) / (dim//2))
        )
        
        # Compute sin and cos tables for half dimensions
        emb = position * div_term
        self.register_buffer("sin_table", emb.sin().unsqueeze(0))  # [1, max_seq_len, dim//2]
        self.register_buffer("cos_table", emb.cos().unsqueeze(0))  # [1, max_seq_len, dim//2]
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Tensor with positional information encoded
        """
        batch_size, num_heads, seq_len, dim = x.shape
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}")
            
        # Get sin and cos values for current sequence length
        sin = self.sin_table[:, :seq_len, :]  # [1, seq_len, dim//2]
        cos = self.cos_table[:, :seq_len, :]  # [1, seq_len, dim//2]
        
        # Duplicate the sin/cos for the full dimension
        sin = torch.cat([sin, sin], dim=-1)  # [1, seq_len, dim]
        cos = torch.cat([cos, cos], dim=-1)  # [1, seq_len, dim]
        
        # Reshape sin and cos for broadcasting
        sin = sin.unsqueeze(1)  # [1, 1, seq_len, dim]
        cos = cos.unsqueeze(1)  # [1, 1, seq_len, dim]
        
        # Expand to match input shape
        sin = sin.expand(batch_size, num_heads, -1, -1)
        cos = cos.expand(batch_size, num_heads, -1, -1)
        
        # Apply rotation using complex number multiplication:
        # (a + ib)(cos θ + i sin θ) = (a cos θ - b sin θ) + i(a sin θ + b cos θ)
        return (x * cos) + (self._rotate_half(x) * sin)


class voMoE(nn.Module):
    def __init__(self, num_experts, hidden_size,k=2):
        super().__init__()
    
        self.rank=torch.distributed.get_rank()
        self.k=k
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_size, num_experts)
        
        self.world_size=torch.distributed.get_world_size()
        #self.experts = nn.ModuleList([nn.Linear(hidden_size,hidden_size) for _ in range(num_experts)]) #deprecated feature
        experts_per_rank = num_experts // self.world_size
        assert num_experts % self.world_size == 0, "Number of experts must be divisible by world size"
        start_idx = self.rank * experts_per_rank
        end_idx = start_idx + experts_per_rank
        self.local_experts_ids = list(range(start_idx, end_idx))
        self.local_experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(experts_per_rank)])

    def forward(self, x):
        # x: (B, S, H)
        device=torch.cuda.current_device()
        b,s,h=x.shape
        scores = self.router(x) # (B, S, E)
        probs_0 = F.softmax(scores,dim=-1) # (B, S, E)
        p,expert_id=torch.topk(probs_0,self.k,dim=-1) #(B,S,K)
        p=p.unsqueeze(-1) #B,S,K,1
        out=torch.empty((b,s,self.k,h))
        global_out=torch.empty((self.world_size*b,s,self.k,h),device=device,dtype=x.dtype) # each gpu will have it's data parallel we need to grab
        global_expert_id=torch.empty((self.world_size*b,s,self.k),device=device,dtype=expert_id.dtype)
        torch.distributed.all_gather_into_tensor(global_expert_id,expert_id) 
        global_x=torch.empty((self.world_size*b,s,h),device=device,dtype=x.dtype) #gathers x from dp
        torch.distributed.all_gather_into_tensor(global_x,x)
        output_total = torch.zeros((self.world_size*b,s,self.k,h),dtype=x.dtype,device=device)
        for i, expert in enumerate(self.local_experts):
            local_expert_id = self.local_experts_ids[i] #1,2,3,4 | 5,6,7,8 for default case
            B, S, K = torch.where(global_expert_id == local_expert_id)
            out = expert(global_x[B, S, :])
            output_total[B, S, K,:] = out #should only store the first 4 k slices on the zeroth gpu
                
        output_local = torch.empty((b, s,self.k, h), device=device,dtype=output_total.dtype)
        #output_local_list=[torch.empty((b,s,h)) for _ in range(self.world_size)]
        #output_total_list=list(torch.tensor_split(output_total,self.world_size,dim=0))
        torch.distributed.reduce_scatter_tensor(output_local, output_total) 
        p=p/(p.sum(dim=2).unsqueeze(2))
        output=(p*output_local).sum(dim=2)
        return output

    
class ffMoE(nn.Module):
    def __init__(self, num_experts, hidden_size,k=2):
        super().__init__()
    
        
        self.k=k
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_size, num_experts)
        
        self.world_size=torch.distributed.get_world_size()
        self.rank=torch.distributed.get_rank()
        
        #self.experts = nn.ModuleList([FeedForward(hidden_size) for _ in range(num_experts)]) #deprecated feature
        
        experts_per_rank = num_experts // self.world_size
        assert num_experts % self.world_size == 0, "Number of experts must be divisible by world size"
        start_idx = self.rank * experts_per_rank
        end_idx = start_idx + experts_per_rank
        self.local_experts_ids = list(range(start_idx, end_idx))
        self.local_experts = nn.ModuleList([FeedForward(hidden_size) for _ in range(experts_per_rank)])


    def load_balance(self,p):
        avgprobs=p.mean(dim=[0,1]) #returns 1D vector shape: (E,), differentiable
        #need to also return top1 selections
        b,s,e=p.shape
        tops=torch.argmax(p,dim=-1).squeeze(0) #1D, non negative input
        counts=torch.bincount(tops,minlength=self.num_experts) #shape (E,), not differentiable
        freqs=(counts/(b*s)).to(torch.float)
        return 0.001*torch.dot(freqs,avgprobs) #small load balancing
        
            

    def forward(self, x):
        # x: (B, S, H)
        device=torch.cuda.current_device()
        b,s,h=x.shape
        scores = self.router(x) # (B, S, E)
        probs_0 = F.softmax(scores,dim=-1) # (B, S, E)
        p,expert_id=torch.topk(probs_0,self.k,dim=-1) #(B,S,K)
        bal_loss=self.load_balance(probs_0)
        p=p.unsqueeze(-1) #B,S,K,1
        #out=torch.empty((b,s,self.k,h))
        #global_out=torch.empty((self.world_size*b,s,self.k,h),device=device,dtype=x.dtype) # each gpu will have it's data parallel we need to grab
        global_expert_id=torch.empty((self.world_size*b,s,self.k),device=device,dtype=expert_id.dtype)
        torch.distributed.all_gather_into_tensor(global_expert_id,expert_id) 
        global_x=torch.empty((self.world_size*b,s,h),device=device,dtype=x.dtype) #gathers x from dp
        torch.distributed.all_gather_into_tensor(global_x,x)
        output_total = torch.zeros((self.world_size*b,s,self.k,h),dtype=x.dtype,device=device)
        for i, expert in enumerate(self.local_experts):
            local_expert_id = self.local_experts_ids[i] #1,2,3,4 | 5,6,7,8 for default case
            B, S, K = torch.where(global_expert_id == local_expert_id)
            out = expert(global_x[B, S, :])
            output_total[B, S, K,:] = out #should only store the first 4 k slices on the zeroth gpu
                
        output_local = torch.empty((b, s,self.k, h), device=device,dtype=output_total.dtype)
        #output_local_list=[torch.empty((b,s,h)) for _ in range(self.world_size)]
        #output_total_list=list(torch.tensor_split(output_total,self.world_size,dim=0))
        torch.distributed.reduce_scatter_tensor(output_local, output_total) 
        p=p/(p.sum(dim=2).unsqueeze(2))
        output=(p*output_local).sum(dim=2)
        return output,bal_loss




#we do NOT use long-short sliding windows or hybridization on this version
class moegpt(nn.Module):
    def __init__(self, hidden_dim, num_blocks,vocab=50257):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        #embedding from tokens
        self.embd=nn.Embedding(vocab,hidden_dim)
        #self.initial_proj = nn.Linear(in_channels, hidden_dim)  
        
        
        self.blocks = nn.ModuleList([
            Block(hidden_dim) for _ in range(num_blocks)
        ])
        #unique_blocks = [Block(hidden_dim) for _ in range(num_blocks//2)] #repeating twice
        #Univeral Transformer Parameter Sharing
        #self.blocks = nn.ModuleList(unique_blocks*2)
        
        
        
        self.out_proj = nn.Linear(hidden_dim, vocab)  
        
        
     
    def forward(self, x):
        #x is already tokenized in shape B,S,1 (squeezed), B=1 from dataloader due to flattening for flex attention
        tokens=x
        x=self.embd(x).unsqueeze(0)
        #layer norm before blocks, empirical improvement
        x=F.rms_norm(x, (x.size(-1),))
        bl=0
        for block in self.blocks:
            x,bal_loss = block(x,tokens)
            bl=bl+bal_loss
        #layer norm after blocks, standard
        x=F.rms_norm(x, (x.size(-1),))
        logits=self.out_proj(x)
        
        return logits,bl #b,h*w,vocab

            
            
    
class Block(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        #self.attention =SHAttention(8,hidden_dim,num_heads=8)
        self.attention=MultiLatentAttention(hidden_dim)
        #self.ff = ffMoE(8,hidden_dim) #num experts, hidden_size
        self.ff=ffMoE(8,hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim) #perhaps rms norm is better here, llama uses rms_norm
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x,tokens):
        x = x + self.attention(self.norm1(x),tokens)
        ff,bal_loss=self.ff(self.norm2(x))
        x = x + ff
        return x, bal_loss



class FeedForward(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4):
        super().__init__()
        #self.net = nn.Sequential(
            #nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            #nn.GELU(), 
            #nn.Linear(hidden_dim * expansion_factor, hidden_dim)
        #)
        self.fc1=nn.Linear(hidden_dim,hidden_dim*expansion_factor)
        self.fc2=nn.Linear(hidden_dim*expansion_factor,hidden_dim)
        
    def forward(self, x):
        x=self.fc1(x)
        x=F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU, from nanogpt experiments
        out=self.fc2(x) #potentially layer norm before out is worth experimenting as per gemma                   
        return out



#flex attention
#might add moe for keys and output
#theoretically because the qkv is low rank computed, it might be beneficial to use moes for each linear layer
#for flex attention compiled, we need even lower rank for q and k
class MultiLatentAttention(nn.Module):
    def __init__(self,hidden_dim,num_heads=12,low_rank=2,block_size=128,max_seq_len=1024):
        super().__init__()
        self.num_heads=num_heads
        self.head_dim=hidden_dim//num_heads
        self.block_size=block_size
        self.max_seq_len=max_seq_len
        #assert hidden_dim//num_heads
        #downproj for q
        self.qd_proj=nn.Linear(hidden_dim,hidden_dim//low_rank) 
        self.qu_proj=nn.Linear(hidden_dim//low_rank,hidden_dim)
        #self.qr_proj=nn.Linear(hidden_dim,self.head_dim) #original
        self.qr_proj=nn.Linear(hidden_dim,self.head_dim)
        #shared downproj for k,v
        self.kvd=nn.Linear(hidden_dim,hidden_dim//low_rank)
        self.v_up_proj=nn.Linear(hidden_dim//low_rank,hidden_dim*2)
        self.k_up_proj=nn.Linear(hidden_dim//low_rank,hidden_dim)
        #self.kr_proj=nn.Linear(hidden_dim,self.head_dim) #original
        self.kr_proj=nn.Linear(hidden_dim,self.head_dim)
        #output proj
        self.o_proj = nn.Linear(hidden_dim*2, hidden_dim)
        #self.rope=RotaryPositionEmbedding(self.head_dim) #orignal
        self.rope=RotaryPositionEmbedding(self.head_dim)
        self.scale = (2*self.head_dim) ** -0.5 #original was 2, now we are doing larger attention at 3/2
        #self.scale=(self.head_dim)**-0.5
        
        self.world_size=torch.distributed.get_world_size()
        
    

    def forward(self, x,token_seq):
        #layer norm prior to input
        B, N,dim = x.shape
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        
        # Create block masks with document boundary handling
        #block_mask = self.create_block_masks(input_seq=token_seq, sliding_window_num_blocks=None)
        
        # query projections
        qd=self.qd_proj(x) #B,N,low_rank_dim
        qr=self.qr_proj(x).unsqueeze(2)# B,N,1,head_dim/2
        qr=qr.expand(-1,-1,self.num_heads,-1).permute(0,2,1,3) #B,num_heads,seq_len,head_dim//2
        qr=self.rope(qr)
        q=self.qu_proj(qd) #B,N,dim
        q=q.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        q=torch.cat((q,qr),dim=-1) #B,num_heads,seq_len,head_dim
        
        
        #keys
        low_rank_kv=self.kvd(x) #B,S,compressed_dim
        k=self.k_up_proj(low_rank_kv)
        kr=self.kr_proj(x).unsqueeze(2)
        kr=kr.expand(-1,-1,self.num_heads,-1).permute(0,2,1,3)
        kr=self.rope(kr)
        k= k.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        k=torch.cat((k,kr),dim=-1) #B,num_heads,seq_len,head_dim
        
        #values
        ### the point of doing low rank is not just parameter count reduction, but also kv cache size reduction
        v=self.v_up_proj(low_rank_kv) 
        v=v.reshape(B,N,self.num_heads,(self.head_dim*2)).permute(0,2,1,3)

        
        docs = (token_seq == 50256).cumsum(0)
        def document_causal_mask(b, h, q_idx, kv_idx):
          causal_mask = q_idx >= kv_idx
          document_mask = docs[q_idx] == docs[kv_idx]
          window_mask = q_idx - kv_idx < 1024
          return causal_mask & document_mask & window_mask

        S = len(token_seq)
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)

        #q=F.rms_norm(q, (q.size(-1),))
        #k=F.rms_norm(k, (k.size(-1),))
        x = flex_attention(q, k, v, block_mask=block_mask, scale=self.scale)
        x = x.transpose(1, 2).reshape(B, N, -1)

        x=self.o_proj(x)
        return x
