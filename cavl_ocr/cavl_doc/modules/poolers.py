# src/cavl_doc/modules/poolers.py
import torch
import torch.nn as nn
from typing import Optional

class AttentionPooling(nn.Module):
    """
    Attention-based pooling that learns a global query vector.
    
    Evolução (Retrocompatível):
    - Se num_queries=1 (padrão): Comporta-se EXATAMENTE como o antigo (mesmos nomes de pesos).
    - Se num_queries>1: Ativa o modo Multi-Query para capturar contextos distintos.
    """
    def __init__(self, hidden_dim: int = 1536, num_heads: int = 8, num_queries: int = 1, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        
        # MHA padrão
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        
        # --- COMPATIBILIDADE DE PESOS ---
        # Mantemos o nome 'self.query' e 'self.ln' para carregar checkpoints antigos sem erro.
        if num_queries == 1:
            # Shape antigo: [1, hidden_dim]
            self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        else:
            # Shape novo para multi-query: [num_queries, 1, hidden_dim]
            self.query = nn.Parameter(torch.randn(num_queries, 1, hidden_dim) * 0.02)
        
        # Mantemos o nome 'self.ln' (LayerNorm de entrada/saída única)
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        
        # Camadas extras apenas para o modo Multi-Query
        if num_queries > 1:
            self.proj = nn.Linear(num_queries * hidden_dim, hidden_dim)
            self.ln_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        else:
            self.proj = None

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if tokens is None:
            raise ValueError("tokens must be provided to AttentionPooling")
        
        b, seq_len, d = tokens.shape
        
        # Garante mesmo dtype (bfloat16/float32)
        target_dtype = self.query.dtype
        if tokens.dtype != target_dtype:
            tokens = tokens.to(dtype=target_dtype)
            
        # Transpose para [Seq_Len, Batch, Dim]
        tokens_t = tokens.transpose(0, 1)
        
        # --- Lógica de Expansão da Query ---
        if self.num_queries == 1:
            # Lógica Antiga: unsqueeze(1) manual
            # q shape: [1, B, Dim]
            q = self.query.unsqueeze(1).expand(1, b, d)
        else:
            # Lógica Nova: expand direto (pois já tem 3 dimensões)
            # q shape: [Num_Queries, B, Dim]
            q = self.query.expand(-1, b, -1)
        
        # Máscara
        if mask is not None:
            key_padding_mask = ~mask if mask.dtype == torch.bool else ~(mask.bool())
        else:
            key_padding_mask = None
            
        # Atenção
        attn_out, _ = self.mha(query=q, key=tokens_t, value=tokens_t, key_padding_mask=key_padding_mask)
        
        # --- Processamento de Saída ---
        if self.num_queries == 1:
            # Caminho Legado (Idêntico ao original)
            # Squeeze dim 0 -> [B, Dim] -> LayerNorm
            return self.ln(attn_out.squeeze(0))
        else:
            # Caminho Multi-Query
            # [Num, B, Dim] -> [B, Num, Dim]
            attn_out = attn_out.transpose(0, 1)
            # Flatten -> [B, Num*Dim]
            flat = attn_out.reshape(b, -1)
            # Projeção e Norm final
            return self.ln_out(self.proj(flat))

class PositionalAttentionPooling(nn.Module):
    """
    Attention-based pooling that adds learnable positional encodings to tokens before attention.
    Useful for tasks where the sequence order/position is critical and might be lost.
    """
    def __init__(self, hidden_dim: int = 1536, num_heads: int = 8, num_queries: int = 1, dropout: float = 0.0, max_len: int = 4096):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        
        # Learnable Positional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)
        
        # MHA
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        
        # Queries
        if num_queries == 1:
            self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        else:
            self.query = nn.Parameter(torch.randn(num_queries, 1, hidden_dim) * 0.02)
        
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        
        if num_queries > 1:
            self.proj = nn.Linear(num_queries * hidden_dim, hidden_dim)
            self.ln_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        else:
            self.proj = None

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if tokens is None:
            raise ValueError("tokens must be provided")
        
        b, seq_len, d = tokens.shape
        
        # Garante mesmo dtype
        target_dtype = self.query.dtype
        if tokens.dtype != target_dtype:
            tokens = tokens.to(dtype=target_dtype)
            
        # --- Add Positional Encoding ---
        # Se a sequência for maior que o max_len, interpolamos o pos_embed
        if seq_len > self.pos_embed.shape[1]:
            # [1, Max, D] -> [1, D, Max] for interpolate
            pe = self.pos_embed.transpose(1, 2)
            pe = torch.nn.functional.interpolate(pe, size=seq_len, mode='linear', align_corners=False)
            pe = pe.transpose(1, 2) # [1, Seq, D]
        else:
            pe = self.pos_embed[:, :seq_len, :]
            
        tokens = tokens + pe.to(dtype=tokens.dtype, device=tokens.device)
        
        # Transpose para [Seq_Len, Batch, Dim] (MHA default é batch_first=False)
        tokens_t = tokens.transpose(0, 1)
        
        # --- Query Expansion ---
        if self.num_queries == 1:
            q = self.query.unsqueeze(1).expand(1, b, d)
        else:
            q = self.query.expand(-1, b, -1)
        
        # Mask
        if mask is not None:
            key_padding_mask = ~mask if mask.dtype == torch.bool else ~(mask.bool())
        else:
            key_padding_mask = None
            
        # Attention
        attn_out, _ = self.mha(query=q, key=tokens_t, value=tokens_t, key_padding_mask=key_padding_mask)
        
        # Output
        if self.num_queries == 1:
            return self.ln(attn_out.squeeze(0))
        else:
            attn_out = attn_out.transpose(0, 1)
            flat = attn_out.reshape(b, -1)
            return self.ln_out(self.proj(flat))

class MeanPooling(nn.Module):
    """Pooling simples (Média)."""
    def __init__(self, hidden_dim: int = 1536, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)
        
    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is None:
            pooled = tokens.mean(dim=1)
        else:
            mask = mask.unsqueeze(-1).float()
            tokens = tokens * mask
            sum_tokens = tokens.sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_tokens / sum_mask
        return self.ln(pooled)

# --- REGISTRO ---
POOLER_REGISTRY = {
    "attention": AttentionPooling,
    "positional_attention": PositionalAttentionPooling,
    "mean": MeanPooling,
}

def build_pooler(pooler_type: str, **kwargs):
    if pooler_type not in POOLER_REGISTRY:
        raise ValueError(f"Pooler '{pooler_type}' não encontrado.")
    
    # Configuração inteligente de num_queries
    valid_args = kwargs.copy()
    if pooler_type == "attention":
        # Se não passado, assume 1 (comportamento padrão)
        # Se quiser testar o "turbo", passe num_queries=4 no script de treino
        valid_args.setdefault('num_queries', 1) 
        
    return POOLER_REGISTRY[pooler_type](**valid_args)