# src/cavl_doc/modules/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """
    Cabeça padrão (SimCLR Style).
    Útil para aprendizado contrastivo puro, mas pode distorcer geometria.
    """
    def __init__(self, input_dim: int = 1536, proj_hidden: int = 4096, proj_out: int = 512, use_norm: bool = True, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim, eps=1e-6)
        self.fc1 = nn.Linear(input_dim, proj_hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(proj_hidden, proj_out, bias=True)
        self.use_norm = use_norm

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        if self.fc1.bias is not None: nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None: nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor):
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.use_norm:
            x = F.normalize(x, p=2, dim=-1)
        return x

class MPProjectionHead(nn.Module):
    """Legacy simple head."""
    def __init__(self, input_dim: int, proj_out: int = 512, proj_hidden: int = 2048, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, proj_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(proj_hidden, proj_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))

# ==========================================
# 3. RESIDUAL PROJECTION HEAD (Recomendada)
# ==========================================
class ResidualProjectionHead(nn.Module):
    """
    Cabeça Residual para preservar geometria.
    Estrutura: Input -> [Linear->SiLU->Linear] + Shortcut -> Norm
    Ideal para Metric Learning (ArcFace/CosFace).
    """
    def __init__(self, input_dim: int = 1536, proj_hidden: int = 4096, proj_out: int = 512, dropout: float = 0.1, **kwargs):
        super().__init__()
        
        self.ln_in = nn.LayerNorm(input_dim, eps=1e-6)
        
        # Bloco de Transformação
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_hidden),
            nn.SiLU(),  # Swish é moderno e suave
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, proj_out)
        )
        
        # Atalho (Shortcut) para lidar com mudança de dimensão
        if input_dim != proj_out:
            self.shortcut = nn.Linear(input_dim, proj_out, bias=False)
        else:
            self.shortcut = nn.Identity()
            
        self.ln_out = nn.LayerNorm(proj_out, eps=1e-6)
        
        self._init_weights()

    def _init_weights(self):
        # Inicialização Kaiming para SiLU/ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # x: [B, input_dim]
        normalized_x = self.ln_in(x)
        
        # Caminho principal
        features = self.net(normalized_x)
        
        # Caminho residual (Shortcut projeta x se as dimensões mudarem)
        residual = self.shortcut(normalized_x)
        
        # Soma e Normaliza final (Crucial para ArcFace)
        out = features + residual
        
        # LayerNorm antes da projeção na esfera ajuda na estabilidade do ArcFace
        out = self.ln_out(out)
        
        return F.normalize(out, p=2, dim=-1)

# --- REGISTRO ATUALIZADO ---
HEAD_REGISTRY = {
    "mlp": ProjectionHead,
    "simple_mlp": MPProjectionHead,
    "residual": ResidualProjectionHead  # <--- NOVA OPÇÃO
}

def build_head(head_type: str, **kwargs):
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Head '{head_type}' não encontrado. Opções: {list(HEAD_REGISTRY.keys())}")
    return HEAD_REGISTRY[head_type](**kwargs)