# src/cavl_doc/models/modeling_cavl.py
from typing import Any, Callable, Optional, Tuple, List
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoTokenizer

# Imports internos
from cavl_doc.models.configuration_cavl import CaVLConfig
# Importa os BUILDERS (fábricas), não as classes diretas
from cavl_doc.modules.poolers import build_pooler
from cavl_doc.modules.heads import build_head

class CaVLModel(PreTrainedModel):
    config_class = CaVLConfig
    base_model_prefix = "backbone" 

    def __init__(self, 
                 backbone_or_config: Any, 
                 # Argumentos opcionais para modo legado (treino)
                 cut_layer: int = 27,
                 hidden_dim: int = 1536,
                 proj_hidden: int = 4096,
                 proj_out: int = 512,
                 num_pool_heads: int = 8,
                 num_queries: int = 1, # Novo
                 pooler_type: str = "attention", # Novo
                 head_type: str = "mlp", # Novo
                 encode_fn: Optional[Callable] = None,
                 head: Optional[nn.Module] = None, # Injeção direta (legado)
                 pooler: Optional[nn.Module] = None, # Injeção direta (legado)
                 tokenizer: Any = None,
                 prompt: str = "<image> Analyze this document"):
        
        # --- Lógica Híbrida de Inicialização ---
        if isinstance(backbone_or_config, CaVLConfig):
            # MODO 1: Inicialização via Config (from_pretrained)
            config = backbone_or_config
            super().__init__(config)
            
            # Carrega o backbone do HF automaticamente
            self.backbone = AutoModel.from_pretrained(
                config.backbone_name, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            # Sobrescreve parâmetros com os da config
            self.cut_layer = config.cut_layer
            self.hidden_dim = config.hidden_dim
            # Configs não usadas na inferência direta
            self.encode_fn = None 
            self.tokenizer = None
            
            # Parâmetros modulares da config
            self.num_queries = getattr(config, 'num_queries', 1)
            self.pooler_type = getattr(config, 'pooler_type', 'attention')
            self.head_type = getattr(config, 'head_type', 'mlp')
            
        else:
            # MODO 2: Inicialização Manual (Seu Treino Atual)
            # Cria uma config on-the-fly para manter o PreTrainedModel feliz
            config = CaVLConfig(
                cut_layer=cut_layer,
                hidden_dim=hidden_dim,
                proj_hidden=proj_hidden,
                proj_out=proj_out,
                num_pool_heads=num_pool_heads,
                num_queries=num_queries,
                pooler_type=pooler_type,
                head_type=head_type
            )
            super().__init__(config)
            self.backbone = backbone_or_config # Aqui é o objeto backbone passado
            self.cut_layer = cut_layer
            self.encode_fn = encode_fn
            self.tokenizer = tokenizer 
            self.num_queries = num_queries
            self.pooler_type = pooler_type
            self.head_type = head_type

        self.prompt = prompt

        # --- Configurações Comuns ---
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()
        else:
            def mk_grad(m, i, o): o.requires_grad_(True)
            self.backbone.get_input_embeddings().register_forward_hook(mk_grad)

        try:
            if hasattr(self.backbone.language_model, "lm_head"): self.backbone.language_model.lm_head = nn.Identity()
            if hasattr(self.backbone.language_model.model, "norm"): self.backbone.language_model.model.norm = nn.Identity()
        except: pass

        # --- MONTAGEM MODULAR DOS COMPONENTES ---
        
        # 1. Pooler
        if pooler is not None:
            self.pool = pooler
        else:
            # Usa o Builder com o tipo escolhido
            self.pool = build_pooler(
                self.pooler_type, 
                hidden_dim=config.hidden_dim, 
                num_heads=config.num_pool_heads,
                num_queries=self.num_queries
            )

        # 2. Head
        if head is not None:
            self.head = head
        else:
            # Usa o Builder com o tipo escolhido
            self.head = build_head(
                self.head_type,
                input_dim=config.hidden_dim,
                proj_hidden=config.proj_hidden,
                proj_out=config.proj_out
            )

        self.freeze_all_backbone()

    # --- Utilitários ---
    def freeze_all_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def set_default_trainable(self):
        self.freeze_all_backbone()
        cut = self.config.cut_layer 
        keys = [f"layers.{cut}.self_attn", f"layers.{cut}.mlp", f"layers.{cut}.input_layernorm", f"layers.{cut}.post_attention_layernorm"]
        for n, p in self.backbone.named_parameters():
            for k in keys:
                if k in n and p.dtype.is_floating_point:
                    p.requires_grad = True
                    break

    def trainable_summary(self):
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {tot:,} | Trainable: {tr:,} ({100*tr/tot:.2f}%)")
        return tot, tr

    # --- Forward Logic ---
    def _extract_tokens_via_encode_fn(self, images, device=None, **encode_kwargs):
        assert callable(self.encode_fn), "encode_fn not provided in legacy mode."
        out = self.encode_fn(self.backbone, images, cut_layer=self.cut_layer, **encode_kwargs)
        return (out[0], out[1]) if isinstance(out, tuple) else (out, None)

    def _extract_tokens_via_hidden_states(self, input_ids=None, attention_mask=None, device=None, **kwargs):
        lm = self.backbone.language_model.model
        call_args = dict(output_hidden_states=True, return_dict=True)
        if input_ids is not None: call_args['input_ids'] = input_ids.to(next(self.parameters()).device)
        if attention_mask is not None: call_args['attention_mask'] = attention_mask.to(next(self.parameters()).device)
        call_args.update(kwargs)
        out = lm(**call_args)
        hs = out.hidden_states
        idx = self.config.cut_layer + 1 if len(hs) == (len(lm.layers) + 1) else self.config.cut_layer
        return hs[idx], None

    def forward(self, images=None, input_ids=None, attention_mask=None, device=None, encode_kwargs=None, image_a=None, image_b=None):
        device = device or (next(self.parameters()).device)
        
        if image_a is not None and image_b is not None:
            za = self.forward(images=image_a, device=device)
            zb = self.forward(images=image_b, device=device)
            return za, zb

        if self.encode_fn is not None and images is not None:
            # Se for tensor, move para device. Se for lista, deixa o encode_fn lidar.
            if isinstance(images, torch.Tensor):
                images = images.to(device)
            tokens, mask = self._extract_tokens_via_encode_fn(images, device=device, **(encode_kwargs or {}))
        else:
            tokens, mask = self._extract_tokens_via_hidden_states(input_ids=input_ids, attention_mask=attention_mask, device=device, **(encode_kwargs or {}))
        
        pooled = self.pool(tokens, mask=mask)
        return self.head(pooled)

    # --- Métodos de Salvar/Carregar ---
    def save_pretrained(self, save_directory, **kwargs):
        import os
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        state_dict = {}
        for n, p in self.named_parameters():
            if "head." in n or "pool." in n or p.requires_grad:
                state_dict[n] = p.cpu()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        if self.tokenizer: self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = CaVLConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config, *model_args, **kwargs)
        weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            keys = model.load_state_dict(state_dict, strict=False)
            print(f"CaVL Model carregado de {pretrained_model_name_or_path}")
        return model

# ----------------------
# Factory (Compatível com Treino Atual)
# ----------------------
def build_cavl_model(
        backbone: Any,
        tokenizer: Any = None,
        cut_layer: int = 27,
        encode_fn: Optional[Callable] = None,
        hidden_dim: int = 1536,
        proj_hidden: int = 4096,
        proj_out: int = 512,
        num_pool_heads: int = 8,
        pool_dim: Optional[int] = None,
        set_trainable: bool = True,
        # Novos argumentos
        pooler_type: str = "attention",
        head_type: str = "mlp",
        num_queries: int = 1,
        **kwargs 
) -> CaVLModel:
    if pool_dim is not None: hidden_dim = pool_dim

    model = CaVLModel(
        backbone_or_config=backbone,
        cut_layer=cut_layer,
        hidden_dim=hidden_dim,
        proj_hidden=proj_hidden,
        proj_out=proj_out,
        num_pool_heads=num_pool_heads,
        encode_fn=encode_fn,
        tokenizer=tokenizer,
        # Passando os novos
        pooler_type=pooler_type,
        head_type=head_type,
        num_queries=num_queries
    )
    if set_trainable:
        model.set_default_trainable()
    return model