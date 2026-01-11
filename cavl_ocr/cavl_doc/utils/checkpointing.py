# src/utils/loadings.py

import torch
# Tenta importar a função de encode
from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding
from cavl_doc.models.modeling_cavl import CaVLModel

def load_trained_siamese(checkpoint_path, base_model, tokenizer, device, default_proj_out=512):
    print(f"⏳ Carregando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # --- 1. Configuração ---
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        print("   -> Formato: Smart Checkpoint (Novo)")
        config = checkpoint['config']
        cut_layer = config.get('cut_layer', 27)
        proj_out = config.get('projection_output_dim', default_proj_out)
        hidden_dim = config.get('hidden_dim', 1536)
        if 'metrics' in checkpoint:
            print(f"   -> EER no Treino: {checkpoint['metrics']['eer']*100:.2f}% (Epoch {checkpoint['epoch']})")
    else:
        print("   -> Formato: Legacy (Antigo)")
        cut_layer = 27
        proj_out = default_proj_out
        hidden_dim = 1536

    # --- 2. Detecção de Dimensão ---
    proj_hidden = 4096 
    if isinstance(checkpoint, dict) and 'siam_head' in checkpoint:
        if 'fc1.weight' in checkpoint['siam_head']:
            proj_hidden = checkpoint['siam_head']['fc1.weight'].shape[0]
    elif isinstance(checkpoint, dict) and 'head' in checkpoint:
        if 'fc1.weight' in checkpoint['head']:
            proj_hidden = checkpoint['head']['fc1.weight'].shape[0]
            
    print(f"   -> Config Final: Cut={cut_layer}, Hidden={proj_hidden}, Out={proj_out}")

    # --- 3. Encode Function (Closure) ---
    def _encode_fn(backbone, pv_tensor, cut_layer=cut_layer, **kwargs):
        prompt = "<image> Analyze this document"
        
        # --- CORREÇÃO CRÍTICA: Achatamento de 5D para 4D ---
        # O dataset retorna [1, N_Patches, 3, 448, 448]. O InternVL quer [N_Patches, 3, 448, 448].
        if pv_tensor.dim() == 5:
            b, n, c, h, w = pv_tensor.shape
            # Achata para processamento no vision encoder
            pv_tensor = pv_tensor.view(b * n, c, h, w)
            num_patches = n
        else:
            num_patches = pv_tensor.shape[0] # Assume que dim 0 já são patches ou batch 1

        # A função utilitária lida com a criação dos tokens <IMG_CONTEXT> baseada no num_patches
        inputs = prepare_inputs_for_multimodal_embedding(backbone, tokenizer, pv_tensor, prompt)
        
        out = backbone(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            pixel_values=inputs['pixel_values'].to(device, dtype=torch.bfloat16),
            image_flags=inputs['image_flags'].to(device),
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = out.hidden_states
        lm = backbone.language_model.model
        idx = cut_layer + 1 if len(hidden_states) == (len(lm.layers) + 1) else cut_layer
        return hidden_states[idx], None

    # --- 4. Instancia ---
    siam = CaVLModel(
        base_model,
        cut_layer=cut_layer,
        hidden_dim=hidden_dim,
        proj_hidden=proj_hidden, 
        proj_out=proj_out,
        encode_fn=_encode_fn
    )
    
    # --- 5. Carrega Pesos ---
    try:
        if isinstance(checkpoint, dict) and 'siam_head' in checkpoint:
            siam.head.load_state_dict(checkpoint['siam_head'])
            siam.pool.load_state_dict(checkpoint['siam_pool'])
            if 'backbone_trainable' in checkpoint:
                print(f"   -> Carregando pesos do backbone ({len(checkpoint['backbone_trainable'])} tensores)...")
                siam.backbone.load_state_dict(checkpoint['backbone_trainable'], strict=False)
        elif isinstance(checkpoint, dict) and 'head' in checkpoint:
            siam.head.load_state_dict(checkpoint['head'])
            if 'pool' in checkpoint: siam.pool.load_state_dict(checkpoint['pool'])
        else:
            siam.head.load_state_dict(checkpoint, strict=False)
            
    except Exception as e:
        print(f"❌ Erro ao carregar pesos: {e}")
        raise e

    siam.to(device)
    siam.eval()
    return siam