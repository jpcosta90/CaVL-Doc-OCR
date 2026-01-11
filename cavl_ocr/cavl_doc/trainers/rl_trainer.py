# src/cavl_doc/trainers/rl_trainer.py
import os
import csv
import logging
import math
from tqdm import tqdm
import random
import time
import gc

import numpy as np
from sklearn.metrics import roc_curve

import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Subset

# Imports do Projeto
from cavl_doc.models.policy import ProfessorNetwork
from cavl_doc.modules.losses import build_loss
from cavl_doc.data.dataset import DocumentPairDataset
from cavl_doc.models.modeling_cavl import build_cavl_model

# Importa métrica de k-NN para validação robusta
try:
    from cavl_doc.evaluation.metrics import compute_knn_metrics
except ImportError:
    # Fallback simples se a função não existir ainda
    def compute_knn_metrics(*args, **kwargs): return {'R@1': 0.0}

try:
    import wandb
except ImportError:
    wandb = None

try:
    from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding
except ImportError:
    try:
        from cavl_doc.metrics.evaluation import prepare_inputs_for_multimodal_embedding
    except ImportError:
        raise ImportError("prepare_inputs_for_multimodal_embedding não encontrado.")

logger = logging.getLogger(__name__)
EMBEDDING_PROMPT = "<image> Analyze this document"

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def rl_full_collate_fn(batch):
    img_a_list = [item['image_a'] for item in batch]
    img_b_list = [item['image_b'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    
    if 'class_a' in batch[0]:
        class_a = torch.tensor([item['class_a'] for item in batch], dtype=torch.long)
        class_b = torch.tensor([item['class_b'] for item in batch], dtype=torch.long)
    else:
        class_a = torch.zeros(len(batch), dtype=torch.long)
        class_b = torch.zeros(len(batch), dtype=torch.long)
        
    return img_a_list, img_b_list, labels, class_a, class_b

def calculate_batch_recall(embeddings, labels, k=1):
    """
    Computa Recall@k para o batch atual.
    """
    if embeddings.size(0) < 2: return 0.0
    
    # Normaliza
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Matriz de Similaridade
    sim_matrix = torch.matmul(embeddings, embeddings.T)
    
    # Remove diagonal
    diag_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool).to(embeddings.device)
    sim_matrix.masked_fill_(diag_mask, -float('inf'))
    
    # Top-k
    _, indices = sim_matrix.topk(k, dim=1)
    
    # Check Neighbors
    neighbor_labels = labels[indices]
    target_labels = labels.unsqueeze(1)
    hits = (neighbor_labels == target_labels).any(dim=1).float()
    
    return hits.mean().item()

def pairwise_eer_from_scores(labels: np.ndarray, scores: np.ndarray):
    if len(labels) == 0: return 1.0, 0.0
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0, thresholds[idx]

def validate_siam_on_loader(siam, val_loader, device, student_criterion, limit_batches=None):
    siam.eval()
    student_criterion.eval()
    losses = []
    all_scores = []
    all_labels = []
    
    # Listas para k-NN
    knn_embeds = []
    knn_classes = []

    with torch.no_grad():
        # Ajusta o total da barra de progresso para refletir o limite
        total_batches = len(val_loader)
        if limit_batches:
            total_batches = min(total_batches, limit_batches)
            
        pbar_val = tqdm(val_loader, desc="Validating", ncols=100, leave=False, total=total_batches)
        batch_count = 0
        for img_a_list, img_b_list, labels, cls_a, cls_b in pbar_val:
            if limit_batches and batch_count >= limit_batches:
                pbar_val.close()
                break
            batch_count += 1
            
            # Processamento em chunks para evitar OOM se o batch for grande (ex: 96)
            # InternVL consome muita VRAM, então processamos em mini-batches
            chunk_size = 12 
            emb_a_chunks = []
            emb_b_chunks = []
            
            for i in range(0, len(img_a_list), chunk_size):
                chunk_a = img_a_list[i:i+chunk_size]
                chunk_b = img_b_list[i:i+chunk_size]
                
                # Inferência Batched (Otimizada)
                # Passamos a lista de tensores diretamente para o modelo.
                # O _encode_fn atualizado no script de treino lida com o batching e padding.
                za = siam(images=chunk_a)
                zb = siam(images=chunk_b)
                
                emb_a_chunks.append(za)
                emb_b_chunks.append(zb)
            
            za = torch.cat(emb_a_chunks, dim=0)
            zb = torch.cat(emb_b_chunks, dim=0)
            labels = labels.to(device)
            
            # --- [DIAGNÓSTICO DA LOSS] ---
            # Tenta calcular loss original, se falhar (ArcFace), usa proxy de cosseno
            try:
                ind_losses = student_criterion.forward_individual(za, zb, labels)
                losses.append(ind_losses.mean().item())
            except:
                cos_sim = torch.nn.functional.cosine_similarity(za, zb)
                dummy_loss = (labels * (1 - cos_sim) + (1 - labels) * torch.relu(cos_sim)).mean()
                losses.append(dummy_loss.item())
            # -----------------------------

            scores = torch.nn.functional.cosine_similarity(za, zb, dim=-1).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.cpu().numpy())

            # Acumula k-NN
            knn_embeds.append(za.cpu().numpy())
            knn_embeds.append(zb.cpu().numpy())
            knn_classes.append(cls_a.cpu().numpy())
            knn_classes.append(cls_b.cpu().numpy())

    if len(losses) == 0: return float('nan'), 1.0, 0.0, 0.0
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mean_loss = float(np.mean(losses))
    eer, thr = pairwise_eer_from_scores(all_labels, all_scores)

    # --- [DIAGNÓSTICO DO K-NN] ---
    r1 = 0.0
    try:
        full_embeds = np.vstack(knn_embeds)
        full_classes = np.concatenate(knn_classes, axis=0)
        
        mask = full_classes != -1
        
        if mask.sum() > 1:
            metrics = compute_knn_metrics(full_embeds[mask], full_classes[mask], k_vals=[1])
            r1 = metrics.get('R@1', 0.0)
        else:
            # Se não houver classes válidas, R@1 fica 0.0
            pass
            
    except Exception as e:
        print(f"\n[DEBUG k-NN] ❌ Erro no cálculo: {e}")

    return mean_loss, eer, thr, r1

def run_rl_siamese_loop(
    base_model, student_head, professor_model, tokenizer, dataset, epochs, student_lr, professor_lr,
    device, output_dir, candidate_pool_size, student_batch_size, max_num_image_tokens,
    val_csv_path=None, base_image_dir=None,
    cut_layer=27, projection_output_dim=512, val_fraction=0.05, val_min_size=200,
    patience=3, lr_reduce_factor=0.5, baseline_alpha=0.01, entropy_coeff=0.01, seed=42,
    use_wandb=False,
    loss_type="contrastive", pooler_type="attention", head_type="mlp", num_classes=None, num_queries=1,
    val_samples_per_class=20, # Novo argumento com default
    # Novos argumentos para losses
    margin=0.5, scale=64.0, num_sub_centers=3, std=0.05,
    # Argumento para retomar treinamento
    resume_checkpoint_path=None,
    # Optimizer/Scheduler configs
    optimizer_type="adam",
    scheduler_type=None,
    # Debug/Sweep Control
    max_steps_per_epoch=None,
    professor_warmup_steps=0,
    easy_mining_steps=0, # Novo argumento para curriculum de exemplos fáceis
    gradient_accumulation_steps=1 # Novo argumento para acumulação de gradiente
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(device if isinstance(device, str) else device)

    model_config = {
        'cut_layer': cut_layer,
        'projection_output_dim': projection_output_dim,
        'max_num_image_tokens': max_num_image_tokens,
        'hidden_dim': 1536,
        'loss_type': loss_type,
        'head_type': head_type,
        'pooler_type': pooler_type,
        'num_queries': num_queries,
        # Use as variáveis diretamente, pois elas são argumentos da função
        'margin': margin,  
        'scale': scale,
        'num_sub_centers': num_sub_centers,
        'std': std
    }

    # Define encode_fn (COM CORREÇÃO DE FLATTENING 5D E SUPORTE A LISTAS)
    def _encode_fn(backbone, images, cut_layer=cut_layer, **kwargs):
        # images: Tensor [B, N, C, H, W] ou List[Tensor [Ni, C, H, W]]
        
        input_ids_list = []
        pixel_values_list = []
        image_flags_list = []
        
        # Normaliza entrada para lista
        if isinstance(images, torch.Tensor):
            if images.dim() == 5:
                images_list = [images[i] for i in range(images.shape[0])]
            else:
                images_list = [images]
        else:
            images_list = images

        # Prepara inputs individualmente
        for img in images_list:
            out = prepare_inputs_for_multimodal_embedding(backbone, tokenizer, img, EMBEDDING_PROMPT)
            input_ids_list.append(out['input_ids'][0]) 
            pixel_values_list.append(out['pixel_values']) 
            image_flags_list.append(out['image_flags']) 

        # Padding manual dos input_ids
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            p_ids = torch.cat([ids, torch.full((pad_len,), pad_id, device=ids.device, dtype=ids.dtype)])
            p_mask = torch.cat([torch.ones_like(ids), torch.zeros((pad_len,), device=ids.device, dtype=ids.dtype)])
            padded_input_ids.append(p_ids)
            padded_attention_mask.append(p_mask)

        # Batch final
        batch_input_ids = torch.stack(padded_input_ids).to(device)
        batch_attention_mask = torch.stack(padded_attention_mask).to(device)
        batch_pixel_values = torch.cat(pixel_values_list, dim=0).to(device, dtype=torch.bfloat16)
        batch_image_flags = torch.cat(image_flags_list, dim=0).to(device)

        out = backbone(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            pixel_values=batch_pixel_values,
            image_flags=batch_image_flags,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = out.hidden_states
        lm = backbone.language_model.model
        idx = cut_layer + 1 if len(hidden_states) == (len(lm.layers) + 1) else cut_layer
        return hidden_states[idx], None

    siam = build_cavl_model(
        backbone=base_model, 
        cut_layer=cut_layer, 
        encode_fn=_encode_fn,
        pool_dim=student_head.ln.normalized_shape[0] if hasattr(student_head, 'ln') else 1536,
        proj_hidden=getattr(student_head, 'fc1').out_features if hasattr(student_head, 'fc1') else 4096,
        proj_out=projection_output_dim,
        set_trainable=True,
        tokenizer=tokenizer,
        pooler_type=pooler_type,
        head_type=head_type,
        num_queries=num_queries
    )
    siam.to(device)
    siam.set_default_trainable()
    
    professor_model.to(device).train()

    # Loss Builder com novos parâmetros
    print(f"Construindo Loss: {loss_type} (m={margin}, s={scale}, k={num_sub_centers}, std={std})")
    student_criterion = build_loss(
        loss_type, 
        margin=margin, 
        m=margin, # ArcFace/CosFace/Circle/IsoCircle usam 'm'
        s=scale,
        gamma=scale, # Circle/IsoCircle usa gamma
        num_classes=num_classes, 
        k=num_sub_centers,
        std=std,
        in_features=projection_output_dim
    ).to(device)

    trainable_params = [p for n,p in siam.named_parameters() if p.requires_grad]
    loss_params = list(student_criterion.parameters())
    
    # --- Optimizer Setup ---
    print(f"DEBUG: Optimizer: {optimizer_type}, Scheduler: {scheduler_type}")
    if optimizer_type and optimizer_type.lower() == "sgd":
        student_optimizer = optim.SGD(trainable_params + loss_params, lr=student_lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_type and optimizer_type.lower() == "adam":
        student_optimizer = optim.Adam(trainable_params + loss_params, lr=student_lr, weight_decay=1e-4) # Adam tradicional
    else:
        # Default AdamW (Padrão Moderno)
        student_optimizer = optim.AdamW(trainable_params + loss_params, lr=student_lr, weight_decay=1e-4)
        
    professor_optimizer = optim.Adam(professor_model.parameters(), lr=professor_lr) # Professor usa Adam padrão
    
    # --- Scheduler Setup ---
    student_scheduler = None
    if scheduler_type:
        if scheduler_type.lower() == "step":
             student_scheduler = optim.lr_scheduler.StepLR(student_optimizer, step_size=1, gamma=lr_reduce_factor)
        elif scheduler_type.lower() == "cosine":
             # T_max usually epochs
             student_scheduler = optim.lr_scheduler.CosineAnnealingLR(student_optimizer, T_max=epochs)
        elif scheduler_type.lower() == "plateau":
             student_scheduler = optim.lr_scheduler.ReduceLROnPlateau(student_optimizer, mode='min', factor=lr_reduce_factor, patience=1) # Aggressive reduction


    # Debug Resume Path
    print(f"DEBUG: resume_checkpoint_path received: '{resume_checkpoint_path}'")
    if resume_checkpoint_path:
        print(f"DEBUG: Exists? {os.path.exists(resume_checkpoint_path)}")

    # --- Lógica de Retomada (Resume) ---
    start_epoch = 0
    best_val_eer = 1.0
    baseline = 0.0
    global_batch_step = 0
    
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"🔄 Retomando treinamento do checkpoint: {resume_checkpoint_path}")
        ckpt = torch.load(resume_checkpoint_path, map_location=device)
        
        # Carrega pesos dos modelos
        siam.pool.load_state_dict(ckpt['siam_pool'])
        siam.head.load_state_dict(ckpt['siam_head'])
        professor_model.load_state_dict(ckpt['professor_state'])
        
        # Carrega backbone se houver pesos salvos
        if 'backbone_trainable' in ckpt and ckpt['backbone_trainable']:
            siam.backbone.load_state_dict(ckpt['backbone_trainable'], strict=False)
            
        # Carrega otimizadores (se disponíveis)
        if 'student_optimizer' in ckpt:
            student_optimizer.load_state_dict(ckpt['student_optimizer'])
        if 'professor_optimizer' in ckpt:
            professor_optimizer.load_state_dict(ckpt['professor_optimizer'])
        if 'student_scheduler' in ckpt and student_scheduler is not None:
             student_scheduler.load_state_dict(ckpt['student_scheduler'])

            
        # Restaura estado do treinamento
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_eer = ckpt.get('metrics', {}).get('eer', 1.0)
        baseline = ckpt.get('baseline', 0.0)
        global_batch_step = ckpt.get('global_batch_step', 0)
        no_improve = ckpt.get('no_improve', 0)
        
        # Verifica se o checkpoint foi salvo ANTES da validação (Pre-Validation)
        # Se sim, precisamos rodar a validação daquela época antes de prosseguir.
        stage = ckpt.get('stage', 'post_validation')
        
        if stage == 'pre_validation':
            print(f"⚠️ Retomando de checkpoint PRE-VALIDATION (Época {start_epoch}).")
            print("   -> A validação da época anterior não foi concluída. Ajustando start_epoch para repetir a validação.")
            # Recua o start_epoch para que o loop comece na época correta, mas precisamos pular o treino?
            # Não dá para pular o treino facilmente dentro do loop for.
            # Solução: Rodar a validação AGORA, salvar o checkpoint pós-validação, e seguir o baile.
            start_epoch -= 1 # Volta para a época que terminou o treino
            run_immediate_validation = True
        else:
            run_immediate_validation = False
            # Se já validou e atingiu paciência, para aqui.
            if no_improve >= patience:
                print(f"⏹️ Treinamento já finalizado anteriormente por critério de paciência ({no_improve} >= {patience}).")
                print("   -> Ignorando execução.")
                return

        print(f"   -> Reiniciando na Época {start_epoch+1}. Best EER anterior: {best_val_eer:.4f}. Patience Counter: {no_improve}/{patience}")

    if val_csv_path and os.path.exists(val_csv_path):
        print(f"Carregando validação: {val_csv_path}")
        val_dataset = DocumentPairDataset(val_csv_path, base_image_dir, 448, max_num_image_tokens, 'cpu')
        train_dataset = dataset
    else:
        print("Split automático treino/val.")
        if isinstance(dataset, Subset): full_ds, ds_indices = dataset.dataset, list(dataset.indices)
        else: full_ds, ds_indices = dataset, list(range(len(dataset)))
        
        # Garante que o split seja aleatório para evitar viés de classe (ex: pegar só as últimas classes)
        random.shuffle(ds_indices)
        
        val_size = min(max(val_min_size, int(len(ds_indices) * val_fraction)), len(ds_indices)//10)
        if val_size <= 0: val_size = min(val_min_size, len(ds_indices)//10)
        val_indices = ds_indices[-val_size:]; train_indices = ds_indices[:-val_size]
        
        train_dataset = Subset(full_ds, train_indices)
        val_dataset = Subset(full_ds, val_indices)

    # Validação com shuffle=True para garantir que os 20 batches limitados sejam uma amostra aleatória (balanceada estatisticamente)
    # num_workers=0 para evitar crash por falta de RAM (16GB é pouco para InternVL + Workers)
    train_loader = DataLoader(train_dataset, batch_size=candidate_pool_size, shuffle=True, num_workers=0, collate_fn=rl_full_collate_fn)
    
    # --- BALANCED VALIDATION SUBSET ---
    # Cria um subset de validação balanceado e reduzido para avaliação rápida entre épocas
    # Garante que todas as classes presentes na validação sejam avaliadas.
    balanced_val_indices = []
    # Usa o valor parametrizado
    samples_per_class = val_samples_per_class 
    
    # Identifica o dataset base e os índices da validação
    if isinstance(val_dataset, Subset):
        val_source_ds = val_dataset.dataset
        current_val_indices = list(val_dataset.indices)
    else:
        val_source_ds = val_dataset
        current_val_indices = list(range(len(val_dataset)))

    if hasattr(val_source_ds, 'df') and 'class_a_name' in val_source_ds.df.columns:
        print(f"Criando subset de validação balanceado ({samples_per_class} pares/classe)...")
        subset_df = val_source_ds.df.iloc[current_val_indices]
        classes = subset_df['class_a_name'].tolist()
        
        class_map = {}
        for i, cls_name in enumerate(classes):
            real_idx = current_val_indices[i]
            if cls_name not in class_map: class_map[cls_name] = []
            class_map[cls_name].append(real_idx)
            
        under_populated = []
        for cls_name, idxs in class_map.items():
            if len(idxs) > samples_per_class:
                balanced_val_indices.extend(random.sample(idxs, samples_per_class))
            else:
                balanced_val_indices.extend(idxs)
                under_populated.append(f"{cls_name}: {len(idxs)}")
        
        if under_populated:
            print(f"⚠️  {len(under_populated)} classes têm menos de {samples_per_class} amostras na validação:")
            print(f"   -> {', '.join(under_populated[:10])}..." if len(under_populated) > 10 else f"   -> {', '.join(under_populated)}")
        
        print(f"Subset balanceado criado: {len(balanced_val_indices)} amostras (de {len(current_val_indices)} originais). Classes: {len(class_map)}")
        val_dataset_balanced = Subset(val_source_ds, balanced_val_indices)
    else:
        # Fallback se não tiver info de classe
        print("Aviso: 'class_a_name' não encontrado. Usando subset aleatório de 1000 amostras.")
        n_samples = min(len(current_val_indices), 1000)
        balanced_val_indices = random.sample(current_val_indices, n_samples)
        val_dataset_balanced = Subset(val_source_ds, balanced_val_indices)

    # num_workers=0 para estabilidade
    val_loader = DataLoader(val_dataset_balanced, batch_size=24, shuffle=False, num_workers=0, collate_fn=rl_full_collate_fn)
    
    print(f"Dataset Sizes: Train={len(train_dataset)} | Val (Balanced Subset)={len(val_dataset_balanced)}")

    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "training_log.csv")
    
    # Se estiver retomando, append; senão, write
    mode = 'a' if (resume_checkpoint_path and os.path.exists(resume_checkpoint_path)) else 'w'
    log_file = open(log_file_path, mode, newline='')
    log_writer = csv.writer(log_file)
    
    if mode == 'w':
        log_writer.writerow(['epoch', 'batch', 'aluno_loss', 'prof_loss', 'reward', 'baseline', 'entropy', 'adv_std', 'val_mean_loss', 'val_eer'])

    # Variáveis de estado (já inicializadas acima se resume=True)
    # best_val_eer, baseline, start_epoch, global_batch_step já definidos
    no_improve = 0

    def student_forward_pass(pv_list_a, pv_list_b, train_student=True):
        if train_student: siam.train()
        else: siam.eval()
        emb_a_list, emb_b_list = [], []
        for pv in pv_list_a:
            pv = pv.unsqueeze(0).to(device)
            if train_student: z = siam(images=pv)
            else:
                with torch.no_grad(): z = siam(images=pv)
            emb_a_list.append(z)
        for pv in pv_list_b:
            pv = pv.unsqueeze(0).to(device)
            if train_student: z = siam(images=pv)
            else:
                with torch.no_grad(): z = siam(images=pv)
            emb_b_list.append(z)
        return torch.cat(emb_a_list, dim=0), torch.cat(emb_b_list, dim=0)

    # --- VALIDAÇÃO IMEDIATA (RESUME FIX) ---
    if resume_checkpoint_path and run_immediate_validation:
        print(f"🚨 Executando validação pendente da Época {start_epoch+1}...")
        # Validação no subset balanceado (sem limite de batches, pois o subset já é pequeno)
        vloss, veer, vthr, vr1 = validate_siam_on_loader(siam, val_loader, device, student_criterion)
        print(f"Val (Recovered): EER={veer*100:.2f}% | R@1={vr1*100:.2f}% | Loss={vloss:.4f}")
        
        if use_wandb and wandb:
            wandb.log({"val/eer": veer, "val/loss": vloss, "val/recall_at_1": vr1, "epoch": start_epoch + 1})
        
        log_writer.writerow([start_epoch+1, "end", "", "", "", "", "", "", f"{vloss:.4f}", f"{veer:.4f}"])
        
        if veer < best_val_eer:
            best_val_eer = veer
            no_improve = 0
            backbone_trainable = {n: p.detach().cpu() for n, p in siam.backbone.named_parameters() if p.requires_grad}
            ckpt = {
                'epoch': start_epoch, 'metrics': {'eer': veer, 'loss': vloss}, 'config': model_config,
                'siam_pool': siam.pool.state_dict(), 'siam_head': siam.head.state_dict(),
                'backbone_trainable': backbone_trainable, 'professor_state': professor_model.state_dict()
            }
            os.makedirs(output_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(output_dir, "best_siam.pt"))
            print("Saved best_siam.pt (Recovered)")
            if use_wandb and wandb: wandb.log({"val/best_eer": best_val_eer})
        
        # --- SAVE RECOVERED STATE ---
        # Salva o checkpoint atualizado para evitar loop infinito se houver crash em seguida
        # Isso garante que, se o script morrer por OOM ao iniciar o treino, ele não repetirá a validação.
        backbone_trainable = {n: p.detach().cpu() for n, p in siam.backbone.named_parameters() if p.requires_grad}
        recovered_ckpt = {
            'epoch': start_epoch, 
            'metrics': {'eer': veer, 'loss': vloss}, 
            'config': model_config,
            'siam_pool': siam.pool.state_dict(), 
            'siam_head': siam.head.state_dict(),
            'backbone_trainable': backbone_trainable, 
            'professor_state': professor_model.state_dict(),
            'student_optimizer': student_optimizer.state_dict(),
            'professor_optimizer': professor_optimizer.state_dict(),
            'baseline': baseline,
            'global_batch_step': global_batch_step,
            'no_improve': no_improve, # Salva estado da paciência
            'stage': 'post_validation' # Marcamos como concluído!
        }
        torch.save(recovered_ckpt, os.path.join(output_dir, "last_checkpoint.pt"))
        print(f"💾 Checkpoint de recuperação salvo: last_checkpoint.pt (Stage: post_validation)")

        # Avança para a próxima época agora que validamos
        start_epoch += 1
        print(f"✅ Validação recuperada. Continuando treino na Época {start_epoch+1}...")
        
        # Limpeza de memória após validação recuperada
        gc.collect()
        torch.cuda.empty_cache()

    print("Iniciando treinamento...")
    for epoch in range(start_epoch, epochs):
        # Limpeza preventiva antes de iniciar o loop de treino
        gc.collect()
        torch.cuda.empty_cache()
        
        student_criterion.train()
        print(f"\n--- Época {epoch + 1}/{epochs} ---")
        avg_loss = AverageMeter(); avg_rew = AverageMeter(); avg_prof_loss = AverageMeter()
        
        # Update lento para não travar UI
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", mininterval=30.0, ncols=100)

        for batch_idx, (img_a, img_b, labels, cls_a, cls_b) in enumerate(pbar):
            # --- Check Max Steps (Sweep) ---
            if max_steps_per_epoch is not None and batch_idx >= max_steps_per_epoch:
                print(f"\n⚠️  Limite de steps atingido ({max_steps_per_epoch}). Encerrando época.")
                break

            labels = labels.to(device).float()
            cls_a = cls_a.to(device)
            cls_b = cls_b.to(device)

            # 1. Professor Selection
            professor_model.train()
            
            # Warmup Check
            is_warmup = global_batch_step < professor_warmup_steps
            
            with torch.no_grad():
                ea, eb = student_forward_pass(img_a, img_b, False)
                if loss_type in ['contrastive', 'angular']:
                    sl = student_criterion.forward_individual(ea, eb, labels)
                elif loss_type == 'triplet':
                    full_emb = torch.cat([ea, eb], dim=0)
                    full_cls = torch.cat([cls_a, cls_b], dim=0)
                    full_loss = student_criterion.forward_individual(full_emb, full_cls)
                    sl = (full_loss[:len(ea)] + full_loss[len(ea):]) / 2.0
                else:
                    loss_a = student_criterion.forward_individual(ea, cls_a)
                    loss_b = student_criterion.forward_individual(eb, cls_b)
                    sl = (loss_a + loss_b) / 2.0

                denom = (sl.max() - sl.min()).item() or 1.0
                sl_norm = (sl - sl.min()) / (denom + 1e-6)
                state = sl_norm.unsqueeze(-1)
            
            # Decide Indices
            if is_warmup:
                 # Uniform Sampling
                 idxs = torch.randint(0, len(img_a), (student_batch_size,)).to(device)
                 log_probs = None
            else:
                 logits = professor_model(state).squeeze(-1)
                 dist = Categorical(logits=logits)
                 idxs = dist.sample((student_batch_size,))
                 log_probs = dist.log_prob(idxs)

            # 2. Student Update
            sel_idxs = idxs.tolist()
            sa = [img_a[i] for i in sel_idxs]; sb = [img_b[i] for i in sel_idxs]
            slbs = labels[sel_idxs]
            s_cls_a = cls_a[sel_idxs]; s_cls_b = cls_b[sel_idxs]
            
            # Accumulalation Logic: Zero Grad at start of accumulation block
            if (global_batch_step % gradient_accumulation_steps) == 0:
                student_optimizer.zero_grad()
                
            sea, seb = student_forward_pass(sa, sb, True)
            
            if loss_type in ['contrastive', 'angular']:
                loss = student_criterion(sea, seb, slbs)
            elif loss_type == 'triplet':
                # Triplet: Concatenar A e B para formar um batch maior
                full_emb = torch.cat([sea, seb], dim=0)
                full_cls = torch.cat([s_cls_a, s_cls_b], dim=0)
                loss = student_criterion(full_emb, full_cls)
            else:
                loss = student_criterion(sea, s_cls_a) + student_criterion(seb, s_cls_b)

            # --- DIAGNOSTICO BATCH RECALL ---
            with torch.no_grad():
                 batch_emb = torch.cat([sea.detach(), seb.detach()], dim=0)
                 batch_cls = torch.cat([s_cls_a, s_cls_b], dim=0)
                 batch_recall = calculate_batch_recall(batch_emb, batch_cls, k=1)
            # --------------------------------
            
            if torch.isnan(loss): continue
            
            # Scale Loss for Accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Step only after N accumulation steps
            if ((global_batch_step + 1) % gradient_accumulation_steps) == 0:
                # Gradient Clipping (Global) - Deve ser APÓS backward e ANTES de step
                torch.nn.utils.clip_grad_norm_(trainable_params + loss_params, max_norm=1.0)
                student_optimizer.step()


            # 3. Update Prof
            with torch.no_grad():
                if loss_type in ['contrastive', 'angular']:
                    s_ind = student_criterion.forward_individual(sea.detach(), seb.detach(), slbs)
                else:
                    l_a = student_criterion.forward_individual(sea.detach(), s_cls_a)
                    l_b = student_criterion.forward_individual(seb.detach(), s_cls_b)
                    s_ind = (l_a + l_b) / 2.0

            # --- RL REWARD STRATEGY ---
            raw_loss_rewards = s_ind.detach()
            
            is_easy_mining = global_batch_step < easy_mining_steps
            
            if is_easy_mining:
                 # Easy Mining Phase: Maximize probability of LOW loss.
                 # Reward = -Loss. Higher reward for smaller loss.
                 rew = -raw_loss_rewards
            else:
                 # Hard Mining Phase (Default): Maximize probability of HIGH loss.
                 # Reward = Loss. Higher reward for larger loss.
                 rew = raw_loss_rewards

            avg_r = float(rew.mean().item())
            
            # Professor Update (Only if NOT warmup)
            ploss_val = 0.0
            if not is_warmup and log_probs is not None:
                adv = rew - baseline
                adv_std_val = adv.std(unbiased=False).clamp(min=1e-6)
                adv_norm = (adv - adv.mean()) / (adv_std_val + 1e-6)
                
                professor_optimizer.zero_grad()
                ploss = - (log_probs * adv_norm).mean() - entropy_coeff * dist.entropy().mean()
                ploss.backward()
                professor_optimizer.step()
                ploss_val = ploss.item()

            baseline = (1 - baseline_alpha) * baseline + baseline_alpha * avg_r
            
            global_batch_step += 1
            # Rescale loss for logging
            actual_loss = loss.item() * gradient_accumulation_steps
            avg_loss.update(actual_loss); avg_rew.update(avg_r); avg_prof_loss.update(ploss_val)
            
            status_str = "Warmup" if is_warmup else ("Easy" if is_easy_mining else "Hard")
            # Removendo 'Rec' do postfix pois batch_recall não está no escopo global dessa função de substituição
            # e poderia quebrar se eu não o definisse aqui.
            # Vou manter simples:
            pbar.set_postfix({'L': f"{avg_loss.avg:.4f}", 'R': f"{avg_rew.avg:.4f}", 'Prof': status_str})
            
            if use_wandb and wandb:
                log_dict = {
                    "train/loss": actual_loss, "train/reward": avg_r, "train/prof_loss": ploss_val,
                    "train/baseline": baseline, 
                    "train/entropy": dist.entropy().mean().item() if not is_warmup else 0.0,
                    "step": global_batch_step,
                    "professor_active": 0 if is_warmup else 1
                }
                # Se 'batch_recall' existir no escopo local (definido anteriormente), adiciona
                if 'batch_recall' in locals():
                     log_dict["train/batch_recall"] = batch_recall
                
                wandb.log(log_dict)

            log_writer.writerow([epoch+1, global_batch_step, f"{loss.item():.4f}", f"{ploss_val:.4f}", f"{avg_r:.4f}", f"{status_str}", "", "", "", ""])

        pbar.close()

        # --- Save Checkpoint PRE-VALIDATION (Safety) ---
        # Salva o estado logo após o treino, antes da validação (que pode demorar ou falhar).
        # Se o script for interrompido na validação, ao retomar, ele pulará a validação desta época
        # e irá para a próxima, mas pelo menos o treino da época não é perdido.
        backbone_trainable = {n: p.detach().cpu() for n, p in siam.backbone.named_parameters() if p.requires_grad}
        pre_val_ckpt = {
            'epoch': epoch, 
            'metrics': {'eer': best_val_eer, 'loss': 0.0}, # Usa best anterior como placeholder
            'config': model_config,
            'siam_pool': siam.pool.state_dict(), 
            'siam_head': siam.head.state_dict(),
            'backbone_trainable': backbone_trainable, 
            'professor_state': professor_model.state_dict(),
            'student_optimizer': student_optimizer.state_dict(),
            'professor_optimizer': professor_optimizer.state_dict(),
            'student_scheduler': student_scheduler.state_dict() if student_scheduler else None,
            'baseline': baseline,
            'global_batch_step': global_batch_step,
            'no_improve': no_improve, # Salva estado da paciência
            'stage': 'pre_validation' # Flag para indicar que validação ainda não ocorreu
        }
        os.makedirs(output_dir, exist_ok=True)
        torch.save(pre_val_ckpt, os.path.join(output_dir, "last_checkpoint.pt"))
        print(f"Saved last_checkpoint.pt (Epoch {epoch+1} - Pre-Validation)")

        # Validação no subset balanceado
        vloss, veer, vthr, vr1 = validate_siam_on_loader(siam, val_loader, device, student_criterion)
        print(f"Val: EER={veer*100:.2f}% | R@1={vr1*100:.2f}% | Loss={vloss:.4f}")
        
        if use_wandb and wandb:
            wandb.log({"val/eer": veer, "val/loss": vloss, "val/recall_at_1": vr1, "epoch": epoch + 1})

        log_writer.writerow([epoch+1, "end", "", "", "", "", "", "", f"{vloss:.4f}", f"{veer:.4f}"])

        if veer < best_val_eer:
            best_val_eer = veer
            no_improve = 0
            backbone_trainable = {n: p.detach().cpu() for n, p in siam.backbone.named_parameters() if p.requires_grad}
            ckpt = {
                'epoch': epoch, 'metrics': {'eer': veer, 'loss': vloss}, 'config': model_config,
                'siam_pool': siam.pool.state_dict(), 'siam_head': siam.head.state_dict(),
                'backbone_trainable': backbone_trainable, 'professor_state': professor_model.state_dict()
            }
            
            # Garante que o diretório existe antes de salvar (defensivo contra falhas de FS)
            os.makedirs(output_dir, exist_ok=True)
            torch.save(ckpt, os.path.join(output_dir, "best_siam.pt"))
            print("Saved best_siam.pt")
            
            if use_wandb and wandb: wandb.log({"val/best_eer": best_val_eer})
        else:
            no_improve += 1
        
        # --- Scheduler Step ---
        if student_scheduler:
             if isinstance(student_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                 student_scheduler.step(veer)
             else:
                 student_scheduler.step()
             
             current_lr = student_optimizer.param_groups[0]['lr']
             print(f"Current LR: {current_lr}")
             if use_wandb and wandb:
                 wandb.log({"train/lr": current_lr, "epoch": epoch + 1})

        # --- Save Last Checkpoint (para Resume) ---
        backbone_trainable = {n: p.detach().cpu() for n, p in siam.backbone.named_parameters() if p.requires_grad}
        last_ckpt = {
            'epoch': epoch, 
            'metrics': {'eer': veer, 'loss': vloss}, 
            'config': model_config,
            'siam_pool': siam.pool.state_dict(), 
            'siam_head': siam.head.state_dict(),
            'backbone_trainable': backbone_trainable, 
            'professor_state': professor_model.state_dict(),
            'student_optimizer': student_optimizer.state_dict(),
            'professor_optimizer': professor_optimizer.state_dict(),
            'student_scheduler': student_scheduler.state_dict() if student_scheduler else None,
            'baseline': baseline,
            'global_batch_step': global_batch_step,
            'no_improve': no_improve, # Salva estado da paciência
            'stage': 'post_validation' # Flag para indicar que validação já ocorreu
        }
        
        os.makedirs(output_dir, exist_ok=True)
        torch.save(last_ckpt, os.path.join(output_dir, "last_checkpoint.pt"))
        print(f"Saved last_checkpoint.pt (Epoch {epoch+1})")
        
        if no_improve >= patience: break

    log_file.close()