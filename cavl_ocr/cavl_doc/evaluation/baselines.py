#src/cavl_doc/evaluation/baselines.py
import torch
from tqdm import tqdm
from peft import PeftModel
import os
from cavl_doc.data.transforms import load_image 
from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding
import torch.nn.functional as F
from PIL import Image
import pandas as pd

def run_meanpooling_embedding_comparison(
    dataset, 
    model, 
    tokenizer, 
    prompt, 
    metric_type='cosine', 
    student_head=None
):
    """
    Iterates over the dataset, generates embeddings, and calculates the specified metric.
    [MODIFICADO] Agora suporta um 'student_head' opcional para a lógica RL-Head.
    """
    if metric_type not in ['cosine', 'euclidean']:
        raise ValueError("O parâmetro 'metric_type' deve ser 'cosine' ou 'euclidean'.")

    metric_name = "Similaridade de Cosseno" if metric_type == 'cosine' else "Distância Euclidiana"
    print(f"\nIniciando comparação com Embedding (Mean Pooling) e {metric_name}...")
    
    scores = []
    embedding_cache = {}
    
    model.eval()

    # VVV --- [LÓGICA NOVA] --- VVV
    # Verifica se a cabeça (head) foi fornecida e a prepara
    if student_head is not None:
        print(" -------------------------------------------------------------------------")
        print("   -> Detectado 'student_head' (Aluno-RL). Usando Base Congelada + Cabeça.")
        print(" -------------------------------------------------------------------------")
        student_head.to(model.device).eval()
    else:
        print(" -------------------------------------------------------------------------")
        print("   -> 'student_head' não fornecida. Usando modelo padrão (Base ou LoRA).")
        print(" -------------------------------------------------------------------------")
    # ^^^ --- [FIM DA LÓGICA NOVA] --- ^^^


    if isinstance(model, PeftModel):
        inference_model = model.base_model
        print("   -> Detectado PeftModel (fine-tuned). Usando 'model.base_model' para inferência.")
    else:
        inference_model = model
        print("   -> Detectado modelo base padrão. Usando 'model' diretamente para inferência.")

    with torch.no_grad():
        for i in tqdm(range(len(dataset.df)), desc="Calculando Embeddings (Mean Pooling)"):
            row = dataset.df.iloc[i]
            path_a = os.path.join(dataset.base_dir, row["file_a_path"])
            path_b = os.path.join(dataset.base_dir, row["file_b_path"])

            # Gerar embedding para a imagem A
            if path_a not in embedding_cache:
                pixel_values1 = load_image(path_a, max_num=12).to(model.device)
                inputs1 = prepare_inputs_for_multimodal_embedding(model, tokenizer, pixel_values1, prompt)
                
                outputs1 = inference_model(
                    input_ids=inputs1['input_ids'],
                    attention_mask=inputs1['attention_mask'],
                    pixel_values=inputs1['pixel_values'],
                    image_flags=inputs1['image_flags'],
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # VVV --- [LÓGICA MODIFICADA] --- VVV
                # 1. Pega o embedding de mean pooling (como antes)
                pooled_output1 = outputs1.hidden_states[-1].mean(dim=1)

                # 2. Passa pela cabeça (head) se ela existir
                if student_head is not None:
                    # Converte para float32 (pois a cabeça foi treinada em float32)
                    final_embedding1 = student_head(pooled_output1)
                else:
                    final_embedding1 = pooled_output1
                
                embedding_cache[path_a] = final_embedding1.cpu().squeeze()
                # ^^^ --- [FIM DA LÓGICA MODIFICADA] --- ^^^


            # Gerar embedding para a imagem B
            if path_b not in embedding_cache:
                pixel_values2 = load_image(path_b, max_num=12).to(model.device)
                inputs2 = prepare_inputs_for_multimodal_embedding(model, tokenizer, pixel_values2, prompt)
                
                outputs2 = inference_model(
                    input_ids=inputs2['input_ids'],
                    attention_mask=inputs2['attention_mask'],
                    pixel_values=inputs2['pixel_values'],
                    image_flags=inputs2['image_flags'],
                    output_hidden_states=True,
                    return_dict=True
                )

                # VVV --- [LÓGICA MODIFICADA] --- VVV
                # 1. Pega o embedding de mean pooling
                pooled_output2 = outputs2.hidden_states[-1].mean(dim=1)
                
                # 2. Passa pela cabeça (head) se ela existir
                if student_head is not None:
                    final_embedding2 = student_head(pooled_output2)
                else:
                    final_embedding2 = pooled_output2

                embedding_cache[path_b] = final_embedding2.cpu().squeeze()
                # ^^^ --- [FIM DA LÓGICA MODIFICADA] --- ^^^

            embedding1 = embedding_cache[path_a]
            embedding2 = embedding_cache[path_b]

            if metric_type == 'cosine':
                score = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()
            else: # euclidean
                score = F.pairwise_distance(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()
            
            scores.append(score)

    results_df = dataset.df.copy()
    results_df['metric_score'] = scores
    
    return results_df

def _calculate_image_distance(image_path1: str, image_path2: str, metric_type: str, resize_dim: tuple = (256, 256)) -> float:
    """
    Função interna que calcula a distância (cosseno ou euclidiana) entre duas imagens.
    """
    try:
        # Carregar, redimensionar, converter para grayscale e achatar em vetor
        img1 = Image.open(image_path1).convert('L').resize(resize_dim)
        vec1 = torch.tensor(list(img1.getdata()), dtype=torch.float32)

        img2 = Image.open(image_path2).convert('L').resize(resize_dim)
        vec2 = torch.tensor(list(img2.getdata()), dtype=torch.float32)

        # Normalizar os vetores
        vec1 = (vec1 - vec1.mean()) / vec1.std()
        vec2 = (vec2 - vec2.mean()) / vec2.std()

        # <<< LÓGICA CONDICIONAL AQUI >>>
        if metric_type == 'cosine':
            similarity = F.cosine_similarity(vec1, vec2, dim=0)
            return 1 - similarity.item() # Retorna distância
        elif metric_type == 'euclidean':
            return torch.dist(vec1, vec2).item() # Retorna distância euclidiana
        else:
            raise ValueError(f"Tipo de métrica de baseline desconhecido: {metric_type}")

    except Exception as e:
        print(f"Erro ao processar par de imagens: {e}")
        return float('inf') # Retorna distância máxima em caso de erro
    
def run_pixel_comparison(dataset, metric_type: str = 'cosine') -> pd.DataFrame:
    """
    Itera sobre o dataset e calcula a distância entre pixels para cada par,
    usando a métrica especificada.
    """
    if metric_type not in ['cosine', 'euclidean']:
        raise ValueError("O parâmetro 'metric_type' para baseline deve ser 'cosine' ou 'euclidean'.")

    print(f"\nIniciando a comparação de baseline (distância de pixel: {metric_type})...")
    
    distances = []
    for i in tqdm(range(len(dataset.df)), desc=f"Processando pares (pixel {metric_type})"):
        row = dataset.df.iloc[i]
        path_a = os.path.join(dataset.base_dir, row["file_a_path"])
        path_b = os.path.join(dataset.base_dir, row["file_b_path"])
        
        # Chama a função generalizada
        distance = _calculate_image_distance(path_a, path_b, metric_type=metric_type)
        distances.append(distance)

    results_df = dataset.df.copy()
    results_df['metric_score'] = distances
    
    return results_df