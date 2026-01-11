# src/cavl_doc/evaluation/metrics.py
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier

def compute_eer(labels, scores):
    """Computa EER dado labels binários (0/1) e scores de similaridade."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    thr = thresholds[idx]
    return eer, thr

def compute_knn_metrics(embeddings, labels, k_vals=[1, 5]):
    """
    Computa acurácia k-NN (Top-1 e Top-k).
    embeddings: [N, Dim]
    labels: [N] (Class IDs)
    """
    # Precisamos de pelo menos uma amostra de cada classe para treinar o KNN
    if len(set(labels)) < 2:
        return {k: 0.0 for k in k_vals}

    # Para validação rápida, usamos 'Leave-One-Out' implícito ou fit/predict no mesmo set
    # A maneira correta para 'Zero-Shot Retrieval' em um batch único é:
    # Para cada query, buscar no resto do set.
    
    knn = KNeighborsClassifier(n_neighbors=max(k_vals), metric='cosine')
    knn.fit(embeddings, labels)
    
    # Pega os vizinhos (o primeiro vizinho é ele mesmo, distância 0, então pegamos k+1)
    distances, indices = knn.kneighbors(embeddings, n_neighbors=max(k_vals) + 1)
    
    accuracies = {}
    for k in k_vals:
        correct = 0
        total = 0
        for i in range(len(embeddings)):
            # Ignora o vizinho 0 (ele mesmo)
            # Verifica se algum dos k vizinhos subsequentes tem a mesma classe
            # (Lógica de Top-k Accuracy clássica: se a classe correta está nos top-k)
            # Mas para k-NN puro (classificação), fazemos votação.
            # Para RETRIEVAL (sua tarefa), queremos saber: O vizinho mais próximo é da mesma classe?
            
            # Vamos usar a métrica: P@1 (Precision at 1) - O vizinho mais próximo está certo?
            if k == 1:
                neighbor_idx = indices[i, 1] # O vizinho mais próximo real
                if labels[neighbor_idx] == labels[i]:
                    correct += 1
            else:
                # P@k ou Recall@k: A classe correta aparece nos k vizinhos?
                neighbors_indices = indices[i, 1:k+1]
                neighbor_labels = labels[neighbors_indices]
                if labels[i] in neighbor_labels:
                    correct += 1
            total += 1
        accuracies[f"R@{k}"] = correct / total
        
    return accuracies