# src/cavl_doc/data/dataset.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

# Mantendo seus imports originais
from cavl_doc.data.transforms import build_transform, dynamic_preprocess

class DocumentPairDataset(Dataset):
    def __init__(self, csv_path, base_dir, input_size=448, max_num=12, device="cuda"):
        """
        csv_path: path para train_pairs.csv
        base_dir: caminho base onde estão as imagens
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.input_size = input_size
        self.max_num = max_num
        self.device = device
        
        # Mantendo sua lógica de transformação
        self.transform = build_transform(input_size)

        # --- NOVO: Mapeamento de Classes para ArcFace ---
        # Verifica se as colunas de nome de classe existem no CSV
        if 'class_a_name' in self.df.columns and 'class_b_name' in self.df.columns:
            # Pega todas as classes únicas
            unique_classes = pd.concat([self.df['class_a_name'], self.df['class_b_name']]).dropna().unique()
            unique_classes = sorted([str(c) for c in unique_classes])
            
            # Cria mapa: {'invoice': 0, 'resume': 1, ...}
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
            self.num_classes = len(unique_classes)
            print(f"Dataset carregado: {len(self.df)} pares | {self.num_classes} classes (ArcFace enabled).")
        else:
            self.class_to_idx = {}
            self.num_classes = 0
            print(f"Dataset carregado: {len(self.df)} pares (Sem info de classes).")

    def __len__(self):
        return len(self.df)

    def _load_tensor(self, rel_path):
        img_path = os.path.join(self.base_dir, str(rel_path))
        try:
            image = Image.open(img_path).convert("RGB")
            blocks = dynamic_preprocess(image, image_size=self.input_size, use_thumbnail=True, max_num=self.max_num)
            tensor = [self.transform(b) for b in blocks]
            # Retorna [N, 3, H, W] - Mantém float32 na CPU para estabilidade dos workers
            return torch.stack(tensor)
        except Exception as e:
            print(f"Erro carregando imagem {img_path}: {e}")
            # Retorna tensor vazio para não quebrar o batch
            return torch.zeros((1, 3, self.input_size, self.input_size))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_a = self._load_tensor(row["file_a_path"])
        img_b = self._load_tensor(row["file_b_path"])
        label = float(row["is_equal"])

        # --- NOVO: Retorna IDs das classes ---
        cls_a = -1
        cls_b = -1
        
        if self.class_to_idx:
            # Pega o nome e converte para ID. Se não achar (ex: validação zero-shot), retorna -1
            name_a = str(row.get("class_a_name", ""))
            name_b = str(row.get("class_b_name", ""))
            cls_a = self.class_to_idx.get(name_a, -1)
            cls_b = self.class_to_idx.get(name_b, -1)

        return {
            "image_a": img_a,
            "image_b": img_b,
            "label": label,
            "class_a": cls_a, # Necessário para ArcFace
            "class_b": cls_b  # Necessário para ArcFace
        }