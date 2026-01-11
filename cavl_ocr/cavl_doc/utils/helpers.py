import sys
import os
from datetime import datetime
from peft import PeftModel
import torch

class SuppressSpecificOutput:
    """
    Um gerenciador de contexto para suprimir mensagens de warning específicas
    do stdout, sem quebrar as barras de progresso como o tqdm.
    """
    def __init__(self, message_to_suppress):
        self.message = message_to_suppress
        self.original_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout

    def write(self, data):
        # Escreve os dados no stdout original apenas se não contiverem a mensagem a ser suprimida
        if self.message not in data:
            self.original_stdout.write(data)

    def flush(self):
        self.original_stdout.flush()

    def isatty(self):
        # Devolve a chamada para o stdout original para que bibliotecas como o tqdm
        # saibam que estão em um terminal interativo.
        return hasattr(self.original_stdout, 'isatty') and self.original_stdout.isatty()

def setup_experiment_dir(base_output_path: str, experiment_name: str) -> str:
    """
    Cria um diretório para salvar os resultados de um experimento, com um nome único.

    Args:
        base_output_path (str): O caminho base onde os diretórios de experimento serão criados (ex: "checkpoints", "results").
        experiment_name (str): O nome do experimento (geralmente gerado com informações do modelo e data/hora).

    Returns:
        str: O caminho completo para o diretório de saída do experimento.
    """
    full_output_dir = os.path.join(base_output_path, experiment_name)
    os.makedirs(full_output_dir, exist_ok=True)
    return full_output_dir

# ==========================================================
# 2. FUNÇÕES AUXILIARES
# ==========================================================
def count_total_parameters(model) -> float:
    # ... (A sua função count_total_parameters não muda) ...
    if isinstance(model, PeftModel):
        return sum(p.numel() for p in model.parameters()) / 1_000_000
    elif isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters()) / 1_000_000
    return 0.0