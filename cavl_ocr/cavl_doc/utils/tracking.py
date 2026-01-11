#src/cavl_doc/utils/tracking.py

import pandas as pd
from datetime import datetime
import os
import wandb

class ExperimentTracker:
    """
    Gerencia uma tabela de resultados de EER, salvando todos os parâmetros
    de cada experimento de forma flexível.
    """
    def __init__(self, file_path='results/default_results.csv'):
        """
        Inicializa o tracker, criando o diretório de resultados se necessário
        e carregando dados existentes.
        """
        self.file_path = file_path
        # Garante que o diretório de resultados exista antes de tentar carregar
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.results = self.load()
        print(f"ExperimentTracker inicializado. Rastreando resultados em '{self.file_path}'.")

    def load(self) -> pd.DataFrame:
        """Carrega a tabela de resultados de um arquivo CSV."""
        try:
            return pd.read_csv(self.file_path)
        except FileNotFoundError:
            return pd.DataFrame()

    def save(self):
        """Salva a tabela de resultados atual no arquivo CSV."""
        self.results.to_csv(self.file_path, index=False)
        print(f"Tabela de resultados salva em '{self.file_path}'.")

    def log(self, **kwargs):
        """
        Registra uma nova linha no log de experimentos com todos os dados fornecidos.
        Usa **kwargs para aceitar qualquer número de parâmetros de experimento.
        """
        # Adiciona um carimbo de data/hora universal a cada registro
        new_data = kwargs
        new_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_result_df = pd.DataFrame([new_data])
        
        # Concatena o novo resultado, garantindo que todas as colunas sejam mantidas
        if self.results.empty:
            self.results = new_result_df
        else:
            self.results = pd.concat([self.results, new_result_df], ignore_index=True)
        
        method_name = new_data.get('method_name', 'N/A')
        eer_score = new_data.get('eer', 'N/A')
        print(f"Resultado registrado: Método='{method_name}', EER={eer_score:.4f}")
        self.save()

    def display(self):
        """Exibe a tabela de resultados formatada, ordenada pelo EER."""
        print(f"\n--- Tabela de Resultados de '{os.path.basename(self.file_path)}' ---")
        if self.results.empty:
            print("Nenhum resultado registrado ainda.")
        else:
            # Define uma ordem de colunas preferencial para exibição
            display_cols = [
                'method_name', 'eer', 'model', 'metric', 
                'dataset', 'partition', 'timestamp', 'prompt_text'
            ]
            # Filtra apenas as colunas que realmente existem no DataFrame
            existing_cols = [col for col in display_cols if col in self.results.columns]
            
            # Ordena pelo EER para mostrar os melhores resultados primeiro
            sorted_df = self.results.sort_values(by='eer', ascending=True)
            print(sorted_df[existing_cols].to_string())
        print("---------------------------------------")

def fetch_wandb_runs(entity=None, project="CaVL-Doc-Experiments"):
    """
    Baixa o histórico de experimentos direto do WandB.
    
    Args:
        entity: Seu usuário ou organização no WandB (ex: 'jpcosta1990...'). 
                Se None, usa o default configurado no login.
        project: Nome do projeto.
    """
    api = wandb.Api()
    
    # Constrói o caminho "usuario/projeto"
    path = f"{entity}/{project}" if entity else project
    
    try:
        runs = api.runs(path)
    except Exception as e:
        print(f"Erro ao conectar ao WandB ({path}): {e}")
        return pd.DataFrame()

    summary_list = [] 
    config_list = [] 
    name_list = [] 
    status_list = []
    
    print(f"Baixando dados de {len(runs)} experimentos do WandB...")
    
    for run in runs: 
        # 1. Resumo (Métricas finais: best_eer, loss, etc)
        # O .summary contém o último valor ou o valor marcado como summary (min/max)
        summary_list.append(run.summary._json_dict) 
        
        # 2. Configuração (Hiperparâmetros: lr, batch_size, head_type...)
        config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')}) 
        
        # 3. Metadados
        name_list.append(run.name) 
        status_list.append(run.state) # running, finished, crashed

    # Cria DataFrames
    summary_df = pd.DataFrame(summary_list) 
    config_df = pd.DataFrame(config_list) 
    
    # Junta tudo
    runs_df = pd.concat([
        pd.DataFrame({'name': name_list, 'status': status_list}),
        config_df, 
        summary_df
    ], axis=1)
    
    return runs_df