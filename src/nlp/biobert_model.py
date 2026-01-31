"""
Módulo de BioBERT para Processamento de Narrativas Clínicas

Autor: Frederico Guilheme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2024-08-01
Última Modificação: 2025-11-30

Descrição:
    Implementa BioBERT (BERT especializado em textos biomédicos) para extração
    de features de narrativas clínicas de pacientes com tuberculose.
    
    BioBERT é pré-treinado em corpus biomédico (PubMed) e é mais eficaz que
    BERT padrão para textos clínicos.

Licença: MIT
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class BioBERTModel:
    """
    Modelo BioBERT para extração de embeddings de narrativas clínicas.
    
    Características:
    - Usa transformers pré-treinados em textos biomédicos
    - Extrai embeddings contextualizados de alta dimensionalidade
    - Reduz dimensionalidade para features práticas
    - Compatível com o pipeline NLP
    
    Attributes:
        model_name: Nome do modelo BioBERT
        tokenizer: Tokenizador do modelo
        model: Modelo transformers
        max_length: Comprimento máximo de tokens
        output_dir: Diretório para salvar modelos
    """
    
    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-v1.1",
        max_length: int = 512,
        output_dir: str = 'results/nlp/biobert'
    ):
        """
        Inicializa o modelo BioBERT.
        
        Parâmetros:
        -----------
        model_name : str
            Nome do modelo BioBERT no HuggingFace
            Opções: "dmis-lab/biobert-v1.1", "dmis-lab/biobert-base-cased-v1.1"
        max_length : int
            Comprimento máximo de tokens (padrão: 512)
        output_dir : str
            Diretório para salvar modelos e resultados
        """
        self.model_name = model_name
        self.max_length = max_length
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.device = None
        
        logger.info(f"BioBERTModel inicializado com modelo: {model_name}")
        
        # Tentar carregar transformers
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            self.torch = torch
            self.AutoTokenizer = AutoTokenizer
            self.AutoModel = AutoModel
            
            # Detectar device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Device: {self.device}")
            
            # Carregar tokenizador e modelo
            self._load_model()
            
        except ImportError:
            logger.warning("PyTorch/Transformers não disponível. BioBERT funcionará em modo simulado.")
            self.torch = None
    
    def _load_model(self):
        """Carrega o modelo BioBERT pré-treinado."""
        try:
            logger.info(f"Carregando modelo BioBERT: {self.model_name}")
            
            self.tokenizer = self.AutoTokenizer.from_pretrained(self.model_name)
            self.model = self.AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Modelo BioBERT carregado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar BioBERT: {e}")
            logger.warning("Continuando em modo simulado")
            self.tokenizer = None
            self.model = None
    
    def extract_embeddings(
        self,
        texts: List[str],
        use_pooled: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extrai embeddings de narrativas clínicas usando BioBERT.
        
        Parâmetros:
        -----------
        texts : List[str]
            Lista de narrativas clínicas
        use_pooled : bool
            Se True, usa pooled output (representação da sentença inteira)
            Se False, usa last hidden state (representação token-a-token)
        batch_size : int
            Tamanho do batch para processamento
            
        Retorna:
        --------
        np.ndarray
            Matriz de embeddings (n_samples, embedding_dim)
        """
        if self.model is None:
            logger.warning("Modelo não disponível, retornando embeddings simulados")
            return self._get_simulated_embeddings(texts)
        
        logger.info(f"Extraindo embeddings de {len(texts)} textos com BioBERT")
        
        embeddings = []
        
        # Processar em batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenizar
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Mover para device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extrair embeddings
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            
            # Selecionar tipo de output
            if use_pooled:
                # Usar pooled output (CLS token)
                batch_embeddings = outputs.pooler_output.cpu().numpy()
            else:
                # Usar média de todos os tokens
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        # Concatenar todos os batches
        embeddings = np.vstack(embeddings)
        
        logger.info(f"✅ Embeddings extraídos: shape {embeddings.shape}")
        
        return embeddings
    
    def _get_simulated_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Retorna embeddings simulados quando BioBERT não está disponível.
        
        Parâmetros:
        -----------
        texts : List[str]
            Lista de textos
            
        Retorna:
        --------
        np.ndarray
            Embeddings simulados (n_samples, 768)
        """
        logger.info(f"Gerando embeddings simulados para {len(texts)} textos")
        
        # Usar hash do texto para gerar embeddings determinísticos
        embeddings = []
        
        for text in texts:
            # Usar hash do texto como seed
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            
            # Gerar embedding de 768 dimensões (tamanho padrão do BERT)
            embedding = np.random.randn(768) * 0.1
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def reduce_embeddings(
        self,
        embeddings: np.ndarray,
        n_components: int = 50,
        method: str = 'pca'
    ) -> np.ndarray:
        """
        Reduz dimensionalidade dos embeddings para uso prático.
        
        Parâmetros:
        -----------
        embeddings : np.ndarray
            Embeddings originais (n_samples, 768)
        n_components : int
            Número de componentes para redução
        method : str
            Método de redução ('pca', 'tsne', 'umap')
            
        Retorna:
        --------
        np.ndarray
            Embeddings reduzidos (n_samples, n_components)
        """
        logger.info(f"Reduzindo embeddings de {embeddings.shape[1]} para {n_components} dimensões ({method})")
        
        try:
            if method == 'pca':
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(embeddings)
                
            elif method == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
                reduced = reducer.fit_transform(embeddings)
                
            elif method == 'umap':
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(embeddings)
                
            else:
                raise ValueError(f"Método inválido: {method}")
            
            logger.info(f"✅ Embeddings reduzidos: shape {reduced.shape}")
            
            return reduced
            
        except ImportError as e:
            logger.error(f"Erro ao importar {method}: {e}")
            logger.warning("Retornando embeddings originais")
            return embeddings[:, :n_components] if embeddings.shape[1] > n_components else embeddings
    
    def extract_clinical_entities(
        self,
        texts: List[str]
    ) -> Dict[int, List[Dict[str, str]]]:
        """
        Extrai entidades clínicas das narrativas.
        
        Parâmetros:
        -----------
        texts : List[str]
            Lista de narrativas clínicas
            
        Retorna:
        --------
        Dict[int, List[Dict]]
            Dicionário com entidades por documento
        """
        logger.info(f"Extraindo entidades clínicas de {len(texts)} textos")
        
        entities_dict = {}
        
        # Palavras-chave clínicas para TB
        clinical_keywords = {
            'comorbidades': ['hiv', 'aids', 'diabetes', 'doença mental', 'tuberculose', 'tb'],
            'fatores_sociais': ['rua', 'álcool', 'drogas', 'desemprego', 'pobreza'],
            'sintomas': ['tosse', 'febre', 'fraqueza', 'emagrecimento', 'hemoptise'],
            'tratamento': ['rifampicina', 'isoniazida', 'pirazinamida', 'etambutol', 'medicação'],
            'desfecho': ['cura', 'abandono', 'óbito', 'transferência', 'falha']
        }
        
        for doc_id, text in enumerate(texts):
            text_lower = text.lower()
            entities = []
            
            for category, keywords in clinical_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        entities.append({
                            'category': category,
                            'entity': keyword,
                            'position': text_lower.find(keyword)
                        })
            
            entities_dict[doc_id] = entities
        
        logger.info(f"✅ Entidades extraídas: {sum(len(e) for e in entities_dict.values())} total")
        
        return entities_dict
    
    def get_embeddings_statistics(
        self,
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calcula estatísticas dos embeddings.
        
        Parâmetros:
        -----------
        embeddings : np.ndarray
            Matriz de embeddings
            
        Retorna:
        --------
        Dict[str, Any]
            Estatísticas dos embeddings
        """
        stats = {
            'shape': embeddings.shape,
            'mean': embeddings.mean(axis=0).mean(),
            'std': embeddings.std(axis=0).mean(),
            'min': embeddings.min(),
            'max': embeddings.max(),
            'norm_mean': np.linalg.norm(embeddings, axis=1).mean(),
            'norm_std': np.linalg.norm(embeddings, axis=1).std()
        }
        
        logger.info(f"Estatísticas dos embeddings: {stats}")
        
        return stats
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filename: str = 'biobert_embeddings.npy'
    ) -> str:
        """
        Salva embeddings em arquivo.
        
        Parâmetros:
        -----------
        embeddings : np.ndarray
            Matriz de embeddings
        filename : str
            Nome do arquivo
            
        Retorna:
        --------
        str
            Caminho do arquivo salvo
        """
        filepath = self.output_dir / filename
        np.save(filepath, embeddings)
        logger.info(f"✅ Embeddings salvos em: {filepath}")
        return str(filepath)
    
    def load_embeddings(self, filename: str = 'biobert_embeddings.npy') -> np.ndarray:
        """
        Carrega embeddings de arquivo.
        
        Parâmetros:
        -----------
        filename : str
            Nome do arquivo
            
        Retorna:
        --------
        np.ndarray
            Matriz de embeddings
        """
        filepath = self.output_dir / filename
        embeddings = np.load(filepath)
        logger.info(f"✅ Embeddings carregados de: {filepath}")
        return embeddings


def train_biobert_pipeline(
    texts: List[str],
    output_dir: str = 'results/nlp/biobert',
    reduce_dim: bool = True,
    n_components: int = 50
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pipeline completo de extração de features com BioBERT.
    
    Parâmetros:
    -----------
    texts : List[str]
        Lista de narrativas clínicas
    output_dir : str
        Diretório para salvar resultados
    reduce_dim : bool
        Se True, reduz dimensionalidade
    n_components : int
        Número de componentes para redução
        
    Retorna:
    --------
    Tuple[np.ndarray, Dict]
        (embeddings, metadata)
    """
    logger.info("Iniciando pipeline BioBERT")
    
    # Inicializar modelo
    biobert = BioBERTModel(output_dir=output_dir)
    
    # Extrair embeddings
    embeddings = biobert.extract_embeddings(texts)
    
    # Reduzir dimensionalidade
    if reduce_dim:
        embeddings_reduced = biobert.reduce_embeddings(embeddings, n_components=n_components)
    else:
        embeddings_reduced = embeddings
    
    # Extrair entidades
    entities = biobert.extract_clinical_entities(texts)
    
    # Calcular estatísticas
    stats = biobert.get_embeddings_statistics(embeddings_reduced)
    
    # Salvar embeddings
    biobert.save_embeddings(embeddings_reduced, 'biobert_embeddings_reduced.npy')
    
    # Metadados
    metadata = {
        'n_texts': len(texts),
        'embedding_dim': embeddings_reduced.shape[1],
        'original_dim': embeddings.shape[1],
        'n_entities': sum(len(e) for e in entities.values()),
        'statistics': stats
    }
    
    logger.info(f"✅ Pipeline BioBERT concluído")
    logger.info(f"   Textos: {metadata['n_texts']}")
    logger.info(f"   Dimensão de embeddings: {metadata['embedding_dim']}")
    logger.info(f"   Entidades extraídas: {metadata['n_entities']}")
    
    return embeddings_reduced, metadata


# Exemplo de uso
if __name__ == "__main__":
    # Textos de exemplo
    texts = [
        "Paciente com tuberculose pulmonar, 45 anos, masculino. Apresenta HIV/AIDS. "
        "Histórico de uso de álcool. Risco elevado de abandono.",
        
        "Diagnóstico de TB em paciente de 32 anos. Sem comorbidades. "
        "Contexto social estável. Baixo risco de abandono.",
        
        "Caso de tuberculose em paciente feminino de 28 anos. "
        "Diabetes e doença mental. Situação de rua. Necessita acompanhamento intensivo."
    ]
    
    # Executar pipeline
    embeddings, metadata = train_biobert_pipeline(texts, reduce_dim=True, n_components=50)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Metadata: {metadata}")
