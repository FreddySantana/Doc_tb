"""
Módulo para geração de narrativas clínicas sintéticas

Autor: Frederico
Instituição: Programa de Doutorado
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2025-07-10
Última Modificação: 2025-09-10

Descrição:
    Este módulo faz parte do framework multi-paradigma desenvolvido para predição
    de abandono de tratamento em pacientes com tuberculose. Implementa geração
    de narrativas clínicas sintéticas usando modelos de linguagem para enriquecer
    o dataset com informações textuais.

Licença: MIT
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SyntheticNarrativeGenerator:
    """
    Gerador de narrativas clínicas sintéticas para pacientes com tuberculose.
    
    Cria textos clínicos baseados em características dos pacientes para
    enriquecer o dataset com informações textuais que podem ser processadas
    por modelos de NLP.
    
    Escalável para processar toda a base de 106.759 pacientes (2006-2016).
    
    Attributes:
        templates: Dicionário de templates de narrativas
        output_dir: Diretório para salvar narrativas geradas
    """
    
    def __init__(self, output_dir: str = 'results/nlp/narratives'):
        """
        Inicializa o gerador de narrativas.
        
        Args:
            output_dir: Diretório para salvar resultados
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Templates de narrativas por perfil de risco
        self.templates = self._load_templates()
        
        logger.info('SyntheticNarrativeGenerator inicializado')
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """
        Carrega templates de narrativas clínicas.
        
        Returns:
            Dicionário com templates por categoria
        """
        templates = {
            'high_risk': [
                "Paciente {idade} anos, {sexo}, diagnóstico de tuberculose {tipo_tb}. "
                "Apresenta comorbidade de {comorbidade}. Histórico de {social_factor}. "
                "Risco elevado de abandono devido a múltiplos fatores sociais e clínicos.",
                
                "Caso de TB {tipo_tb} em paciente {sexo} de {idade} anos. "
                "Condição agravada por {comorbidade}. Situação social: {social_factor}. "
                "Necessita acompanhamento intensivo para prevenir abandono.",
                
                "Paciente com tuberculose {tipo_tb}, {idade} anos. "
                "Fatores de risco: {comorbidade} e {social_factor}. "
                "Perfil de alto risco para não adesão ao tratamento."
            ],
            
            'medium_risk': [
                "Paciente {sexo}, {idade} anos, TB {tipo_tb}. "
                "Apresenta {comorbidade}. {social_factor}. "
                "Risco moderado de abandono, requer monitoramento regular.",
                
                "Diagnóstico de tuberculose {tipo_tb} em paciente de {idade} anos. "
                "Condição clínica: {comorbidade}. Contexto social: {social_factor}. "
                "Acompanhamento necessário para garantir adesão.",
                
                "Caso de TB {tipo_tb}, paciente {sexo} com {idade} anos. "
                "{comorbidade}. {social_factor}. Prognóstico favorável com suporte adequado."
            ],
            
            'low_risk': [
                "Paciente {idade} anos, {sexo}, TB {tipo_tb}. "
                "Sem comorbidades significativas. Contexto social estável. "
                "Baixo risco de abandono, prognóstico favorável.",
                
                "Diagnóstico de tuberculose {tipo_tb} em paciente de {idade} anos. "
                "Condições clínicas e sociais favoráveis. "
                "Alta probabilidade de conclusão do tratamento.",
                
                "Caso de TB {tipo_tb}, {sexo}, {idade} anos. "
                "Perfil de baixo risco. Adesão esperada ao protocolo terapêutico."
            ]
        }
        
        return templates
    
    def _classify_risk(self, row: pd.Series) -> str:
        """
        Classifica o nível de risco do paciente.
        
        Args:
            row: Linha do DataFrame com dados do paciente
        
        Returns:
            Categoria de risco ('high_risk', 'medium_risk', 'low_risk')
        """
        risk_score = 0
        
        # Fatores de risco clínicos
        if row.get('hiv_aids', 0) == 1:
            risk_score += 3
        if row.get('diabetes', 0) == 1:
            risk_score += 1
        if row.get('doenca_mental', 0) == 1:
            risk_score += 2
        
        # Fatores de risco sociais
        if row.get('situacao_rua', 0) == 1:
            risk_score += 3
        if row.get('uso_alcool', 0) == 1:
            risk_score += 2
        if row.get('uso_drogas', 0) == 1:
            risk_score += 2
        
        # Classificação
        if risk_score >= 5:
            return 'high_risk'
        elif risk_score >= 2:
            return 'medium_risk'
        else:
            return 'low_risk'
    
    def _get_comorbidity_text(self, row: pd.Series) -> str:
        """
        Gera texto descritivo de comorbidades.
        
        Args:
            row: Linha do DataFrame
        
        Returns:
            Texto descrevendo comorbidades
        """
        comorbidities = []
        
        if row.get('hiv_aids', 0) == 1:
            comorbidities.append('HIV/AIDS')
        if row.get('diabetes', 0) == 1:
            comorbidities.append('diabetes')
        if row.get('doenca_mental', 0) == 1:
            comorbidities.append('doença mental')
        
        if comorbidities:
            return ', '.join(comorbidities)
        else:
            return 'sem comorbidades registradas'
    
    def _get_social_factor_text(self, row: pd.Series) -> str:
        """
        Gera texto descritivo de fatores sociais.
        
        Args:
            row: Linha do DataFrame
        
        Returns:
            Texto descrevendo fatores sociais
        """
        factors = []
        
        if row.get('situacao_rua', 0) == 1:
            factors.append('situação de rua')
        if row.get('uso_alcool', 0) == 1:
            factors.append('uso de álcool')
        if row.get('uso_drogas', 0) == 1:
            factors.append('uso de drogas')
        
        if factors:
            return ', '.join(factors)
        else:
            return 'contexto social estável'
    
    def generate_narrative(self, row: pd.Series) -> str:
        """
        Gera uma narrativa clínica para um paciente.
        
        Args:
            row: Linha do DataFrame com dados do paciente
        
        Returns:
            Narrativa clínica sintética
        """
        # Classificar risco
        risk_category = self._classify_risk(row)
        
        # Selecionar template aleatório
        template = np.random.choice(self.templates[risk_category])
        
        # Preencher template
        narrative = template.format(
            idade=int(row.get('idade', 40)),
            sexo='masculino' if row.get('sexo', 'M') == 'M' else 'feminino',
            tipo_tb=row.get('tipo_tb', 'pulmonar'),
            comorbidade=self._get_comorbidity_text(row),
            social_factor=self._get_social_factor_text(row)
        )
        
        return narrative
    
    def generate_narratives(
        self,
        df: pd.DataFrame,
        n_narratives: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Gera narrativas para um conjunto de pacientes.
        
        Args:
            df: DataFrame com dados dos pacientes
            n_narratives: Número de narrativas a gerar (None = todas)
        
        Returns:
            DataFrame com narrativas adicionadas
        """
        logger.info(f'Gerando narrativas clínicas sintéticas')
        
        df_with_narratives = df.copy()
        
        # Selecionar subset se especificado
        if n_narratives is not None:
            df_subset = df_with_narratives.sample(n=min(n_narratives, len(df)))
        else:
            df_subset = df_with_narratives
        
        # Gerar narrativas
        narratives = []
        for idx, row in df_subset.iterrows():
            narrative = self.generate_narrative(row)
            narratives.append(narrative)
        
        df_with_narratives.loc[df_subset.index, 'narrativa_clinica'] = narratives
        
        logger.info(f'Total de narrativas geradas: {len(narratives)}')
        
        # Salvar narrativas
        self._save_narratives(narratives)
        
        return df_with_narratives
    
    def _save_narratives(self, narratives: List[str]) -> None:
        """
        Salva narrativas geradas em arquivo.
        
        Args:
            narratives: Lista de narrativas
        """
        output_file = self.output_dir / 'synthetic_narratives.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total': len(narratives),
                'narratives': narratives
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f'Narrativas salvas em {output_file}')
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcula estatísticas sobre as narrativas geradas.
        
        Args:
            df: DataFrame com narrativas
        
        Returns:
            Dicionário com estatísticas
        """
        if 'narrativa_clinica' not in df.columns:
            return {}
        
        narratives = df['narrativa_clinica'].dropna()
        
        stats = {
            'total_narratives': len(narratives),
            'avg_length': narratives.str.len().mean(),
            'min_length': narratives.str.len().min(),
            'max_length': narratives.str.len().max(),
            'avg_words': narratives.str.split().str.len().mean()
        }
        
        logger.info(f'Estatísticas das narrativas: {stats}')
        
        return stats
