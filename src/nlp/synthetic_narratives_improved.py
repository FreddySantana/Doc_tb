"""
Módulo Melhorado para Geração de Narrativas Clínicas Sintéticas

Autor: Frederico Guilheme Santana da Silva Filho
Instituição: Programa de Doutorado em Engenharia Elétrica - UFPA
Projeto: Framework Multi-Paradigma para Predição de Abandono de Tratamento de Tuberculose

Data de Criação: 2024-08-05
Última Modificação: 2025-11-30

Descrição:
    Versão melhorada do gerador de narrativas que:
    1. Remove aleatoriedade desnecessária (usa determinismo com seed)
    2. Gera narrativas mais realistas e clinicamente apropriadas
    3. Permite integração com dados reais quando disponíveis
    4. Mantém consistência entre execuções

Licença: MIT
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ImprovedSyntheticNarrativeGenerator:
    """
    Gerador melhorado de narrativas clínicas sintéticas.
    
    Características:
    - Determinístico (sem aleatoriedade desnecessária)
    - Narrativas mais realistas e clinicamente apropriadas
    - Suporta dados reais quando disponíveis
    - Mantém rastreabilidade (seed-based)
    
    Attributes:
        templates: Dicionário de templates de narrativas
        output_dir: Diretório para salvar narrativas
        use_real_data: Se True, usa dados reais quando disponíveis
    """
    
    def __init__(
        self,
        output_dir: str = 'results/nlp/narratives',
        use_real_data: bool = False,
        seed: int = 42
    ):
        """
        Inicializa o gerador melhorado de narrativas.
        
        Parâmetros:
        -----------
        output_dir : str
            Diretório para salvar narrativas
        use_real_data : bool
            Se True, tenta usar dados reais quando disponíveis
        seed : int
            Seed para determinismo
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_real_data = use_real_data
        self.seed = seed
        np.random.seed(seed)
        
        # Templates melhorados (mais realistas)
        self.templates = self._load_improved_templates()
        
        logger.info(f"ImprovedSyntheticNarrativeGenerator inicializado (seed={seed})")
    
    def _load_improved_templates(self) -> Dict[str, List[str]]:
        """
        Carrega templates melhorados de narrativas clínicas.
        
        Retorna:
        --------
        Dict[str, List[str]]
            Dicionário com templates por categoria de risco
        """
        templates = {
            'high_risk': [
                # Template 1: Paciente com múltiplas comorbidades
                "Paciente {idade} anos, {sexo}. Diagnóstico de tuberculose {tipo_tb}. "
                "Apresenta comorbidade significativa: {comorbidade}. "
                "Contexto social adverso: {social_factor}. "
                "Avaliação inicial: Paciente apresenta múltiplos fatores de risco para abandono do tratamento. "
                "Necessário acompanhamento intensivo e suporte multidisciplinar. "
                "Prognóstico: Risco elevado sem intervenção.",
                
                # Template 2: Paciente com história de abandono
                "Caso de tuberculose {tipo_tb} em paciente {sexo} de {idade} anos. "
                "Comorbidade associada: {comorbidade}. "
                "Fator social relevante: {social_factor}. "
                "Histórico: Paciente com perfil de não-adesão ao tratamento. "
                "Recomendação: Implementar estratégias de retenção e acompanhamento domiciliar. "
                "Risco de abandono: ALTO.",
                
                # Template 3: Paciente vulnerável
                "Paciente {sexo}, {idade} anos, TB {tipo_tb}. "
                "Condição clínica: {comorbidade}. "
                "Situação social: {social_factor}. "
                "Análise: Paciente em situação de vulnerabilidade social e clínica. "
                "Necessário envolvimento de assistência social e saúde mental. "
                "Classificação de risco: ALTO para abandono."
            ],
            
            'medium_risk': [
                # Template 1: Paciente com risco moderado
                "Paciente {sexo}, {idade} anos, TB {tipo_tb}. "
                "Comorbidade: {comorbidade}. "
                "Fator social: {social_factor}. "
                "Avaliação: Paciente apresenta alguns fatores de risco moderados. "
                "Recomendação: Acompanhamento regular com reforço educativo. "
                "Risco de abandono: MODERADO.",
                
                # Template 2: Paciente com adesão questionável
                "Diagnóstico de tuberculose {tipo_tb} em paciente de {idade} anos. "
                "Condição clínica: {comorbidade}. "
                "Contexto social: {social_factor}. "
                "Observação: Paciente com adesão questionável ao tratamento. "
                "Necessário monitoramento regular e reforço de orientações. "
                "Prognóstico: Favorável com suporte adequado.",
                
                # Template 3: Paciente estável com fatores de risco
                "Caso de TB {tipo_tb}, paciente {sexo} com {idade} anos. "
                "{comorbidade}. {social_factor}. "
                "Avaliação clínica: Paciente clinicamente estável. "
                "Fatores de risco presentes mas controláveis. "
                "Recomendação: Acompanhamento mensal e reforço de adesão."
            ],
            
            'low_risk': [
                # Template 1: Paciente de baixo risco
                "Paciente {idade} anos, {sexo}, TB {tipo_tb}. "
                "Sem comorbidades significativas. "
                "Contexto social estável. "
                "Avaliação: Paciente em bom estado geral com suporte social adequado. "
                "Prognóstico: Favorável. Baixo risco de abandono. "
                "Recomendação: Acompanhamento padrão.",
                
                # Template 2: Paciente com bom prognóstico
                "Diagnóstico de tuberculose {tipo_tb} em paciente de {idade} anos. "
                "Condições clínicas e sociais favoráveis. "
                "Sem comorbidades relevantes. "
                "Avaliação: Paciente com alta probabilidade de conclusão do tratamento. "
                "Risco de abandono: BAIXO. "
                "Adesão esperada: Excelente.",
                
                # Template 3: Paciente com suporte adequado
                "Caso de TB {tipo_tb}, {sexo}, {idade} anos. "
                "Perfil de baixo risco. "
                "Suporte familiar e social presente. "
                "Avaliação: Paciente com fatores protetores bem estabelecidos. "
                "Recomendação: Acompanhamento padrão com reforço trimestral."
            ]
        }
        
        return templates
    
    def _classify_risk_deterministic(self, row: pd.Series) -> str:
        """
        Classifica risco de forma determinística (sem aleatoriedade).
        
        Parâmetros:
        -----------
        row : pd.Series
            Linha do DataFrame com dados do paciente
            
        Retorna:
        --------
        str
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
        
        # Classificação determinística
        if risk_score >= 5:
            return 'high_risk'
        elif risk_score >= 2:
            return 'medium_risk'
        else:
            return 'low_risk'
    
    def _get_comorbidity_text(self, row: pd.Series) -> str:
        """
        Gera texto descritivo de comorbidades.
        
        Parâmetros:
        -----------
        row : pd.Series
            Linha do DataFrame
            
        Retorna:
        --------
        str
            Texto descrevendo comorbidades
        """
        comorbidities = []
        
        if row.get('hiv_aids', 0) == 1:
            comorbidities.append('HIV/AIDS')
        if row.get('diabetes', 0) == 1:
            comorbidities.append('diabetes mellitus tipo 2')
        if row.get('doenca_mental', 0) == 1:
            comorbidities.append('transtorno mental')
        
        if comorbidities:
            return ', '.join(comorbidities)
        else:
            return 'sem comorbidades registradas'
    
    def _get_social_factor_text(self, row: pd.Series) -> str:
        """
        Gera texto descritivo de fatores sociais.
        
        Parâmetros:
        -----------
        row : pd.Series
            Linha do DataFrame
            
        Retorna:
        --------
        str
            Texto descrevendo fatores sociais
        """
        factors = []
        
        if row.get('situacao_rua', 0) == 1:
            factors.append('situação de rua')
        if row.get('uso_alcool', 0) == 1:
            factors.append('uso de álcool')
        if row.get('uso_drogas', 0) == 1:
            factors.append('uso de drogas ilícitas')
        
        if factors:
            return ', '.join(factors)
        else:
            return 'contexto social estável'
    
    def generate_narrative_deterministic(
        self,
        row: pd.Series,
        template_index: Optional[int] = None
    ) -> str:
        """
        Gera narrativa de forma determinística.
        
        Parâmetros:
        -----------
        row : pd.Series
            Linha do DataFrame com dados do paciente
        template_index : int, optional
            Índice do template (None = usar índice determinístico)
            
        Retorna:
        --------
        str
            Narrativa clínica sintética
        """
        # Classificar risco
        risk_category = self._classify_risk_deterministic(row)
        
        # Selecionar template de forma determinística
        if template_index is None:
            # Usar hash do paciente para determinismo
            patient_id = row.get('id', 0)
            template_index = hash(patient_id) % len(self.templates[risk_category])
        
        template = self.templates[risk_category][template_index]
        
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
        n_narratives: Optional[int] = None,
        use_deterministic: bool = True
    ) -> pd.DataFrame:
        """
        Gera narrativas para um conjunto de pacientes.
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com dados dos pacientes
        n_narratives : int, optional
            Número de narrativas a gerar (None = todas)
        use_deterministic : bool
            Se True, usa geração determinística
            
        Retorna:
        --------
        pd.DataFrame
            DataFrame com narrativas adicionadas
        """
        logger.info(f"Gerando narrativas clínicas sintéticas (determinístico={use_deterministic})")
        
        df_with_narratives = df.copy()
        
        # Selecionar subset se especificado
        if n_narratives is not None:
            df_subset = df_with_narratives.sample(n=min(n_narratives, len(df)), random_state=self.seed)
        else:
            df_subset = df_with_narratives
        
        # Gerar narrativas
        narratives = []
        for idx, row in df_subset.iterrows():
            if use_deterministic:
                narrative = self.generate_narrative_deterministic(row)
            else:
                narrative = self.generate_narrative_deterministic(row)  # Sempre determinístico
            
            narratives.append(narrative)
        
        df_with_narratives.loc[df_subset.index, 'narrativa_clinica'] = narratives
        
        logger.info(f"✅ Total de narrativas geradas: {len(narratives)}")
        
        # Salvar narrativas
        self._save_narratives(narratives, df_subset)
        
        return df_with_narratives
    
    def _save_narratives(
        self,
        narratives: List[str],
        df_subset: pd.DataFrame
    ) -> None:
        """
        Salva narrativas geradas em arquivo.
        
        Parâmetros:
        -----------
        narratives : List[str]
            Lista de narrativas
        df_subset : pd.DataFrame
            Subset do DataFrame com metadados
        """
        output_file = self.output_dir / 'synthetic_narratives_improved.json'
        
        # Preparar dados para salvar
        data = {
            'total': len(narratives),
            'seed': self.seed,
            'deterministic': True,
            'narratives': narratives,
            'metadata': {
                'n_high_risk': (df_subset.get('risco', 'unknown') == 'high').sum() if 'risco' in df_subset.columns else 'unknown',
                'n_medium_risk': (df_subset.get('risco', 'unknown') == 'medium').sum() if 'risco' in df_subset.columns else 'unknown',
                'n_low_risk': (df_subset.get('risco', 'unknown') == 'low').sum() if 'risco' in df_subset.columns else 'unknown'
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Narrativas salvas em {output_file}")
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcula estatísticas sobre as narrativas geradas.
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame com narrativas
            
        Retorna:
        --------
        Dict[str, Any]
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
            'avg_words': narratives.str.split().str.len().mean(),
            'unique_narratives': narratives.nunique()
        }
        
        logger.info(f"Estatísticas das narrativas: {stats}")
        
        return stats


# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'idade': [45, 32, 28, 55, 40],
        'sexo': ['M', 'M', 'F', 'M', 'F'],
        'tipo_tb': ['pulmonar', 'pulmonar', 'extrapulmonar', 'pulmonar', 'pulmonar'],
        'hiv_aids': [1, 0, 0, 0, 0],
        'diabetes': [0, 0, 1, 1, 0],
        'doenca_mental': [0, 0, 0, 0, 1],
        'situacao_rua': [1, 0, 0, 0, 0],
        'uso_alcool': [1, 0, 1, 0, 0],
        'uso_drogas': [0, 0, 0, 0, 0]
    })
    
    # Gerar narrativas
    gen = ImprovedSyntheticNarrativeGenerator(use_real_data=False, seed=42)
    df_narrativas = gen.generate_narratives(df)
    
    print("\nNarrativas Geradas:")
    for idx, row in df_narrativas.iterrows():
        print(f"\nPaciente {row['id']}:")
        print(row['narrativa_clinica'])
    
    # Estatísticas
    stats = gen.get_statistics(df_narrativas)
    print(f"\nEstatísticas: {stats}")
