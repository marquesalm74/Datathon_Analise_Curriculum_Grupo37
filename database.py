import os
import json
import re
import unidecode
import pandas as pd

from dotenv import load_dotenv
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supabase_client import supabase

# Dados tratados locais (certifique-se que tratarbase.py não importa database)
from tratarbase import applicants, perfil_vagas_tratada, prospects_tratada
from models.analysis import Analysis  # usado no insert_analysis

load_dotenv()

# Para ativar Supabase no futuro, use algo assim:
from supabase_client import supabase

class AnalyseDatabase:
    def __init__(self):
        self.applicants = applicants
        self.vagas = perfil_vagas_tratada
        self.prospects = prospects_tratada
        self.model = SentenceTransformer("all-MiniLM-L6-v2") # Inicializa modelo SentenceTransformer só uma vez

    ####################### Applicants ##################################
    #
    def get_applicants(self, codigo_vaga: str) -> List[Dict]:
        """Retorna todos os candidatos vinculados à vaga."""
        aplicados = self.applicants[self.applicants["codigo_profissional"] == codigo_vaga]
        return aplicados.to_dict(orient="records")

    def get_all_applicants(self) -> List[Dict]:
        """Retorna todos os applicants da base."""
        return self.applicants.to_dict(orient="records")
    
    ###################### Prospects ####################################
    #
    def get_prospects(self, codigo_vaga: str) -> List[Dict]:
        """Retorna todos os prospects vinculados à vaga."""
        df = self.prospects
        return df[df['codigo_vaga'] == codigo_vaga].to_dict(orient="records")
    
    #####################   Vagas #######################################
    #
    def get_all_vagas(self) -> List[Dict]:
        """Retorna todas as vagas disponíveis."""
        return self.vagas.to_dict(orient="records")

    def get_vaga_by_codigo(self, codigo_vaga: str) -> Optional[Dict]:
        """Retorna uma vaga específica com base no código."""
        resultado = self.vagas[self.vagas["codigo_vaga"] == codigo_vaga]
        if resultado.empty:
            return None
        return resultado.iloc[0].to_dict()
    
    ####################################### Candidatos Compativeis
    #
    def get_candidatos_compativeis_por_titulo(self, titulo_vaga: str) -> pd.DataFrame:
        """
        Retorna DataFrame de candidatos ordenados pela similaridade com o título da vaga.
        """
        # 1. Buscar todos os candidatos
        candidatos = self.get_all_applicants()  # lista de dicts

        # 2. Gerar embeddings e calcular similaridade (retorna lista de dicts)
        resultados = self.get_embedding_applicants(titulo_vaga, candidatos)

        # 3. Converter para DataFrame
        df_resultados = pd.DataFrame(resultados)

        # 4. Ordenar os resultados pela similaridade
        resultados_ordenados = df_resultados.sort_values(by="score_similaridade", ascending=False)

        # 5. Retornar o DataFrame ordenado
        return resultados_ordenados

    
    def get_embedding_applicants(self, titulo_vaga: str, candidatos: list[dict]) -> list[dict]:
        """
        Recebe um título de vaga e uma lista de candidatos (lista de dicts),
        retorna os candidatos com o score de similaridade (0-1) adicionado.
        """
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Converte para DataFrame
        df_applicants = pd.DataFrame(candidatos)

        # Embeddings
        embeddings_applicants = model.encode(df_applicants["texto_cv"].tolist(), convert_to_tensor=False)
        embedding_vaga = model.encode(titulo_vaga, convert_to_tensor=False)
        # codigo retirado para tratarbase.py limpar
        # Similaridade
        scores = cosine_similarity([embedding_vaga], embeddings_applicants)[0]
        df_applicants["score_similaridade"] = scores

        # Retorna lista de dicionários incluindo score_similaridade
        return df_applicants

    def insert_analysis(self, analysis_obj: Analysis):
        """Mock de inserção da análise. Pode ser adaptado ao Supabase futuramente."""
        print(f"🔄 Mock insert: {analysis_obj.codigo_vaga} - {analysis_obj.file}")

    # Métodos para Supabase futuramente:
    def inserir_applicant_supabase(self, dados: Dict) -> None:
         supabase.table("applicants").insert(dados).execute()
    
    def atualizar_applicant_supabase(self, dados: Dict) -> None:
         supabase.table("applicants").update(dados).eq("id", dados["id"]).execute()
    
    def get_all_applicants_supabase(self) -> List[Dict]:
         response = supabase.table("applicants").select("*").execute()
         return response.data if response.data else []

    ############################ Atualizar no Supabase e df_applicants
    #
    def atualizar_applicant_supabase(self, dados: Dict) -> None:
        if "id" not in dados:
            raise ValueError("O campo 'id' é obrigatório.")
        response = supabase.table("applicants").update(dados).eq("id", dados["id"]).execute()
        if response.error:
            raise Exception(f"Erro no Supabase: {response.error.message}")

        idx = self.applicants.index[self.applicants['id'] == dados["id"]]
        if not idx.empty:
            for key, value in dados.items():
                self.applicants.loc[idx, key] = value

        return response.data

    #
    def inserir_applicant_supabase(self, dados: Dict) -> None:
        """
        Insere um novo registro na tabela 'applicants' no Supabase.

        Parâmetros:
            dados: dict com os dados do applicant a serem inseridos.
        """
        response = supabase.table("applicants").insert(dados).execute()
        
        if response.error:
            raise Exception(f"Erro ao inserir no Supabase: {response.error.message}")

        # Atualize o DataFrame local adicionando o novo registro 
        novo_registro = pd.DataFrame([dados])
        self.applicants = pd.concat([self.applicants, novo_registro], ignore_index=True)
        
        return response.data
    
    # Suponha que applicants seja seu DataFrame global importado de tratarbase.py
    from tratarbase import applicants

    def inserir_applicant_novo(self,dados: dict):
        """
        Insere novo registro no DataFrame global applicants.
        Se o ID já existir, avisa que é duplicado e não insere.
        """
        global applicants

        if 'id' not in dados or not dados['id']:
            raise ValueError("O campo 'id' é obrigatório para inserir.")

        if dados['id'] in applicants['id'].astype(str).values:
            raise ValueError(f"Já existe um candidato com id={dados['id']}")

        # Transforma dict em DataFrame de 1 linha
        df_novo = pd.DataFrame([dados])

        # Append ao DataFrame global
        applicants = pd.concat([applicants, df_novo], ignore_index=True)

    def atualizar_applicant(self,dados: dict):
        """
        Atualiza registro existente no DataFrame global applicants pelo campo 'id'.
        Se não encontrar o id, lança erro.
        """
        global applicants

        if 'id' not in dados or not dados['id']:
            raise ValueError("O campo 'id' é obrigatório para atualizar.")

        id_str = str(dados['id'])

        idxs = applicants.index[applicants['id'].astype(str) == id_str].tolist()
        if not idxs:
            raise ValueError(f"Nenhum candidato encontrado com id={id_str} para atualizar.")

        idx = idxs[0]

        # Atualiza linha inteira (mantendo a ordem das colunas do DataFrame)
        for coluna in dados:
            if coluna in applicants.columns:
                applicants.at[idx, coluna] = dados[coluna]
            else:
                # Se a coluna não existe, você pode optar por ignorar ou criar nova coluna:
                applicants[coluna] = None  # criar coluna vazia antes
                applicants.at[idx, coluna] = dados[coluna]
                pass 

#################################################################################
#
#import os
#import pandas as pd
#from typing import List, Dict, Optional
#from dotenv import load_dotenv
#from sentence_transformers import SentenceTransformer
#from sklearn.metrics.pairwise import cosine_similarity
#from supabase_client import supabase
#from models.analysis import Analysis

#load_dotenv()

#class AnalyseDatabase:
#    def __init__(self):
#        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    ########################## Applicants ##############################

    #def get_applicants(self, codigo_vaga: str) -> List[Dict]:
    #    """Retorna todos os candidatos vinculados à vaga sem paginação."""
    #    response = supabase.table("applicants")\
    #        .select("*")\
    #        .eq("codigo_profissional", codigo_vaga)\
    #        .limit(50000).execute()

    #    return response.data if response.data else []
    
#    def get_applicants(self, codigo_vaga: str, page: int = 0, page_size: int = 1000) -> List[Dict]:
#        start = page * page_size
#        end = start + page_size - 1
#        response = supabase.table("applicants")\
#            .select("*")\
#            .eq("codigo_profissional", codigo_vaga)\
#            .range(start, end)\
#            .execute()
#        return response.data if response.data else []

    #def get_all_applicants(self) -> List[Dict]:
    #    """Retorna todos os candidatos cadastrados (sem paginação)."""
    #    response = supabase.table("applicants")\
    #        .select("*")\
    #        .limit(50000).execute()

    #    return response.data if response.data else []
    
#    def get_all_applicants(self, page: int = 0, page_size: int = 1000) -> List[Dict]:
#        """Retorna candidatos paginados."""
#        start = page * page_size
#        end = start + page_size - 1
#        response = supabase.table("applicants")\
#            .select("*")\
#            .range(start, end)\
#            .execute()
#        return response.data if response.data else []

    #def inserir_applicant_supabase(self, dados: Dict) -> None:
    #    response = supabase.table("applicants").insert(dados).execute()
    #    if response.error:
    #        raise Exception(f"Erro ao inserir no Supabase: {response.error.message}")
    #    return response.data
    
    ################################ Applicants
    # Inserção de novo applicant
#    def inserir_applicant_supabase(self, dados):
#        response = self.supabase.table("tbl_applicants").insert(dados).execute()
#        if response.status_code >= 400:
#            raise Exception(f"Erro na inserção: {response.data}")
#        return response.data

    # # Atualização de applicant existente
    # def atualizar_applicant_supabase(self, dados):
    #     id_applicant = dados["id"]
    #     dados_sem_id = dados.copy()
    #     del dados_sem_id["id"]
        
    #     response = self.supabase.table("tbl_applicants").update(dados_sem_id).eq("id", id_applicant).execute()
    #     if response.status_code >= 400:
    #         raise Exception(f"Erro na atualização: {response.data}")
    #     return response.data

    ########################## Prospects ###############################

    # def get_prospects(self, codigo_vaga: str) -> List[Dict]:
    #     response = supabase.table("prospects").select("*").eq("codigo_vaga", codigo_vaga).execute()
    #     return response.data if response.data else []

    # ############################ Vagas #################################

    # def get_all_vagas(self) -> List[Dict]:
    #     """Retorna todas as vagas disponíveis sem paginação."""
    #     response = supabase.table("vagas").select("*").limit(50000).execute()
    #     return response.data if response.data else []

    # def get_vaga_by_codigo(self, codigo_vaga: str) -> Optional[Dict]:
    #     response = supabase.table("vagas").select("codigo_vaga,titulo_vaga").eq("codigo_vaga", codigo_vaga).execute()
    #     if response.data:
    #         return response.data[0]
    #     return None

    # ################## IA - Similaridade por Embeddings #################

    # def get_candidatos_compativeis_por_titulo(self, titulo_vaga: str) -> pd.DataFrame:
    #     candidatos = self.get_all_applicants()
    #     resultados = self.get_embedding_applicants(titulo_vaga, candidatos)
    #     df_resultados = pd.DataFrame(resultados)
    #     resultados_ordenados = df_resultados.sort_values(by="score_similaridade", ascending=False)
    #     return resultados_ordenados

    # def get_embedding_applicants(self, titulo_vaga: str, candidatos: List[Dict]) -> List[Dict]:
    #     df_applicants = pd.DataFrame(candidatos)
    #     if "texto_cv" not in df_applicants.columns:
    #         df_applicants["texto_cv"] = ""
    #     embeddings_applicants = self.model.encode(df_applicants["texto_cv"].tolist(), convert_to_tensor=False)
    #     embedding_vaga = self.model.encode(titulo_vaga, convert_to_tensor=False)
    #     scores = cosine_similarity([embedding_vaga], embeddings_applicants)[0]
    #     df_applicants["score_similaridade"] = scores
    #     return df_applicants.to_dict(orient="records")

    # ################## Mock (ex: futuro insert de análise IA) ###########

    # def insert_analysis(self, analysis_obj: Analysis):
    #     print(f"🔄 Mock insert: {analysis_obj.codigo_vaga} - {analysis_obj.file}")

    # def insert_analysis(self, analysis_obj: Analysis):
    #     print(f"🔄 Mock insert: {analysis_obj.codigo_vaga} - {analysis_obj.file}")