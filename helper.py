import os
import re
import uuid
import fitz  # PyMuPDF
import pandas as pd

################################## PARA USAR O ARQUIVO TRANSFORMADO DE JSON -> DATAFRAME
#
def read_pdf(file_path):
    text = ''
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def get_pdf_paths(directory):
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith('.pdf')
    ]


def filtrar_candidatos_validos(df, colunas_obrigatorias):
    df_copy = df.copy()
    for col in colunas_obrigatorias:
        if col in df_copy.columns:
            col_clean = df_copy[col].astype(str).str.strip().str.lower()
            df_copy = df_copy[~col_clean.isin(['', 'nan', 'sem informação'])]
    return df_copy.reset_index(drop=True)


def extract_data_analysis(resum_cv, job_id, resum_id, score) -> dict:
    secoes_dict = {
        "id": resum_id,
        "job_id": job_id,
        "name": "",
        "skills": [],
        "education": [],
        "languages": [],
        "score": score
    }

    patterns = {
        "name": r"(?:## Nome Completo\s*|Nome Completo\s*\|\s*Valor\s*\|\s*\S*\s*\|\s*)(.*)",
        "skills": r"## Habilidades\s*([\s\S]*?)(?=##|$)",
        "education": r"## Educação\s*([\s\S]*?)(?=##|$)",
        "languages": r"## Idiomas\s*([\s\S]*?)(?=##|$)"
    }

    def clean_string(string: str) -> str:
        return re.sub(r"[\*\-]+", "", string).strip()

    for secao, pattern in patterns.items():
        match = re.search(pattern, resum_cv)
        if match:
            if secao == "name":
                secoes_dict[secao] = clean_string(match.group(1))
            else:
                secoes_dict[secao] = [clean_string(item) for item in match.group(1).split('\n') if item.strip()]

    for key in ["name", "education", "skills"]:
        if not secoes_dict[key] or (isinstance(secoes_dict[key], list) and not any(secoes_dict[key])):
            raise ValueError(f"A seção '{key}' não pode ser vazia ou uma string vazia.")

    return secoes_dict   


######################## PARA USAR O SUPABASE 
#
#import os
#import re
#import uuid
#import fitz  # PyMuPDF
#import pandas as pd
#
# def read_pdf(file_path):
#     text = ''
#     with fitz.open(file_path) as pdf:
#         for page in pdf:
#             text += page.get_text()
#     return text


# def get_pdf_paths(directory):
#     return [
#         os.path.join(directory, f)
#         for f in os.listdir(directory)
#         if f.lower().endswith('.pdf')
#     ]


# def filtrar_candidatos_validos(candidatos: list, colunas_obrigatorias: list):
#     """
#     Adaptação: recebe lista de dicts (de Supabase) e filtra candidatos válidos.
#     """
#     candidatos_filtrados = []

#     for candidato in candidatos:
#         valido = True
#         for col in colunas_obrigatorias:
#             valor = candidato.get(col, "")
#             if not isinstance(valor, str):
#                 valor = str(valor)
#             valor_limpo = valor.strip().lower()
#             if valor_limpo in ["", "nan", "sem informação"]:
#                 valido = False
#                 break
#         if valido:
#             candidatos_filtrados.append(candidato)

#     return candidatos_filtrados


# def extract_data_analysis(resum_cv, job_id, resum_id, score) -> dict:
#     secoes_dict = {
#         "id": resum_id,
#         "job_id": job_id,
#         "name": "",
#         "skills": [],
#         "education": [],
#         "languages": [],
#         "score": score
#     }

#     patterns = {
#         "name": r"(?:## Nome Completo\s*|Nome Completo\s*\|\s*Valor\s*\|\s*\S*\s*\|\s*)(.*)",
#         "skills": r"## Habilidades\s*([\s\S]*?)(?=##|$)",
#         "education": r"## Educação\s*([\s\S]*?)(?=##|$)",
#         "languages": r"## Idiomas\s*([\s\S]*?)(?=##|$)"
#     }

#     def clean_string(string: str) -> str:
#         return re.sub(r"[\*\-]+", "", string).strip()

#     for secao, pattern in patterns.items():
#         match = re.search(pattern, resum_cv)
#         if match:
#             if secao == "name":
#                 secoes_dict[secao] = clean_string(match.group(1))
#             else:
#                 secoes_dict[secao] = [clean_string(item) for item in match.group(1).split('\n') if item.strip()]

#     for key in ["name", "education", "skills"]:
#         if not secoes_dict[key] or (isinstance(secoes_dict[key], list) and not any(secoes_dict[key])):
#             raise ValueError(f"A seção '{key}' não pode ser vazia ou uma string vazia.")

#     return secoes_dict

