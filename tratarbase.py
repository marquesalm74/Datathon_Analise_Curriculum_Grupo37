import pandas as pd
import numpy as np
import json
import unidecode
import re
import warnings

from supabase_client import supabase

############################# LER ARQUIVOS JSON
######## applicants.json

with open('documents/applicants.json', 'r', encoding='utf-8') as candidatos:
    dados = json.load(candidatos)

# Vamos transformar os dados aninhados
lista_candidatos = []

# Percorrer cada candidato
for id_candidato, info in dados.items():
    registro = {'id': id_candidato}  # Começamos armazenando o ID
    registro.update(info.get('infos_basicas', {}))
    registro.update(info.get('informacoes_pessoais', {}))
    registro.update(info.get('informacoes_profissionais', {}))
    registro.update(info.get('formacao_e_idiomas', {}))
    registro.update(info.get('cargo_atual', {}))
    registro['cv_pt'] = info.get('cv_pt', '')
    lista_candidatos.append(registro)

# Criar o DataFrame estruturado
df = pd.DataFrame(lista_candidatos)

def tratar_base(df):
    # 1. Substituir valores NaN e strings vazias por 'sem informacao'
    df = df.replace({':': np.nan, '-': np.nan, '': np.nan}).infer_objects(copy=False)        
    df = df.fillna('sem informacao')

    # 2. Converter colunas de data para datetime
    colunas_data = [col for col in df.columns if 'data' in col.lower()]
    for col in colunas_data:
        try:
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors='coerce')
        except Exception:
            # Se erro, tenta converter sem format (fallback)
            df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].fillna(pd.Timestamp('1900-01-01'))

    # 3. Converter coluna 'remuneracao' para float, se existir
    if 'remuneracao' in df.columns:
        df['remuneracao'] = df['remuneracao'].replace('sem informacao', np.nan)
        df['remuneracao'] = df['remuneracao'].astype(str)
        df['remuneracao'] = df['remuneracao'].str.replace('R$', '', regex=False)
        df['remuneracao'] = df['remuneracao'].str.replace('.', '', regex=False)
        df['remuneracao'] = df['remuneracao'].str.replace(',', '.', regex=False)
        df['remuneracao'] = df['remuneracao'].fillna(0.00)

    return df

df_tratado = tratar_base(df)

def limpar_texto(texto):
    if not isinstance(texto, str):
        return texto
    return (
        texto.replace('\n', ' ')
             .replace('\r', ' ')
             .replace('\\', ' ')
             .replace('"', "'")
             .strip()
    )


colunas_texto = ['cv_pt', 'experiencias', 'qualificacoes', 'cursos', 'projeto_atual']
for col in colunas_texto:
    if col in df_tratado.columns:
        df_tratado[col] = df_tratado[col].apply(limpar_texto)

applicants = df_tratado[[
    'id','codigo_profissional','nome','email','sexo','estado_civil',
    'data_nascimento','telefone','telefone_celular','local','endereco',
    'pcd','cargo_atual','objetivo_profissional','url_linkedin','titulo_profissional',
    'area_atuacao','conhecimentos_tecnicos','nivel_academico','cursos','certificacoes',
    'remuneracao','nivel_profissional', 'nivel_ingles', 'nivel_espanhol','outro_idioma',
    'cv_pt','instituicao_ensino_superior','qualificacoes','experiencias','outro_curso',
    'projeto_atual','unidade','download_cv'
]]

applicants.loc[:, 'pcd'] = applicants['pcd'].apply(lambda x: None if str(x).lower() == 'sem_informacao' else x)

# realocado de embeddings
# Garante colunas necessárias
for col in ["cargo_atual", "objetivo_profissional", "titulo_profissional", "area_atuacao"]:
    if col not in applicants.columns:
        applicants[col] = ""

# Cria campo concatenado
    applicants["texto_cv"] = (
    applicants["cargo_atual"].fillna("") + " " +
    applicants["objetivo_profissional"].fillna("") + " " +
    applicants["titulo_profissional"].fillna("") + " " +
    applicants["area_atuacao"].fillna("")
    )

applicants.head(2)
###################################################################################
#                                   Prospects.json

with open('documents/prospects.json', 'r', encoding='utf-8') as perspectivas:
    objetivo_empresa = json.load(perspectivas)

lista_perspectivas = []

for codigo_vaga, vaga in objetivo_empresa.items():
    titulo = vaga['titulo']
    modalidade = vaga['modalidade']
    for prospect in vaga['prospects']:
        registro = {
            'codigo_vaga': codigo_vaga,
            'titulo': titulo,
            'modalidade': modalidade,
            **prospect
        }
        lista_perspectivas.append(registro)

prospects = pd.DataFrame(lista_perspectivas)

def tratar_propects(df):
    df['data_candidatura'] = pd.to_datetime(df['data_candidatura'], dayfirst=True, errors='coerce')
    df['ultima_atualizacao'] = pd.to_datetime(df['ultima_atualizacao'], dayfirst=True, errors='coerce')
    df['comentario'] = df['comentario'].apply(lambda x: 'sem informacao' if str(x).strip() == '' else x)
    return df

prospects_tratada = tratar_propects(prospects)
prospects_tratada.drop(columns='modalidade', inplace=True)

###############################################################################
#                                 vagas.json

with open('documents/vagas.json', 'r', encoding='utf-8') as vagas:
    perfil_vagas = json.load(vagas)

def tratar_vagas(dados_vagas):
    lista_vagas = []
    for codigo_vaga, vaga in dados_vagas.items():
        registro = {'codigo_vaga': codigo_vaga}
        registro.update(vaga.get('informacoes_basicas', {}))
        registro.update(vaga.get('perfil_vaga', {}))
        registro.update(vaga.get('beneficios', {}))
        lista_vagas.append(registro)
    df_vagas = pd.DataFrame(lista_vagas)
    return df_vagas

perfil_vagas = tratar_vagas(perfil_vagas)

def tratar_vagas(df):
    df = df.copy()

    # Função auxiliar para substituir valores nulos ou vazios
    def substituir_valores(x, substituto='sem_informacao'):
        if pd.isnull(x):
            return substituto
        if isinstance(x, str):
            if x.strip() in ['', '-']:
                return substituto
        return x

    # Colunas que terão tratamento simples de preenchimento
    colunas_simples = [
        'prazo_contratacao', 'objetivo_vaga', 'prioridade_vaga',
        'origem_vaga', 'superior_imediato', 'nome', 'telefone',
        'bairro', 'regiao', 'vaga_especifica_para_pcd',
        'nivel_espanhol', 'viagens_requeridas', 'equipamentos_necessarios',
        'valor_compra_2', 'habilidades_comportamentais_necessarias',
        'nome_substituto'
    ]

    # Aplicar substituição somente nas colunas existentes
    for coluna in colunas_simples:
        if coluna in df.columns:
            df[coluna] = df[coluna].apply(lambda x: substituir_valores(x, 'sem_informacao'))

    # Substituir valores específicos nas colunas, se existirem
    if 'faixa_etaria' in df.columns:
        df['faixa_etaria'] = df['faixa_etaria'].replace({'De: Até:': 'sem_informacao', '': 'sem_informacao'}).infer_objects(copy=False).fillna('sem_informacao')

    if 'valor_venda' in df.columns:
        df['valor_venda'] = df['valor_venda'].replace({'-': 'sem_informacao', '': 'sem_informacao'}).fillna('sem_informacao')

    if 'outro_idioma' in df.columns:
        df['outro_idioma'] = df['outro_idioma'].replace({'': 'nenhum'}).fillna('nenhum')

    # Corrigir nome da coluna 'nivel profissional' se existir
    if 'nivel profissional' in df.columns:
        df.rename(columns={'nivel profissional': 'nivel_profissional'}, inplace=True)

    # Corrigir nome da coluna 'data_requicisao' para 'data_requisicao' antes da conversão
    if 'data_requicisao' in df.columns:
        df.rename(columns={'data_requicisao': 'data_requisicao'}, inplace=True)

    datas_para_converter = ['data_inicial', 'data_final', 'data_requisicao', 'limite_esperado_para_contratacao']

    for coluna in datas_para_converter:
        if coluna in df.columns:
            try:
                # Tenta converter com formato explícito (ajuste aqui se necessário)
                df[coluna] = pd.to_datetime(df[coluna], format="%d/%m/%Y", errors='coerce')
            except Exception:
                # Se falhar, tenta conversão genérica com dayfirst=True
                df[coluna] = pd.to_datetime(df[coluna], dayfirst=True, errors='coerce')

            # Preencher datas inválidas com data padrão
            df[coluna] = df[coluna].fillna(pd.Timestamp('1900-01-01'))

    return df

perfil_vagas_tratada = tratar_vagas(perfil_vagas)

####################################################################################
# Função para pegar colunas da tabela no Supabase

def pegar_colunas_tabela(tabela):
    query = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{tabela}';
    """
    resultado = supabase.rpc('sql', {'q': query}).execute()
    if resultado.data:
        return [row['column_name'] for row in resultado.data]
    else:
        print(f"⚠️ Não foi possível obter colunas para a tabela '{tabela}'. Usando todas as colunas do DataFrame.")
        return None

####################################################################################
# Função para subir dados ao Supabase removendo colunas inválidas
COLUNAS_VALIDAS = {
    "prospects": [
        "codigo_vaga", "titulo", "data_candidatura", "ultima_atualizacao",
        "nome", "email", "telefone", "comentario", "status", "situacao_candidato"
    ],
    "vagas": [
        "codigo_vaga", "nome", "objetivo_vaga", "prioridade_vaga",
        "prazo_contratacao", "origem_vaga", "superior_imediato",
        "nivel_espanhol", "faixa_etaria", "valor_venda",
        "data_inicial", "data_final", "data_requisicao", "limite_esperado_para_contratacao"
    ],
    "applicants_new": [
        "id","codigo_profissional","nome","email","sexo","estado_civil",
        "data_nascimento","telefone","telefone_celular","local","endereco",
        "pcd","cargo_atual","objetivo_profissional","url_linkedin","titulo_profissional",
        "area_atuacao","conhecimentos_tecnicos","nivel_academico","cursos","certificacoes",
        "remuneracao","nivel_profissional","nivel_ingles","nivel_espanhol","outro_idioma",
        "cv_pt","instituicao_ensino_superior","qualificacoes","experiencias","outro_curso",
        "projeto_atual","unidade","download_cv"
    ]
}

#####################################################################################
#def subir_para_supabase(df, tabela):
#    df = df.copy()

#    colunas_validas = COLUNAS_VALIDAS.get(tabela)
#    if colunas_validas:
#        colunas_extras = [col for col in df.columns if col not in colunas_validas]
#        if colunas_extras:
#            print(f"Removendo colunas que não existem na tabela '{tabela}': {colunas_extras}")
#            df = df.drop(columns=colunas_extras)
#    else:
#        print(f"Aviso: tabela '{tabela}' sem definição de colunas válidas.")

#    for col in df.select_dtypes(include=['datetime64[ns]']).columns:
#        df[col] = df[col].fillna(pd.Timestamp('1900-01-01'))
#        df[col] = df[col].dt.strftime('%Y-%m-%d')

#    for col in df.columns:
#        if df[col].dtype in ['float64', 'int64']:
#            df[col] = df[col].fillna(0.0)
#        elif col in df.select_dtypes(include=['object']).columns:
#            df[col] = df[col].fillna('sem_informacao')

#    dados = df.replace({np.nan: None}).to_dict(orient='records')

#    if len(dados) == 0:
#        print(f'Nenhum registro para inserir na tabela {tabela}.')
#        return

#    resposta = supabase.table(tabela).insert(dados).execute()

#    if resposta.data is not None:
#        print(f'{len(dados)} registros inseridos com sucesso na tabela {tabela}.')
#    else:
#        print(f'Erro ao inserir na tabela {tabela}: {resposta.error}')


################# Inserir direto no SUPABASE
#subir_para_supabase(applicants, 'applicants')
#subir_para_supabase(prospects_tratada, 'prospects')
#subir_para_supabase(perfil_vagas_tratada, 'vagas')
#subir_para_supabase(applicants, 'applicants_new')
