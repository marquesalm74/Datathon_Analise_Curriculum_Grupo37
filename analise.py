import uuid
from helper import extract_data_analysis, read_pdf
from database import AnalyseDatabase
from ai import GroqClient
from models.analysis import Analysis

def analisar_vaga(codigo_vaga):
    database = AnalyseDatabase()
    ai = GroqClient()

    # Buscar vaga
    vaga = next((v for v in database.get_all_vagas() if v['codigo_vaga'] == codigo_vaga), None)
    if not vaga:
        print("❌ Vaga não encontrada.")
        return

    # Buscar candidatos vinculados
    candidatos = database.get_applicants(codigo_vaga)
    if not candidatos:
        print("⚠️ Nenhum candidato encontrado para esta vaga.")
        return

    # Descrição da vaga usada para análise
    descricao = vaga.get("descricao_vaga", vaga.get("titulo_vaga", ""))

    for cv in candidatos:
        nome = cv.get("nome", "Sem Nome")
        caminho_pdf = cv.get('caminho_pdf')

        if not caminho_pdf:
            print(f"⚠️ Candidato {nome} sem caminho para o PDF.")
            continue

        content = read_pdf(caminho_pdf)
        if not content:
            print(f"⚠️ Currículo vazio ou ilegível: {nome}")
            continue

        try:
            resumo = ai.resum_cv(content)  # Corrigido: apenas conteúdo
            opiniao = ai.generate_opinion(content, descricao)
            score = ai.generate_score(content, descricao)
        except Exception as e:
            print(f"❌ Erro na análise do candidato {nome}: {e}")
            continue

        resum_schema = Analysis(
            id=str(uuid.uuid4()),
            codigo_vaga=codigo_vaga,
            content=resumo,
            file=caminho_pdf,
            opnion=opiniao,
            score=score
        )

        extract_data_analysis(resumo)

        try:
            # database.insert_analysis(resum_schema)  # 🔴 Esse método não está implementado ainda!
            print(f"✅ Analisado: {nome} | Score: {score:.2f}")
        except Exception as e:
            print(f"❌ Erro ao salvar análise de {nome}: {e}")

######################## Atualização para utilizar o SUPABASE
#
# import uuid
# from helper import extract_data_analysis, read_pdf
# from database import AnalyseDatabase
# from ai import GroqClient
# from models.analysis import Analysis


# def analisar_vaga(codigo_vaga):
#     # 🔄 ALTERADO: Conexão inicial com Supabase via classe atualizada
#     database = AnalyseDatabase()  # 🔄 ALTERADO
#     ai = GroqClient()

#     # Buscar vaga
#     vaga = next((v for v in database.get_all_vagas() if v['codigo_vaga'] == codigo_vaga), None)
#     if not vaga:
#         print("❌ Vaga não encontrada.")
#         return

#     # Buscar candidatos vinculados
#     candidatos = database.get_applicants(codigo_vaga)
#     if not candidatos:
#         print("⚠️ Nenhum candidato encontrado para esta vaga.")
#         return

#     # Descrição da vaga usada para análise
#     descricao = vaga.get("descricao_vaga", vaga.get("titulo_vaga", ""))

#     for cv in candidatos:
#         nome = cv.get("nome", "Sem Nome")
#         caminho_pdf = cv.get('caminho_pdf')

#         if not caminho_pdf:
#             print(f"⚠️ Candidato {nome} sem caminho para o PDF.")
#             continue

#         content = read_pdf(caminho_pdf)
#         if not content:
#             print(f"⚠️ Currículo vazio ou ilegível: {nome}")
#             continue

#         try:
#             resumo = ai.resum_cv(content)  # 🔄 ALTERADO: apenas conteúdo
#             opiniao = ai.generate_opinion(content, descricao)
#             score = ai.generate_score(content, descricao)
#         except Exception as e:
#             print(f"❌ Erro na análise do candidato {nome}: {e}")
#             continue

#         resum_schema = Analysis(
#             id=str(uuid.uuid4()),  # 🔄 ALTERADO: garante ID único
#             codigo_vaga=codigo_vaga,
#             content=resumo,
#             file=caminho_pdf,
#             opnion=opiniao,
#             score=score
#         )

#         extract_data_analysis(resumo)

#         try:
#             # ✅ NOVO: agora com integração Supabase
#             database.insert_analysis(resum_schema)  # ✅ NOVO
#             print(f"✅ Analisado: {nome} | Score: {score:.2f}")
#         except Exception as e:
#             print(f"❌ Erro ao salvar análise de {nome}: {e}")

