import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from database import AnalyseDatabase
from ai import GroqClient
from tratarbase import applicants
from scipy.stats import gaussian_kde
from fpdf import FPDF

# Configuração da página
st.set_page_config(layout='wide', page_title='Análise de Currículo')

# Inicializa conexões
database = AnalyseDatabase()
ai = GroqClient()

# Menu lateral
st.sidebar.markdown("## MENU")
acao = st.sidebar.selectbox(
    "Ação",
    ["Análise de Currículo", "Inserir/Atualizar Applicants"] # "IA Groq / Análise dos Escolhidos",
)

# --- Seção: Análise de Currículo ---
if acao == "Análise de Currículo":
    st.title("🔎 Análise de Currículos")

    df_app = pd.DataFrame()
    df_pros = pd.DataFrame()

    # Carrega vagas disponíveis
    vagas = database.get_all_vagas()
    opcoes_codigo_vaga = [""] + sorted([vaga['codigo_vaga'] for vaga in vagas])

    codigo_vaga = st.selectbox(
        'Selecione o Código da Vaga:',
        opcoes_codigo_vaga,
        format_func=lambda x: 'Selecione uma vaga' if x == '' else x
    )

    st.session_state.codigo_vaga_atual = codigo_vaga

    if codigo_vaga:
        vaga_selecionada = next((v for v in vagas if v['codigo_vaga'] == codigo_vaga), None)

        if vaga_selecionada:
            st.session_state["titulo_vaga_atual"] = vaga_selecionada["titulo_vaga"]

            st.markdown(
                f"""
                <span style='font-size:25px; font-weight:bold;'>
                    <span style='color:white;'>Título da Vaga:</span>
                    <span style='color:#2E86C2;'> {vaga_selecionada['titulo_vaga']}</span>
                </span>
                """,
                unsafe_allow_html=True
            )
            st.divider()

            # 📄 Candidatos Inscritos
            st.subheader("📄 Candidatos Inscritos")
            applicants = database.get_applicants(codigo_vaga)
            df_app = pd.DataFrame(applicants)
            if df_app.empty:
                st.info("Nenhum candidato inscrito com dados suficientes para esta vaga.")
            else:
                st.dataframe(df_app, use_container_width=True)

            # 🔍 Profissionais Prospectados
            st.subheader("🧲 Profissionais Prospectados")
            prospects = database.get_prospects(codigo_vaga)
            if prospects:
                df_pros = pd.DataFrame(prospects)
                st.dataframe(df_pros, use_container_width=True)
            else:
                st.info("Nenhum prospect relacionado para esta vaga.")

            # ✅ Totalizadores
            total_applicants = len(df_app)
            total_prospects = len(df_pros)
            st.text(f"👥 Total de Candidatos Inscritos: {total_applicants}")
            st.text(f"🔎 Total de Profissionais Prospectados: {total_prospects}")
            st.text(f"📊 Total Geral (Inscritos + Prospectados): {total_applicants + total_prospects}")

            if total_applicants == 1:
                st.warning(
                    "Apenas 1 candidato está inscrito nesta vaga. Para ampliar as opções, analise os prospectados ou atualize os candidatos cadastrados. Você também tem a opção de inserir novo candidato. Clique abaixo para buscar candidatos compatíveis já cadastrados."
                )

            # 🔎 Análise com IA local via embeddings
            if st.button("🔍 Identificar e Avaliar Candidatos com IA"):
                with st.spinner("Analisando candidatos..."):

                    campos_essenciais = ["cargo_atual", "objetivo_profissional", "titulo_profissional", "area_atuacao"]

                    def campo_valido(x):
                        return bool(str(x).strip()) and str(x).strip().lower() not in ['nan', 'sem informação']

                    titulo_vaga = vaga_selecionada.get("titulo_vaga", "")
                    if not titulo_vaga:
                        st.warning("Título da vaga não encontrado.")
                        st.stop()

                    candidatos_ia = database.get_candidatos_compativeis_por_titulo(titulo_vaga)
                    if candidatos_ia is None or candidatos_ia.empty:
                        st.warning("Nenhum candidato compatível com o título da vaga.")
                        st.stop()

                    df_ia = candidatos_ia
                    for campo in campos_essenciais:
                        if campo not in df_ia.columns:
                            df_ia[campo] = ""

                    df_filtrado = df_ia[df_ia[campos_essenciais].applymap(campo_valido).all(axis=1)]
                    if df_filtrado.empty:
                        st.warning("Nenhum candidato com dados completos para análise.")
                        st.stop()

                    df_avaliado = df_filtrado.copy()
                    df_avaliado["Score"] = (df_avaliado["score_similaridade"].fillna(0) * 10).round(2)
                    df_avaliado.drop(columns=["score_similaridade"], inplace=True, errors='ignore')
                    df_avaliado["tipo"] = "applicant"
                    st.session_state.df_filtrado = df_avaliado.copy()

                    if (df_avaliado["Score"] <= 4).all():
                        st.warning("⚠️ Todos os candidatos avaliados apresentaram score abaixo ou igual a 4.")
                        st.info("📭 Nenhum candidato com compatibilidade suficiente para a vaga. Ajuste a vaga ou amplie os critérios.")
                        st.stop()

                    score_mean = df_avaliado["Score"].mean()
                    score_max = df_avaliado["Score"].max()
                    score_min = df_avaliado["Score"].min()
                    score_stdev = df_avaliado['Score'].std()

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("📈 Score Médio", f"{score_mean:.2f}")
                    col2.metric("🔝 Máximo", f"{score_max:.2f}")
                    col3.metric("🔻 Mínimo", f"{score_min:.2f}")
                    col4.metric("🔻 Desvio Padrão", f"{score_stdev:.2f}")

                    score_bins = pd.cut(df_avaliado["Score"], bins=[0, 2, 4, 6, 8, 10], include_lowest=True)
                    score_dist = score_bins.value_counts().sort_index()
                    score_dist.index = score_dist.index.astype(str)
                    score_dist_df = score_dist.reset_index()
                    score_dist_df.columns = ['Faixa', 'Quantidade']

                    st.write("📊 Distribuição dos scores:")
                    st.bar_chart(score_dist_df.set_index('Faixa'))

                    st.info(f"🔎 Total de candidatos analisados: {len(df_avaliado)}")
                    st.success("✅ Candidatos avaliados com sucesso.")

                    # Top 20 com Score mais alto
                    df_top20 = df_avaliado.sort_values("Score", ascending=False).head(20).copy()
                    df_top20["Score"] = df_top20["Score"].round(2)
                    st.session_state.df_top20 = df_top20

                    st.plotly_chart(
                        px.bar(
                            df_top20,
                            x="Score",
                            y="nome",
                            orientation='h',
                            color="Score",
                            color_continuous_scale='Viridis',
                            labels={"Score": "Pontuação", "nome": "Candidato"},
                            height=600
                        ).update_layout(
                            yaxis=dict(autorange="reversed"),
                            xaxis=dict(range=[0, 10]),
                            margin=dict(l=150, r=40, t=40, b=40),
                            coloraxis_colorbar=dict(title="Score")
                        ),
                        use_container_width=True
                    )
                    

                                    
##################################################################################################

elif acao == "Inserir/Atualizar Applicants":
    st.header('🔎 Inserir/Atualizar Applicants')
    
    # Garantir que o código da vaga esteja disponível (se quiser pode remover essa parte se não usar código_vaga)
    codigo_vaga = st.session_state.get("codigo_vaga_atual")
    if not codigo_vaga:
        st.warning("❗Nenhuma vaga selecionada. Por favor, vá até 'Análise de Currículo' e selecione uma vaga antes de continuar.")
        st.stop()

    modo_cadastro = st.radio("Você deseja:", ["Cadastrar novo candidato", "Editar candidato existente"], horizontal=True)

    with st.expander("Veja o Formulário Expandido!"):
        todos_applicants = database.get_applicants(codigo_vaga)
        lista_ids_applicants = [str(a['id']) for a in todos_applicants]

        # Variáveis para controle do ID e dados
        id_selecionado = ""
        id_digitado = ""
        dados_candidato = {}
        id_final = ""

        if modo_cadastro == "Cadastrar novo candidato":
            id_digitado = st.text_input("ID do candidato (opcional)", placeholder="Digite o Id do Novo Candidato")
            id_final = id_digitado.strip() if id_digitado else None
        elif modo_cadastro == "Editar candidato existente":
            id_selecionado = st.selectbox("Selecione o ID do candidato:", [""] + lista_ids_applicants)
            if id_selecionado:
                dados_candidato = next((item for item in todos_applicants if str(item["id"]) == str(id_selecionado)), {})
                id_final = id_selecionado
            else:
                dados_candidato = {}
                id_final = None

        # Campos do formulário
        nome = st.text_input("Nome", value=dados_candidato.get("nome", ""))
        telefone = st.text_input("Telefone", value=dados_candidato.get("telefone", ""))
        email = st.text_input("E-mail", value=dados_candidato.get("email", ""))
        local = st.text_input("Local", value=dados_candidato.get("local", ""))
        codigo_profissional = st.text_input("Código Profissional", value=dados_candidato.get("codigo_profissional", ""))
        data_nascimento = st.date_input("Data de Nascimento", value=pd.to_datetime(dados_candidato.get("data_nascimento", "2000-01-01")).date())
        telefone_celular = st.text_input("Telefone Celular", value=dados_candidato.get("telefone_celular", ""))

        sexo_opcoes = ["", "Feminino", "Masculino"]
        estado_civil_opcoes = ["", "Casado", "Solteiro", "Separado Judicialmente", "Divorciado"]
        pcd_opcoes = ["False", "True"]
        estado_opcoes = [
            "", "Acre", "Alagoas", "Amapá", "Amazonas", "Bahia", "Ceará", "Distrito Federal",
            "Espírito Santo", "Goiás", "Maranhão", "Mato Grosso", "Mato Grosso do Sul", "Minas Gerais",
            "Pará", "Paraíba", "Paraná", "Pernambuco", "Piauí", "Rio de Janeiro", "Rio Grande do Norte",
            "Rio Grande do Sul", "Rondônia", "Roraima", "Santa Catarina", "São Paulo", "Sergipe", "Tocantins"
        ]

        try:
            sexo_index = sexo_opcoes.index(dados_candidato.get("sexo", ""))
        except ValueError:
            sexo_index = 0
        sexo = st.selectbox("Qual o seu Sexo:", sexo_opcoes, index=sexo_index)

        try:
            ec_index = estado_civil_opcoes.index(dados_candidato.get("estado_civil", ""))
        except ValueError:
            ec_index = 0
        estado_civil = st.selectbox("Qual o seu Estado Civil:", estado_civil_opcoes, index=ec_index)

        try:
            pcd_index = pcd_opcoes.index(str(dados_candidato.get("pcd", "False")))
        except ValueError:
            pcd_index = 0
        pcd = st.selectbox("É Pessoa Com Deficiência:", pcd_opcoes, index=pcd_index)

        try:
            estado_index = estado_opcoes.index(dados_candidato.get("endereco", ""))
        except ValueError:
            estado_index = 0
        endereco = st.selectbox("Endereço (UF):", estado_opcoes, index=estado_index)

        url_linkedin = st.text_input("Digite a url_linkedin:", value=dados_candidato.get("url_linkedin", ""))
        titulo_profissional = st.text_input("Digite o seu Título Profissional:", value=dados_candidato.get("titulo_profissional", ""))
        area_atuacao = st.text_input("Digite a sua Área de Atuação:", value=dados_candidato.get("area_atuacao", ""))
        conhecimentos_tecnicos = st.text_input("Digite os seus Conhecimentos Técnicos:", value=dados_candidato.get("conhecimentos_tecnicos", ""))
        certificacoes = st.text_input("Quais as suas Certificações:", value=dados_candidato.get("certificacoes", ""))
        remuneracao = st.text_input("Qual a sua Remuneração:", value=dados_candidato.get("remuneracao", ""))
        nivel_profissional = st.text_input("Qual o seu Nível Profissional:", value=dados_candidato.get("nivel_profissional", ""))
        nivel_ingles = st.text_input("Qual o seu Nível de Inglês:", value=dados_candidato.get("nivel_ingles", ""))
        nivel_espanhol = st.text_input("Qual o seu Nível de Espanhol:", value=dados_candidato.get("nivel_espanhol", ""))
        outro_idioma = st.text_input("Qual o seu Outro Idioma:", value=dados_candidato.get("outro_idioma", ""))

        # Montar dicionário para salvar (sem codigo_vaga, data_criacao, fonte_indicacao)
        dados = {
            "id": id_final,
            "nome": nome,
            "telefone": telefone,
            "email": email,
            "local": local,
            "codigo_profissional": codigo_profissional,
            "data_nascimento": str(data_nascimento),
            "telefone_celular": telefone_celular,
            "sexo": sexo,
            "estado_civil": estado_civil,
            "pcd": pcd == "True",
            "endereco": endereco,
            "url_linkedin": url_linkedin,
            "titulo_profissional": titulo_profissional,
            "area_atuacao": area_atuacao,
            "conhecimentos_tecnicos": conhecimentos_tecnicos,
            "certificacoes": certificacoes,
            "remuneracao": remuneracao,
            "nivel_profissional": nivel_profissional,
            "nivel_ingles": nivel_ingles,
            "nivel_espanhol": nivel_espanhol,
            "outro_idioma": outro_idioma,
        }

        if st.button("💾 Salvar Candidato"):
            if modo_cadastro == 'Cadastrar novo candidato':
                database.inserir_applicant_novo(dados)
                st.success(f'✅ Novo candidato cadastrado com sucesso!\n🆕 ID: {id_final}')
            else:
                database.atualizar_applicant(dados)
                st.success(f'✅ Dados do candidato {id_final} atualizados com sucesso!')
                
        #if st.button("💾 Salvar Candidato"):
        #    if modo_cadastro == "Cadastrar novo candidato":
        #        database.inserir_applicant_supabase(dados)
        #        st.success(f"✅ Novo candidato cadastrado com sucesso!\n🆕 ID: {id_final}")
        #    else:
        #        database.atualizar_applicant_supabase(dados)
        #        st.success(f"✅ Dados do candidato {id_final} atualizados com sucesso!")
