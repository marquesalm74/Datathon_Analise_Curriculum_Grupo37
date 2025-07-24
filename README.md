Projeto DATATHON 

Link da aplicação - DEPLOY STREAMLIT CLOUD : https://analyserresume-3ud8n6vybz2gwgz4ejrlor.streamlit.app/

Alessandro Lúcio Marques - rm: 358206
Geovanna de Araújo Barros - rm: 358660
Jhonny Brasiliano da Silva - rm: 359212

Recebida 3 bases de dados: propescts, applicants e vagas. Foram tratadas, pois há muitos dados faltantes e caracteres especiais. A base de dados vagas e applicants e prospect tem em comum uma relação, por meio do, id ou codigo_vaga.
Foi trabalhado a inserção no banco de dados de novo candidato e de atualização de dados de vagas selecionadas no front-end, com a intenção de preencher corretamente a base de dados. Tarefa do setor de recursos humanos.

A aplicação consiste em buscar cada base de dados pelo codigo da vaga e seu respectivo título, logo, as db applicants e prospects retornam a aplicação para os registros segundo o codigo da vaga e título da vaga. Percebeu que há um registro para cada candidato inscrito relacionado ao código da vaga. Ou seja, o retorno é de um candidato para cada vaga em um DB de mais de 45 mil registros. A base prospects tem os candidatos prospectados pelos recrutadores e não constam informações relevantes das qualificações e competências técncias de cada prospectado.

A tarefa foi mostrar o resultado da consulta na tela, verificar o quantitativo de candidatos inscritos e prospectados, e sugerir atualização ou inserção na base de dados applicants, para que esta seja atualizada. O objetivo da aplicação em streamlit é verificar as inscrições por vagas e buscar com a utilização de inteligência artificial. No escopo do projeto foi utilizado Embeddings/Cosine Similarity e modelo (all-MiniLM-L6-v2), trabalhado em conjunto, forneceram uma representação vetorial de palavras/frases, a medida da similaridade do coseno como metrica para comparar vetores. O modelo é baseado no Sentence Transformance - Bert/transformers, este é um MinLM (Mini Language Natural), usado para busca semântica, agrupamento (clustering), classificação e cálculo de similaridade (ex: usando cosine similarity).

O resultado da aplicação analisa todos os registros dos applicants, por meio de uma variável Target, definida como text_cv, agregativa das colunas (strings) - "cargo_atual", "objetivo_profissional", "titulo_profissional", "area_atuacao". Por meio desta Target, o modelo avalia numericamente cada candidato e relaciona em ordem decrescente de pontuação - do maior para o menor, o candidato com maior aptidão para participar do processo seletivo. Estabelecemos algumas métricas para o usuário analisar os candidatos selecionados que são: média do score para o grupo de 20 candidatos que são retornados no front-end, o valor máximo, o mínimo e o desvio-padrão. Com estas métricas observa-se onde há maior concentração de candidatos com scores acima ou abaixo da média e fornece uma avaliação mais pormenorizada, pois o setor do recursos humanos pode analisar se os candidatos são realmente bons ou não.

At.te;
