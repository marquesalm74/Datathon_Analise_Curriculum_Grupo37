# Imagem base
FROM python:3.12

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos do Poetry primeiro (para cache eficiente)
COPY pyproject.toml poetry.lock ./

# Instalar Poetry
RUN pip install poetry==1.8.2

# Configurar Poetry para não criar virtualenv (usa o ambiente do sistema)
RUN poetry config virtualenvs.create false

# Instalar dependências
RUN poetry install --no-dev

# Copiar todo o restante do projeto
COPY . .

# Expor a porta do Streamlit
EXPOSE 8501

# Comando para rodar a aplicação - **confirme o script correto aqui!**
CMD ["streamlit", "run", "appcv.py", "--server.port=8501", "--server.address=0.0.0.0"]