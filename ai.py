import re
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class GroqClient:
    def __init__(self, model_id='llama-3.3-70b-versatile'):
        self.model_id = model_id
        self.client = ChatGroq(model=self.model_id)
       
    def generate_response(self, prompt):
        try:
            response = self.client.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"❌ Erro ao gerar resposta da IA: {e}")
            return ""

    def resum_cv(self, cv):
        prompt = f'''
            Solicitação de Resumo de Currículo em Markdown:

            Currículo do Candidato:

            {cv}

            Por favor, gere um resumo formatado em Markdown, seguindo rigorosamente o modelo abaixo.
            Não adicione seções extras, tabelas ou qualquer outra estrutura.

            ```markdown
            ## Nome Completo
            nome_completo aqui

            ## Experiência
            experiencia aqui

            ## Habilidades
            habilidades aqui

            ## Educação
            educacao aqui

            ## Idiomas
            idiomas aqui
            ```
        '''

        result_raw = self.generate_response(prompt)

        try:
            result = result_raw.split("```markdown")[1].split("```")[0].strip()
        except Exception as e:
            print(f"⚠️ Erro ao extrair markdown: {e}")
            result = result_raw  # fallback

        return result
    
    def generate_score(self, cv, vaga, max_attempts=3):
        prompt = f'''
            Objetivo: Avaliar o currículo com base na vaga e gerar uma pontuação final (máximo 10.0).

            Critérios e Pesos:
            1. Experiência (30%)
            2. Habilidades Técnicas (25%)
            3. Educação (10%)
            4. Pontos Fortes (15%)
            5. Pontos Fracos (10%)

            Currículo:
            {cv}

            Descrição da Vaga:
            {vaga}

            Output Esperado:
            ```
            Pontuação Final: x.x
            ```
            Obs: A nota deve estar no formato: Pontuação Final: 7.5 (apenas isso).
        '''

        result_raw = self.generate_response(prompt)
        score = self.extract_score_from_result(result_raw)
        return score if score is not None else 0.0

    def extract_score_from_result(self, result_raw):
        pattern = r"(?i)Pontuação Final[:\s]*([\d.,]+)"
        match = re.search(pattern, result_raw)

        if match:
            try:
                score_str = match.group(1).replace(",", ".").strip()
                return float(score_str)
            except ValueError:
                print(f"⚠️ Erro ao converter score: {match.group(1)}")
                return None

        print("⚠️ Padrão 'Pontuação Final' não encontrado no resultado:")
        print(result_raw)
        return None

    def generate_opinion(self, cv, vaga):
        prompt = f'''
            Você é um recrutador sênior. Analise criticamente o currículo abaixo com base na vaga.

            Estruture a resposta com os seguintes tópicos, com títulos grandes:

            1. **Pontos de Alinhamento**: O que está aderente à vaga?
            2. **Pontos de Desalinhamento**: O que não atende aos requisitos?
            3. **Pontos de Atenção**: Lacunas, mudanças de carreira, etc.

            Currículo:
            {cv}

            Vaga:
            {vaga}

            Gere a resposta como um relatório bem estruturado e profissional.
        '''

        return self.generate_response(prompt)
    
if __name__ == "__main__":
    print("ai.py rodou com sucesso!")
