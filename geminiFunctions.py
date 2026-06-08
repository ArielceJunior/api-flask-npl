import google.generativeai as generativeai
from google import genai
from google.genai import types
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

model_embeddings = 'models/gemini-embedding-001'
model_geracao = 'gemini-2.0-flash'

def gerarBuscarConsulta(consulta, dataset):
    embedding_consulta = generativeai.embed_content(
        model=model_embeddings,
        content=consulta,
        task_type="retrieval_query",
    )
    produtos_escalares = np.dot(
        np.stack(dataset["Embeddings"]),
        embedding_consulta['embedding']
    )
    indice = np.argmax(produtos_escalares)
    return dataset.iloc[indice]['Conteúdo']

def melhorarResposta(inputText):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=inputText),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""
    Você é o MazeBot, um assistente educacional especializado em algoritmos de busca e inteligência artificial aplicados a labirintos.

Seu objetivo é ajudar estudantes de tecnologia a entender de forma clara e visual os algoritmos Backtracking, Dijkstra e Q-Learning, além de conceitos relacionados à plataforma educacional de labirintos.
Utilize exclusivamente o conteúdo recuperado da base de conhecimento para responder à consulta do usuário.
Reescreva as informações de forma natural, didática e acessível, como um professor explicando para um aluno.
Use analogias com labirintos quando possível para tornar a explicação mais concreta.
Não invente informações que não estejam presentes no contexto fornecido.
Seja objetivo, claro e encorajador — lembre que o objetivo é reduzir a dificuldade no aprendizado de algoritmos.
"""),
        ],
    )

    response = client.models.generate_content(
        model=model_geracao,
        contents=contents,
        config=generate_content_config,
    )

    return response.text