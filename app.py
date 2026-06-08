from flask import Flask, jsonify, request
import numpy as np
import google.generativeai as generativeai
from google import genai
from google.genai import types
import pickle
from flask_cors import CORS
from dotenv import load_dotenv
import os
from geminiFunctions import gerarBuscarConsulta, melhorarResposta

load_dotenv()

app = Flask(__name__)
CORS(app)

modeloEmbeddings = pickle.load(open('datasetEmbeddings.pkl', 'rb'))

chave_secreta = os.getenv('GEMINI_API_KEY')
generativeai.configure(api_key=chave_secreta)

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "chatbot": "MazeBot",
        "descricao": "Assistente educacional de algoritmos de busca e IA aplicados a labirintos",
        "uso": "Envie um POST para /api com JSON: { 'consulta': 'sua pergunta aqui' }"
    })

@app.route("/api", methods=["POST"])
def results():
    
    data = request.get_json(force=True)

    if not data or "consulta" not in data:
        return jsonify({"error": "Campo 'consulta' é obrigatório no corpo da requisição"}), 400

    consulta = data["consulta"]

    if not consulta.strip():
        return jsonify({"error": "A consulta não pode ser vazia"}), 400

    resultado = gerarBuscarConsulta(consulta, modeloEmbeddings)
    prompt = f"Consulta: {consulta}\nResposta: {resultado}"
    response = melhorarResposta(prompt)

    return jsonify({"mensagem": response})

if __name__ == "__main__":
    app.run(debug=True)