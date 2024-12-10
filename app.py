from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv
from .src.main import make_agent, evaluate_by_llm_with_criteria

# 環境変数のロード
load_dotenv()

# OpenAI APIキーの設定
openai.api_key = os.getenv("OPENAI_API_KEY")

# Flaskアプリの作成
app = Flask(__name__)
CORS(app)  # CORSを有効化

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')

    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    try:
        # エージェントを初期化
        p_tool_configs = [
            "./docs/tools/code.yaml",
            "./docs/tools/incident_rag.yaml",
            "./docs/tools/knowledge_rag.yaml",
            "./docs/tools/raw_text.yaml",
            "./docs/tools/search.yaml",
        ]
        p_react_prompt = "./docs/react_base_prompt.md"
        k = 3
        llm = openai.ChatCompletion.create(model="gpt-4o-mini")

        agent = make_agent(p_tool_configs, p_react_prompt, k, llm)

        # エージェントで質問を処理
        ai_response = agent.run(user_input)

        return jsonify({"response": ai_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    pred = data.get('pred', '')
    answer = data.get('answer', '')
    question = data.get('question', None)
    criteria_with_weights = data.get('criteria_with_weights', [])

    if not (pred and answer and criteria_with_weights):
        return jsonify({"error": "Required fields: pred, answer, criteria_with_weights"}), 400

    try:
        result = evaluate_by_llm_with_criteria(
            pred=pred,
            answer=answer,
            llm=openai.ChatCompletion.create(model="gpt-4o-mini"),
            question=question,
            criteria_with_weights=criteria_with_weights
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
