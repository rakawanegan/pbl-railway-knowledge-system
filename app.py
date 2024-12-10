from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv
from src.main import make_agent, evaluate_by_llm_with_criteria

# 環境変数のロード
load_dotenv()

# OpenAI APIキーの設定
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

# アプリ起動時に一度だけ初期化
p_tool_configs = [
    "./docs/tools/code.yaml",
    "./docs/tools/incident_rag.yaml",
    "./docs/tools/knowledge_rag.yaml",
    "./docs/tools/raw_text.yaml",
    "./docs/tools/search.yaml",
]
p_react_prompt = "./docs/react_base_prompt.md"
k = 3

# llmを一度だけ定義
def llm(messages):
    return openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )

# agentも一度だけ初期化してグローバルスコープに保持
agent = make_agent(p_tool_configs, p_react_prompt, k, llm)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    try:
        # 毎回新規に初期化せず、既存のagentとllmを使いまわす
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
        # 同じllmを利用する
        result = evaluate_by_llm_with_criteria(
            pred=pred,
            answer=answer,
            llm=llm,
            question=question,
            criteria_with_weights=criteria_with_weights
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
