from langchain.chat_models import ChatOpenAI

from src.main import (
    evaluate_by_llm_with_criteria,
    make_agent,
    plot_tools_count,
    tool_usage_tracker,
)
from src.tool_component import tool_usage_tracker
from src.utils import load_criteria_with_weights


def main():
    question = "新幹線に関する騒音の環境基準はどのように設定されているか、具体的に説明してください。"
    answer = "新幹線の騒音は、環境省の基準に基づき、音源対策により沿線の住宅の集合度合いに応じて75デシベル以下を目標としています。"

    k = 3
    p_tool_configs = [
        "./docs/tools/code.yaml",
        "./docs/tools/incident_rag.yaml",
        "./docs/tools/knowledge_rag.yaml",
        "./docs/tools/raw_text.yaml",
        "./docs/tools/search.yaml",
    ]
    p_eval_configs = [
        "./docs/evals/accuracy.yaml",
        "./docs/evals/calculation.yaml",
        "./docs/evals/evidence.yaml",
        "./docs/evals/expertise.yaml",
        "./docs/evals/expression.yaml",
        "./docs/evals/relevance.yaml",
    ]
    p_react_prompt = "./docs/react_base_prompt.md"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = make_agent(p_tool_configs, p_react_prompt, k, llm)
    criteria_with_weights = load_criteria_with_weights(p_eval_configs)

    prediction = agent.run(question)
    print(prediction)

    eval_result = evaluate_by_llm_with_criteria(
        prediction,
        answer,
        llm,
        question=question,
        criteria_with_weights=criteria_with_weights,
    )
    print(eval_result)

    # ツール使用状況の可視化
    current_tool_usage_counts = tool_usage_tracker.get_counts()
    if len(current_tool_usage_counts) > 0:
        plot_tools_count(current_tool_usage_counts)


if __name__ == "__main__":
    main()
