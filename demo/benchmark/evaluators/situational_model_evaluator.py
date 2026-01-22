import os
import re
from collections import Counter
from typing import List, Dict, Any, Optional

from benchmark.utils.env import load_env

# Load .env early so OPENAI_API_KEY is available even when running from `demo/`.
load_env()


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    # Keep CJK chars and ASCII alphanumerics; drop most punctuation/whitespace.
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。！？、；：“”‘’（）()【】\[\]{}<>《》~`@#$%^&*_+=|\\\\/:;\"'?,.!-]", "", s)
    return s


def _tokenize(s: str) -> List[str]:
    s = _normalize_text(s)
    if not s:
        return []
    # For Chinese, character-level tokens are a simple robust baseline.
    return list(s)


def _f1(pred: str, gold: str) -> float:
    pred_toks = _tokenize(pred)
    gold_toks = _tokenize(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    pred_cnt = Counter(pred_toks)
    gold_cnt = Counter(gold_toks)
    common = sum((pred_cnt & gold_cnt).values())
    if common == 0:
        return 0.0
    precision = common / max(1, len(pred_toks))
    recall = common / max(1, len(gold_toks))
    return (2 * precision * recall) / (precision + recall)


def _call_openai_chat(model: str, messages: List[Dict[str, str]]) -> str:
    # Lazy import so the module can run without openai installed during static inspection.
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def _mock_answer(question: str) -> str:
    # Deterministic fallback that keeps the demo runnable without API access.
    if "雨天" in question:
        return "雨天会显著降低所有区域的使用率。"
    if "两个" in question or "正面因素" in question:
        return "人口密度高，以及有专用自行车道。"
    return "因为专用自行车道会强化投放数量与使用率的相关性，使投放更有效。"


def _expert_comment_from_f1(f1: float) -> str:
    if f1 >= 0.85:
        return "回答准确捕捉了关键细节。"
    if f1 >= 0.55:
        return "回答大体正确，但部分细节不够完整。"
    return "回答可能忽略了重要关系或关键信息。"


def evaluate_situational_model(
    context: str,
    questions: List[str],
    ground_truth_answers: List[str],
    llm_model: str = "gpt-4o",
    expert_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Tier-1 (Situation Model) evaluation:
    - LLM answers closed questions about the context
    - For each question, compute token-overlap F1 vs ground truth
    - Ask an "expert LLM" for a one-sentence critique (fallback to heuristic if no API key)
    """
    if len(questions) != len(ground_truth_answers):
        raise ValueError("questions and ground_truth_answers must have the same length")

    has_key = bool(os.getenv("OPENAI_API_KEY"))
    per_q = []

    for q, gt in zip(questions, ground_truth_answers):
        if has_key:
            predicted = _call_openai_chat(
                llm_model,
                [
                    {"role": "system", "content": "你是严谨的研究助理，请基于给定文本回答问题，尽量简洁准确。"},
                    {"role": "user", "content": f"文本：{context}\n\n问题：{q}\n\n请直接给出答案："},
                ],
            )
        else:
            predicted = _mock_answer(q)

        f1 = _f1(predicted, gt)

        expert_comment: Optional[str]
        if has_key:
            expert_comment = _call_openai_chat(
                expert_model,
                [
                    {"role": "system", "content": "你是评审专家，只输出一句中文评语（不打分、不解释过程）。"},
                    {
                        "role": "user",
                        "content": (
                            f"文本：{context}\n\n问题：{q}\n\n参考答案：{gt}\n\n模型回答：{predicted}\n\n"
                            "请给出一句评语，指出回答的优点或缺陷："
                        ),
                    },
                ],
            )
            expert_comment = expert_comment.splitlines()[0].strip() if expert_comment else ""
        else:
            expert_comment = _expert_comment_from_f1(f1)

        per_q.append(
            {
                "question": q,
                "ground_truth_answer": gt,
                "predicted_answer": predicted,
                "f1": f1,
                "expert_comment": expert_comment,
            }
        )

    mean_f1 = sum(x["f1"] for x in per_q) / max(1, len(per_q))
    return {"per_question": per_q, "mean_f1": mean_f1}
