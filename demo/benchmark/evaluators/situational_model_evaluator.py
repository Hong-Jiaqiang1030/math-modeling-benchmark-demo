import re
from collections import Counter
from typing import List, Dict, Any, Optional

from demo.benchmark.tools.llm_chat import call_chat, has_credentials, DEFAULT_LLM_MODEL, DEFAULT_EXPERT_MODEL


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of a single JSON object from LLM output.
    Returns None if parsing fails.
    """
    import json

    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    else:
        m2 = re.search(r"(\{.*\})", t, flags=re.DOTALL)
        if m2:
            t = m2.group(1).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


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


def _semantic_judge(
    context: str,
    question: str,
    ground_truth: str,
    predicted: str,
    expert_model: str,
) -> Optional[Dict[str, Any]]:
    """
    Use expert LLM to do semantic evaluation. Expected JSON:
    {"score": 0.0-1.0, "comment": "..."}
    """
    raw = call_chat(
        expert_model,
        [
            {
                "role": "system",
                "content": (
                    "你是严格的评审专家。你将比较“模型回答”和“参考答案”在给定文本语境下的语义一致性。"
                    "请按以下标准给出score（0到1的小数）：\n"
                    "- 1.0：语义等价/完全正确（允许同义改写）\n"
                    "- 0.7：大体正确但遗漏了关键要点\n"
                    "- 0.4：部分相关但有明显缺失或轻微错误\n"
                    "- 0.0：不相关或关键事实错误\n"
                    "只输出JSON，不要输出多余文字。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"文本：{context}\n\n"
                    f"问题：{question}\n\n"
                    f"参考答案：{ground_truth}\n\n"
                    f"模型回答：{predicted}\n\n"
                    "请输出：{\"score\":..., \"comment\":\"一句中文评语\"}"
                ),
            },
        ],
        temperature=0.0,
    )
    obj = _extract_json(raw)
    if not obj:
        return None
    score = obj.get("score", None)
    comment = obj.get("comment", "")
    try:
        score_f = float(score)
    except Exception:
        return None
    score_f = max(0.0, min(1.0, score_f))
    return {"score": score_f, "comment": str(comment or "").strip()}


def evaluate_situational_model(
    context: str,
    questions: List[str],
    ground_truth_answers: List[str],
    llm_model: str = DEFAULT_LLM_MODEL,
    expert_model: str = DEFAULT_EXPERT_MODEL,
) -> Dict[str, Any]:
    """
    Tier-1 (Situation Model) evaluation:
    - LLM answers closed questions about the context
    - Expert LLM performs semantic evaluation vs ground truth (score 0..1 + one-sentence comment)
    - Fallback (offline): character-overlap F1 + heuristic comment
    """
    if len(questions) != len(ground_truth_answers):
        raise ValueError("questions and ground_truth_answers must have the same length")

    can_call_pred = has_credentials(llm_model)
    can_call_expert = has_credentials(expert_model)
    per_q = []

    for q, gt in zip(questions, ground_truth_answers):
        if can_call_pred:
            predicted = call_chat(
                llm_model,
                [
                    {"role": "system", "content": "你是严谨的研究助理，请基于给定文本回答问题，尽量简洁准确。"},
                    {"role": "user", "content": f"文本：{context}\n\n问题：{q}\n\n请直接给出答案："},
                ],
            )
        else:
            predicted = _mock_answer(q)

        lexical_f1 = _f1(predicted, gt)

        semantic_score: float
        expert_comment: str
        if can_call_expert:
            judged = _semantic_judge(
                context=context,
                question=q,
                ground_truth=gt,
                predicted=predicted,
                expert_model=expert_model,
            )
            if judged is not None:
                semantic_score = float(judged["score"])
                expert_comment = str(judged["comment"])
            else:
                semantic_score = lexical_f1
                expert_comment = "专家评估输出解析失败，已回退到字符重叠分数。"
        else:
            semantic_score = lexical_f1
            expert_comment = _expert_comment_from_f1(lexical_f1)

        per_q.append(
            {
                "question": q,
                "ground_truth_answer": gt,
                "predicted_answer": predicted,
                "semantic_score": semantic_score,
                "lexical_f1": lexical_f1,
                "expert_comment": expert_comment,
            }
        )

    mean_semantic = sum(x["semantic_score"] for x in per_q) / max(1, len(per_q))
    return {"per_question": per_q, "mean_semantic_score": mean_semantic}
