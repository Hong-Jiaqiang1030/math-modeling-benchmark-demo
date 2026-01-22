import json
import re
from typing import Any, Dict, List

from demo.benchmark.tools.llm_chat import call_chat, has_credentials, DEFAULT_LLM_MODEL, DEFAULT_EXPERT_MODEL


def _normalize_item(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。！？、；：“”‘’（）()【】\[\]{}<>《》~`@#$%^&*_+=|\\\\/:;\"'?,.!-]", "", s)
    return s


def _jaccard(a: List[str], b: List[str]) -> float:
    a_set = {_normalize_item(x) for x in (a or []) if _normalize_item(x)}
    b_set = {_normalize_item(x) for x in (b or []) if _normalize_item(x)}
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def _semantic_overlap_jaccard_binary(
    *,
    context: str,
    category: str,
    pred_items: List[str],
    gt_items: List[str],
    expert_model: str,
) -> float:
    """
    Expert-judged semantic Jaccard with binary coverage decisions.

    The expert judges, for each item, whether it is semantically covered/matched (0 or 1),
    allowing paraphrases/synonyms. We then compute a Jaccard-style score:
      intersection = #covered ground-truth items
      union = |GT| + #unmatched predicted items

    This avoids returning 0 just because of different wording.
    """
    if not pred_items and not gt_items:
        return 1.0
    if not pred_items or not gt_items:
        return 0.0

    raw = call_chat(
        expert_model,
        [
            {
                "role": "system",
                "content": (
                    "你是严格的评审专家。你将比较两个要素列表是否在语义上对应（允许同义改写/不同表述）。\n"
                    "请对每个条目输出0或1：\n"
                    "- gt_covered[i]=1 表示GT第i条的语义在pred列表中被覆盖；否则为0。\n"
                    "- pred_matched[j]=1 表示pred第j条能对应到某个GT条目；否则为0。\n"
                    "只输出JSON，不要输出任何解释。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"类别：{category}\n\n"
                    f"文本语境（可参考）：{context}\n\n"
                    f"GT列表（ground truth）：{json.dumps(gt_items, ensure_ascii=False)}\n\n"
                    f"Pred列表（model output）：{json.dumps(pred_items, ensure_ascii=False)}\n\n"
                    "输出JSON格式：\n"
                    "{\n"
                    '  "gt_covered": [0或1，长度等于GT列表长度],\n'
                    '  "pred_matched": [0或1，长度等于Pred列表长度]\n'
                    "}"
                ),
            },
        ],
        temperature=0.0,
    )
    obj = _extract_json(raw)
    if not obj:
        # Degrade to lexical Jaccard if expert output can't be parsed.
        return _jaccard(pred_items, gt_items)

    gt_covered = obj.get("gt_covered")
    pred_matched = obj.get("pred_matched")
    if not (isinstance(gt_covered, list) and isinstance(pred_matched, list)):
        return _jaccard(pred_items, gt_items)
    if len(gt_covered) != len(gt_items) or len(pred_matched) != len(pred_items):
        return _jaccard(pred_items, gt_items)

    def _to01(x) -> int:
        try:
            xi = int(x)
        except Exception:
            return 0
        return 1 if xi != 0 else 0

    gt_covered_01 = [_to01(x) for x in gt_covered]
    pred_matched_01 = [_to01(x) for x in pred_matched]

    intersection = sum(gt_covered_01)
    unmatched_pred = sum(1 - x for x in pred_matched_01)
    union = len(gt_items) + unmatched_pred
    if union <= 0:
        return 1.0
    return float(intersection) / float(union)


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of a JSON object from LLM output.
    Accepts:
    - raw JSON
    - ```json ... ```
    - extra text around a single top-level {...}
    """
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    else:
        m2 = re.search(r"(\{.*\})", t, flags=re.DOTALL)
        if m2:
            t = m2.group(1).strip()
    return json.loads(t)


def _mock_real_model() -> Dict[str, Any]:
    return {
        "variables": ["人口密度", "是否雨天", "是否有专用自行车道", "投放单车数量", "单车使用率"],
        "assumptions": [
            "比较中其他因素（价格、节假日等）保持不变或影响较小",
            "雨天对所有区域的使用率都有一致的负向影响",
            "专用自行车道会增强投放数量对使用率的影响",
        ],
        "constraints": ["投放数量受预算/运维能力限制", "区域道路与停放空间容量有限"],
        "concept_triples": [
            ["人口密度", "正相关", "使用率"],
            ["雨天", "负相关", "使用率"],
            ["专用自行车道", "增强相关性", "投放数量与使用率"],
            ["投放数量", "正相关", "使用率"],
        ],
    }


def evaluate_real_model(
    context: str,
    ground_truth: Dict[str, Any],
    llm_model: str = DEFAULT_LLM_MODEL,
    expert_model: str = DEFAULT_EXPERT_MODEL,
) -> Dict[str, Any]:
    """
    Tier-2 (Real Model) evaluation (combined):
    - Ask LLM to output a structured "real model" (variables/assumptions/constraints/concept triples)
    - Similarity: expert-judged semantic Jaccard (binary per-item coverage) if expert available; else lexical Jaccard
    - Expert LLM scoring: reasonableness/completeness/simplicity/innovativenss (1-5) + one-sentence comment
    """
    can_call_pred = has_credentials(llm_model)
    can_call_expert = has_credentials(expert_model)

    if can_call_pred:
        raw = call_chat(
            llm_model,
            [
                {
                    "role": "system",
                    "content": (
                        "你是数学建模研究助理。请从文本中抽象出“真实模型（Real Model）”，"
                        "以结构化JSON输出，不要输出任何解释。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"文本：{context}\n\n"
                        "请输出一个JSON对象，必须包含这些字段：\n"
                        "- variables: 字符串列表（关键变量/要素）\n"
                        "- assumptions: 字符串列表（核心假设）\n"
                        "- constraints: 字符串列表（主要约束/限制）\n"
                        "- concept_triples: 三元组列表，每个三元组为 [头实体, 关系, 尾实体]\n\n"
                        "只输出JSON："
                    ),
                },
            ],
        )
        try:
            pred = _extract_json(raw)
        except Exception:
            # If the model didn't follow format, degrade gracefully.
            pred = {"raw": raw, "variables": [], "assumptions": [], "constraints": [], "concept_triples": []}
    else:
        pred = _mock_real_model()

    gt_vars = ground_truth.get("variables", [])
    gt_assumptions = ground_truth.get("assumptions", [])
    gt_constraints = ground_truth.get("constraints", [])
    gt_triples = ["|".join(map(str, t)) for t in ground_truth.get("concept_triples", []) or []]

    pr_vars = pred.get("variables", []) or []
    pr_assumptions = pred.get("assumptions", []) or []
    pr_constraints = pred.get("constraints", []) or []
    pr_triples = ["|".join(map(str, t)) for t in pred.get("concept_triples", []) or []]

    if can_call_expert:
        sim = {
            "variables_jaccard": _semantic_overlap_jaccard_binary(
                context=context, category="variables", pred_items=pr_vars, gt_items=gt_vars, expert_model=expert_model
            ),
            "assumptions_jaccard": _semantic_overlap_jaccard_binary(
                context=context,
                category="assumptions",
                pred_items=pr_assumptions,
                gt_items=gt_assumptions,
                expert_model=expert_model,
            ),
            "constraints_jaccard": _semantic_overlap_jaccard_binary(
                context=context,
                category="constraints",
                pred_items=pr_constraints,
                gt_items=gt_constraints,
                expert_model=expert_model,
            ),
            "concept_triples_jaccard": _semantic_overlap_jaccard_binary(
                context=context,
                category="concept_triples",
                pred_items=pr_triples,
                gt_items=gt_triples,
                expert_model=expert_model,
            ),
        }
        sim_mode = "expert_semantic_binary"
    else:
        sim = {
            "variables_jaccard": _jaccard(pr_vars, gt_vars),
            "assumptions_jaccard": _jaccard(pr_assumptions, gt_assumptions),
            "constraints_jaccard": _jaccard(pr_constraints, gt_constraints),
            "concept_triples_jaccard": _jaccard(pr_triples, gt_triples),
        }
        sim_mode = "lexical_jaccard"
    sim["mean_jaccard"] = sum(sim.values()) / 4.0

    expert: Dict[str, Any] = {}
    if can_call_expert:
        expert_raw = call_chat(
            expert_model,
            [
                {
                    "role": "system",
                    "content": (
                        "你是领域评审专家。请对候选真实模型进行多维度打分。"
                        "只输出JSON，不要解释。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"文本：{context}\n\n"
                        f"候选输出（pred）：{json.dumps(pred, ensure_ascii=False)}\n\n"
                        "请输出JSON，字段如下：\n"
                        "- reasonableness: 1-5\n"
                        "- completeness: 1-5\n"
                        "- simplicity: 1-5\n"
                        "- innovativeness: 1-5\n"
                        "- comment: 一句中文评语\n"
                    ),
                },
            ],
        )
        try:
            expert = _extract_json(expert_raw)
        except Exception:
            expert = {"raw": expert_raw}
    else:
        # Simple heuristic in offline mode.
        base = sim["mean_jaccard"]
        score = 1 + int(round(base * 4))
        expert = {
            "reasonableness": score,
            "completeness": score,
            "simplicity": max(1, 5 - score + 1),
            "innovativeness": 2,
            "comment": "离线模式：用相似度近似评估，供演示用。",
        }

    return {"predicted": pred, "similarity": sim, "similarity_mode": sim_mode, "expert": expert}
