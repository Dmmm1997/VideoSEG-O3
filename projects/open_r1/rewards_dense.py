import re
import ast
import torch

def _parse_vtg_content(text: str, max_frames: int):
    pattern = r"<(select|answer)>.*?<VTG>(.*?)</VTG>.*?</\1>"
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return None

    try:
        _, vtg_content = matches[-1]
        data = ast.literal_eval(vtg_content.strip())

        s = int(data.get('start', 0))
        e = int(data.get('end', 0))
        k = int(data.get('keyframe', 0))

        s = max(0, min(s, max_frames))
        e = max(0, min(e, max_frames))
        k = max(0, min(k, max_frames - 1))

        return s, e, k
    except Exception:
        return None

def lr_vtg_iou_reward_dense(completions, mask_existence, **kwargs):
    rewards = []

    R_PASS = 0.5
    R_FAIL = -0.5
    IOU_THRESH = 0.5

    for m_e, turns in zip(mask_existence, completions):
        if isinstance(turns, str): turns = [turns]

        if not turns:
            rewards.append(R_FAIL)
            continue

        T = len(m_e)

        last_turn = turns[-1]

        parsed = _parse_vtg_content(last_turn, max_frames=T)

        if parsed is None:
            rewards.append(R_FAIL)
            continue

        s, e, _ = parsed

        if s >= e:
            rewards.append(R_FAIL)
            continue

        pred_mask = torch.zeros_like(m_e)
        pred_mask[s:e] = 1

        inter = (pred_mask * m_e).sum().float()
        union = (pred_mask).gt(0.5).sum().float()

        if m_e.sum() == 0:
            iou = 1.0 # No valid GT mask
        else:
            iou = (inter / (union + 1e-6)).item()

        if iou < IOU_THRESH:
            rewards.append(R_FAIL)
        else:
            rewards.append(R_PASS*iou)

    return rewards

def lr_progressive_keyframe_reward_dense(per_round_keyframe_ious, **kwargs):
    rewards = []

    get_val = lambda x: sum(x)/len(x) if isinstance(x, list) and x else (float(x) if x else 0.0)

    for kf_turns in per_round_keyframe_ious:
        if len(kf_turns) < 2:
            rewards.append(0.0)
            continue

        curr_val = get_val(kf_turns[-1])

        prev_vals = [get_val(x) for x in kf_turns[:-1]]
        prev_best_val = max(prev_vals) if prev_vals else 0.0

        delta = curr_val - prev_best_val

        if delta > 0.1:
            score = min((delta-0.05) * 10, 2.0)
        if delta > 0.05:
            score = 0.5
        elif delta > -0.05:
            score = 0.0
        else:
            score = -1.0

        rewards.append(score)

    return rewards

def lr_keyframe_vs_mask_reward_dense(per_round_maskious, per_round_keyframe_ious, **kwargs):
    rewards = []

    get_val = lambda x: sum(x)/len(x) if isinstance(x, list) and x else (float(x) if x else 0.0)

    for mask_turns, kf_turns in zip(per_round_maskious, per_round_keyframe_ious):
        if len(mask_turns) < 2 or len(kf_turns) < 2:
            rewards.append(0.0)
            continue

        curr_kf_val = get_val(kf_turns[-1])
        cur_mask_avgs = get_val(mask_turns[-1])

        delta = curr_kf_val - cur_mask_avgs

        if delta > 0.2:
            score = min((delta-0.1) * 5, 2.0)
        if delta > 0.1:
            score = 0.5
        elif delta > -0.05:
            score = 0.0
        else:
            score = -1.0

        rewards.append(score)

    return rewards

def lr_keyframe_reward_dense(per_round_keyframe_ious, **kwargs):
    rewards = []

    def _get_score(iou_val):
        if iou_val > 0.4: return min((iou_val - 0.4) * 5, 2.5)
        elif iou_val > 0.2: return 0.0
        else: return -1.0

    for sample_rounds in per_round_keyframe_ious:
        last_round_data = sample_rounds[-1]

        if isinstance(last_round_data, list):
            val = float(last_round_data[0]) if last_round_data else 0.0
        else:
            val = float(last_round_data) if last_round_data is not None else 0.0

        rewards.append(_get_score(val))

    return rewards

def lr_maskiou_reward_dense(per_round_maskious, **kwargs):
    rewards = []

    def _get_score(iou_val):
        if iou_val > 0.4: return min((iou_val - 0.4) * 3, 1.5)
        elif iou_val > 0.2: return 0.0
        else: return -1.0

    for sample_rounds in per_round_maskious:
        if not sample_rounds:
            rewards.append(0.0)
            continue

        last_round_data = sample_rounds[-1]

        if isinstance(last_round_data, list):
            if not last_round_data:
                final_score = -1.0
            else:
                scores = [_get_score(iou) for iou in last_round_data]
                final_score = sum(scores) / len(scores)
        else:
            val = float(last_round_data) if last_round_data is not None else 0.0
            final_score = _get_score(val)

        rewards.append(final_score)

    return rewards

def termination_reward(completions, per_round_keyframe_ious, **kwargs):
    """
    Termination Reward (With Efficiency Check):
    1. Answer Mode (<answer>):
       - IoU > 0.7: +1.0 (Good termination)
       - IoU < 0.3: -1.0 (Bad termination/Hallucination)
       - 0.3 <= IoU <= 0.7: 0.0

    2. Select Mode (<select>):
       - IoU > 0.85: -1.0 (Penalty! Found a great frame but didn't answer. "Over-optimization")
       - IoU <= 0.85: 0.0 (Reasonable to continue searching)

    3. Format Error: 0.0
    """
    rewards = []

    def get_val(x):
        if isinstance(x, list):
            return sum(x)/len(x) if x else 0.0
        return float(x) if x is not None else 0.0

    for turns, kf_turns in zip(completions, per_round_keyframe_ious):
        if not turns or not kf_turns:
            rewards.append(0.0)
            continue

        last_text = turns[-1]
        current_iou = get_val(kf_turns[-1])

        has_answer = re.search(r"<answer>.*?</answer>", last_text, re.DOTALL)
        has_select = re.search(r"<select>.*?</select>", last_text, re.DOTALL)

        if has_answer:
            if current_iou > 0.8:
                rewards.append(1.0)
            elif current_iou < 0.2:
                rewards.append(-1.0)
            else:
                rewards.append(0.0)

        elif has_select:
            if current_iou < 0.5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        else:
            rewards.append(0.0)

    return rewards
