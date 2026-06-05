import re
import os
import warnings
from datetime import datetime
from projects.open_r1.utils import extract_bbox_answer, compute_iou
import json
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

def log(content, sol, other_info, reward, tag=None):
    log_dir = os.getenv("LOG_DIR", None)
    os.makedirs(log_dir, exist_ok=True)
    if log_dir is None:
        warnings.warn("LOG_DIR is not set, log will not be saved")
        return
    log_path = os.path.join(log_dir, f"{tag}.log")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    with open(log_path, "a") as f:
        try:
            f.write(f"------------- {current_time} {tag} reward: {reward} -------------\n")
            f.write(f"Content: {content}\n")
            f.write(f"Solution: {sol}\n")
            if other_info is not None:
                for k, v in other_info.items():
                    f.write(f"{k}: {v}\n")
        except:
            f.write("writeing error")

def format_reward(completions, pattern, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, flags=re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def think_format_reward(completions, **kwargs):
    """<think>...</think>\n<answer>...</answer>"""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return format_reward(completions, pattern)

def think_select_format_reward(completions, **kwargs):
    """<think>...</think>\n<answer>...</answer>"""
    pattern = r"<think>.*?</think>\s*<(answer|select)>(.*?)</\1>"
    return format_reward(completions, pattern)

def pr1_grounding_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<\|object_ref_start\|>.*?<\|object_ref_end\|><\|box_start\|>.*?<\|box_end\|><\|im_end\|>"
    completion_contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def pr1_grounding_format_reward_max_0p1(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<|object_ref_start|>.*?<|object_ref_end|><|box_start|>.*?<|box_end|><|im_end|>"
    completion_contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [0.1 if match else 0.0 for match in matches]

def pr1_grounding_format_reward_reason(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\n<\|object_ref_start\|>.*?<\|object_ref_end\|><\|box_start\|>.*?<\|box_end\|><\|im_end\|>"
    completion_contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def pr1_grounding_reward(completions, solution, **kwargs):
    rewards = []
    contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    for completion, sol in zip(contents, solution):
        bbox = extract_bbox_answer(completion)
        iou = compute_iou(bbox, eval(sol))
        rewards.append(iou**2)
        log(completion + f"\nBounding box: {bbox}", sol, None, iou**2, "pr1_grounding_reward")
    return rewards

def seg_existence_reward(completions, solution, **kwargs):
    rewards = []
    contents = [completion[0]["content"].replace("<|endoftext|>", "") for completion in completions]
    length = len(contents)
    for completion, sol in zip(contents, solution):
        if "[SEG]" in completion:
            rewards.append(1.0/length)
        else:
            rewards.append(0)
    return rewards

def strict_format_reward(completions, **kwargs):
    def validate(content):
        if not (m := re.search(r"<think>.*?</think>\s*<(answer|select)>(.*?)</\1>", content, re.DOTALL)):
            return 0.0

        body = m.group(2)
        vtg_m = re.search(r"<VTG>(.*?)</VTG>", body, re.DOTALL)
        refseg_m = re.search(r"<RefSeg>.*?</RefSeg>", body, re.DOTALL)

        try:
            if (refseg_m and vtg_m and
                {"start", "end", "keyframe"} <= json.loads(vtg_m.group(1)).keys()):
                return 1.0
        except: pass
        return 0.0

    return [validate(c[0]["content"]) for c in completions]

def pr_maskiou_reward(per_round_maskious, **kwargs):
    rewards = []

    def _get_per_round_score(iou_val):
        if iou_val > 0.9:
            return 1.0
        elif iou_val > 0.75:
            return 0.75
        elif iou_val > 0.5:
            return 0.5
        elif iou_val > 0.25:
            return -0.5
        else:
            return -1.0

    for sample_rounds in per_round_maskious:
        round_scores = []

        for round_data in sample_rounds:
            if isinstance(round_data, list):
                scores = [_get_per_round_score(iou) for iou in round_data]

                if len(scores) > 0:
                    r_validmask = sum(scores) / len(scores)
                else:
                    r_validmask = -1.0
            else:
                r_validmask = _get_per_round_score(round_data)

            round_scores.append(r_validmask)

        if len(round_scores) > 0:
            final_reward = sum(round_scores) / len(round_scores)
        else:
            final_reward = 0.0

        rewards.append(final_reward)

    return rewards

def pr_keyframe_valid_reward(per_round_maskious, per_round_keyframe_ious, **kwargs):
    rewards = []

    get_val = lambda x: sum(x)/len(x) if isinstance(x, list) and x else (float(x) if x else 0.0)

    for mask_turns, kf_turns in zip(per_round_maskious, per_round_keyframe_ious):
        scores = []

        if len(mask_turns) > 1 and len(kf_turns) > 1:
            for m_data, k_data in zip(mask_turns[1:], kf_turns[1:]):
                i_m, i_k = get_val(m_data), get_val(k_data)

                if i_k > i_m + 0.1 or (i_k > 0.85 and i_k > i_m + 0.02):
                    scores.append(1.0)
                elif i_k > i_m + 0.05:
                    scores.append(0.5)
                elif i_k > i_m - 0.05:
                    scores.append(0.0)
                elif i_k > i_m - 0.1:
                    scores.append(-0.5)
                else:
                    scores.append(-1.0)

        rewards.append(sum(scores) / len(scores) if scores else 0.0)

    return rewards

def mr_progressive_maskiou_reward(per_round_maskious, **kwargs):
    rewards = []

    to_list = lambda x: x if isinstance(x, list) else [float(x)]

    for sample_turns in per_round_maskious:
        if len(sample_turns) < 2:
            rewards.append(0.0)
            continue

        sample_score_sum = 0.0

        prev_bests = to_list(sample_turns[0])[:]

        for turn_data in sample_turns[1:]:
            curr_ious = to_list(turn_data)

            current_round_img_scores = []

            for i, (curr_val, prev_best_val) in enumerate(zip(curr_ious, prev_bests)):
                if curr_val > prev_best_val + 0.1 or (curr_val > prev_best_val + 0.02 and curr_val > 0.85):
                    s = 1.0
                elif curr_val > prev_best_val + 0.05:
                    s = 0.5
                elif curr_val > prev_best_val -0.05:
                    s = 0.0
                elif curr_val > prev_best_val - 0.1:
                    s = -0.5
                else:
                    s = -1.0

                current_round_img_scores.append(s)

                prev_bests[i] = max(prev_best_val, curr_val)

            if len(current_round_img_scores) > 0:
                round_reward = sum(current_round_img_scores) / len(current_round_img_scores)
            else:
                round_reward = 0.0

            sample_score_sum += round_reward

        rewards.append(sample_score_sum)

    return rewards

def mr_progressive_keyframe_reward(per_round_keyframe_ious, **kwargs):
    rewards = []

    def get_val(x):
        if isinstance(x, list):
            return float(x[0]) if x else 0.0
        return float(x) if x is not None else 0.0

    for sample_turns in per_round_keyframe_ious:
        if len(sample_turns) < 3:
            rewards.append(0.0)
            continue

        sample_score_sum = 0.0

        prev_best = get_val(sample_turns[1])

        for turn_data in sample_turns[2:]:
            curr_val = get_val(turn_data)

            if curr_val > prev_best + 0.1 or (curr_val > prev_best + 0.02 and curr_val > 0.85):
                s = 1.0

            elif curr_val > prev_best + 0.05:
                s = 0.5

            elif curr_val > prev_best - 0.05:
                s = 0.0

            elif curr_val > prev_best - 0.1:
                s = -0.5

            else:
                s = -1.0

            sample_score_sum += s

            prev_best = max(prev_best, curr_val)

        rewards.append(sample_score_sum)

    return rewards

def keyframe_maskiou_reward(per_round_maskious, per_round_keyframe_ious, **kwargs):
    """
    Reward based on the Keyframe IoU of the final round.

    Logic:
    1. Base Reward: The continuous value of the last keyframe IoU.
    2. Discrete Bonus: If Keyframe IoU > Video IoU + 0.2 (Significant improvement).
    3. Discrete Penalty: If Keyframe IoU < 0.1 (Absolute failure).

    """
    def calc_reward(vid_ious, kf_ious):
        if not kf_ious:
            return 0.0

        final_kf_iou = float(kf_ious[-1])

        final_vid_iou = float(vid_ious[-1]) if vid_ious else 0.0

        reward = final_kf_iou

        bonus = 0.0

        if final_kf_iou > (final_vid_iou + 0.1) and final_kf_iou>0.7:
            bonus = 0.3

        elif final_kf_iou < (final_vid_iou - 0.1):
            bonus = -0.3

        if len(kf_ious)>2 and final_kf_iou < kf_ious[-2]-0.1:
            bonus = -0.3

        return reward + bonus

    return [
        calc_reward(v_ious, k_ious)
        for v_ious, k_ious in zip(per_round_maskious, per_round_keyframe_ious)
    ]

def step_format_reward_cot(completions, **kwargs):
    """
    Step Format Reward:
    Checks if the <think> block in EACH round contains specific keywords:
    - 'Step 1'
    - 'Step 2'
    - 'Step 3'
    - 'Action'

    Scoring:
    - Each keyword present adds 0.25 to the round's score.
    - Full score (1.0) requires all 4 keywords.
    - Case-insensitive (e.g., 'step 1', 'STEP 1', 'Step 1' are all valid).
    - The final reward is the AVERAGE score across all valid rounds in the sample.
    """
    rewards = []

    # Regex to extract content inside <think>...</think>
    think_pattern = r"<think>(.*?)</think>"

    # Required keywords (lowercase for case-insensitive comparison)
    required_keywords = ["step 1", "step 2", "step 3", "action"]

    for turns in completions:
        # Handle string input vs list input
        if isinstance(turns, str):
            turns = [turns]

        round_scores = []

        for turn_text in turns:
            # 1. Extract <think> content
            match = re.search(think_pattern, turn_text, re.DOTALL | re.IGNORECASE)

            if match:
                content = match.group(1).lower() # Convert to lower case for matching

                # 2. Count how many keywords appear
                hit_count = 0
                for kw in required_keywords:
                    if kw in content:
                        hit_count += 1

                # Calculate score for this round: 0/4, 1/4, 2/4, 3/4, or 4/4
                round_score = hit_count / len(required_keywords)
                round_scores.append(round_score)
            else:
                # If no <think> tag found, score for this round is 0
                round_scores.append(0.0)

        # 3. Calculate Average across all rounds
        if len(round_scores) > 0:
            avg_score = sum(round_scores) / len(round_scores)
            rewards.append(avg_score)
        else:
            rewards.append(0.0)

    return rewards

def strict_format_reward_cot(completions, **kwargs):
    """
    Format Reward:
    1. Last round must be <answer>, preceding rounds must be <select>.
    2. EVERY round (select or answer) must follow <think>...</think> <tag>...</tag>.
    3. EVERY round must contain valid <VTG> JSON and <RefSeg> containing '[SEG]'.
    """
    rewards = []
    pat = r"<think>.*?</think>\s*<(answer|select)>(.*?)</\1>"

    for turns in completions:
        if isinstance(turns, str): turns = [turns]
        if not turns:
            rewards.append(0.0)
            continue

        valid = True
        for i, content in enumerate(turns):
            # 1. Check basic structure <think>...<tag>
            m = re.search(pat, content, re.DOTALL)
            if not m:
                valid = False; break

            tag, body = m.groups()

            # 3. Check Inner Content (Applied to BOTH select and answer)
            vtg = re.search(r"<VTG>(.*?)</VTG>", body, re.DOTALL)
            ref = re.search(r"<RefSeg>(.*?)</RefSeg>", body, re.DOTALL)

            if not (vtg and ref):
                valid = False; break

            # Check JSON keys
            try:
                if not {"start", "end", "keyframe"} <= json.loads(vtg.group(1)).keys():
                    valid = False; break
            except:
                valid = False; break

            # Check [SEG] token
            if "[SEG]" not in ref.group(1):
                valid = False; break

        rewards.append(1.0 if valid else -1.0)

    return rewards

def vtg_iou_reward(completions, mask_existence, **kwargs):
    rewards = []

    answer_pattern = r"<think>.*?</think>\s*<(answer|select)>(.*?)</\1>"

    vtg_pattern = r"<VTG>(.*?)</VTG>"

    for m_e, c in zip(mask_existence, completions):
        try:
            content = c[0]["content"]
            ans_match = re.search(answer_pattern, content, re.DOTALL)

            vtg_match = re.search(vtg_pattern, ans_match.group(2), re.DOTALL)

            data = ast.literal_eval(vtg_match.group(1))

            s, e = int(data['start']), int(data['end'])

            pred_mask = torch.zeros_like(m_e)

            s, e = max(0, s), min(len(m_e), e)
            if s < e:
                pred_mask[s:e] = 1

            intersection = (pred_mask * m_e).sum()
            pred_interval = max(e - s, 0)

            iou = (intersection / (pred_interval + 1e-6)).item() if pred_interval > 0 else 0.0
            if m_e.sum()==0:
                iou = 1.0

            rewards.append(iou * 0.3)

        except (AttributeError, ValueError, SyntaxError, KeyError):
            rewards.append(0.0)

    return rewards

def vtg_iou_reward_cot(completions, mask_existence, **kwargs):
    rewards = []

    R_PASS = 0.5
    R_FAIL = -0.5
    IOU_THRESH = 0.5

    for m_e, turns in zip(mask_existence, completions):
        if isinstance(turns, str): turns = [turns]

        T = len(m_e)
        turn_rewards = []

        for turn_text in turns:
            parsed = _parse_vtg_content(turn_text, max_frames=T)

            if parsed is None:
                turn_rewards.append(R_FAIL)
                continue

            s, e, _ = parsed

            if s >= e:
                turn_rewards.append(R_FAIL)
                continue

            pred_mask = torch.zeros_like(m_e)
            pred_mask[s:e] = 1

            inter = (pred_mask * m_e).sum().float()
            union = (pred_mask).gt(0.5).sum().float()

            if m_e.sum()==0:
                iou = 1.0 # No valid GT mask, all range is acceptable
            else:
                iou = (inter / (union + 1e-6)).item()

            if iou > IOU_THRESH:
                turn_rewards.append(R_PASS)
            else:
                turn_rewards.append(R_FAIL)

        if len(turn_rewards) > 0:
            avg_reward = sum(turn_rewards) / len(turn_rewards)
        else:
            avg_reward = R_FAIL

        rewards.append(avg_reward)

    return rewards

def vtg_keyframe_reward_cot(completions, mask_existence, **kwargs):
    rewards = []

    R_KEY_HIT = 1.0
    R_KEY_MISS = -1.0

    for m_e, turns in zip(mask_existence, completions):
        if isinstance(turns, str): turns = [turns]

        T = len(m_e)
        turn_rewards = []

        for turn_text in turns:
            parsed = _parse_vtg_content(turn_text, max_frames=T)

            if parsed is None:
                turn_rewards.append(R_KEY_MISS)
                continue

            _, _, k = parsed

            if m_e[k] > 0.5:
                turn_rewards.append(R_KEY_HIT)
            else:
                turn_rewards.append(R_KEY_MISS)

        if len(turn_rewards) > 0:
            avg_reward = sum(turn_rewards) / len(turn_rewards)
        else:
            avg_reward = R_KEY_MISS

        rewards.append(avg_reward)

    return rewards

def lr_vtg_iou_reward(completions, mask_existence, **kwargs):
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

        if iou > IOU_THRESH:
            rewards.append(R_PASS)
        else:
            rewards.append(R_FAIL)

    return rewards

def lr_vtg_keyframe_reward(completions, mask_existence, **kwargs):
    rewards = []

    R_KEY_HIT = 1.0
    R_KEY_MISS = -1.0

    for m_e, turns in zip(mask_existence, completions):
        if isinstance(turns, str): turns = [turns]

        if not turns:
            rewards.append(R_KEY_MISS)
            continue

        T = len(m_e)

        last_turn = turns[-1]

        parsed = _parse_vtg_content(last_turn, max_frames=T)

        if parsed is None:
            rewards.append(R_KEY_MISS)
            continue

        _, _, k = parsed

        if m_e[k] > 0.5:
            rewards.append(R_KEY_HIT)
        else:
            rewards.append(R_KEY_MISS)

    return rewards

def lr_progressive_keyframe_reward(per_round_keyframe_ious, **kwargs):
    rewards = []

    get_val = lambda x: sum(x)/len(x) if isinstance(x, list) and x else (float(x) if x else 0.0)

    for kf_turns in per_round_keyframe_ious:
        if len(kf_turns) < 2:
            rewards.append(0.0)
            continue

        curr_val = get_val(kf_turns[-1])

        prev_vals = [get_val(x) for x in kf_turns[:-1]]
        prev_best_val = max(prev_vals) if prev_vals else 0.0

        if curr_val > prev_best_val + 0.1 and curr_val > 0.6:
            score = 1.0
        elif curr_val > prev_best_val + 0.05 and curr_val > 0.4:
            score = 0.5
        elif curr_val > prev_best_val - 0.05:
            score = 0.0
        elif curr_val > prev_best_val - 0.1:
            score = -0.5
        else:
            score = -1.0

        rewards.append(score)

    return rewards

def lr_keyframe_vs_best_mask_reward(per_round_maskious, per_round_keyframe_ious, **kwargs):
    rewards = []

    get_val = lambda x: sum(x)/len(x) if isinstance(x, list) and x else (float(x) if x else 0.0)

    for mask_turns, kf_turns in zip(per_round_maskious, per_round_keyframe_ious):
        if len(mask_turns) < 2 or len(kf_turns) < 2:
            rewards.append(0.0)
            continue

        curr_kf_val = get_val(kf_turns[-1])

        prev_mask_avgs = [get_val(m_data) for m_data in mask_turns[:-1]]

        prev_best_mask_val = max(prev_mask_avgs) if prev_mask_avgs else 0.0

        if curr_kf_val > prev_best_mask_val + 0.1 and curr_kf_val > 0.6:
            score = 1.0

        elif curr_kf_val > prev_best_mask_val + 0.05 and curr_kf_val > 0.4:
            score = 0.5

        elif curr_kf_val > prev_best_mask_val - 0.05:
            score = 0.0

        elif curr_kf_val > prev_best_mask_val - 0.1:
            score = -0.5

        else:
            score = -1.0

        rewards.append(score)

    return rewards

def lr_keyframe_vs_mask_reward(per_round_maskious, per_round_keyframe_ious, **kwargs):
    rewards = []

    get_val = lambda x: sum(x)/len(x) if isinstance(x, list) and x else (float(x) if x else 0.0)

    for mask_turns, kf_turns in zip(per_round_maskious, per_round_keyframe_ious):
        if len(mask_turns) < 2 or len(kf_turns) < 2:
            rewards.append(0.0)
            continue

        curr_kf_val = get_val(kf_turns[-1])

        cur_mask_avgs = get_val(mask_turns[-1])

        if curr_kf_val > cur_mask_avgs + 0.1 or (curr_kf_val > 0.85 and curr_kf_val > cur_mask_avgs + 0.02):
            score = 1.0

        elif curr_kf_val > cur_mask_avgs + 0.05:
            score = 0.5

        elif curr_kf_val > cur_mask_avgs - 0.05:
            score = 0.0

        elif curr_kf_val > cur_mask_avgs - 0.1:
            score = -0.5

        else:
            score = -1.0

        rewards.append(score)

    return rewards

def lr_keyframe_reward(per_round_keyframe_ious, **kwargs):
    rewards = []

    def _get_score(iou_val):
        if iou_val > 0.9: return 1.0
        elif iou_val > 0.75: return 0.75
        elif iou_val > 0.5: return 0.5
        elif iou_val > 0.25: return -0.5
        else: return -1.0

    for sample_rounds in per_round_keyframe_ious:
        if not sample_rounds or len(sample_rounds) < 2:
            rewards.append(0.0)
            continue

        last_round_data = sample_rounds[-1]

        if isinstance(last_round_data, list):
            val = float(last_round_data[0]) if last_round_data else 0.0
        else:
            val = float(last_round_data) if last_round_data is not None else 0.0

        rewards.append(_get_score(val))

    return rewards

def lr_maskiou_reward(per_round_maskious, **kwargs):
    rewards = []

    def _get_score(iou_val):
        if iou_val > 0.9: return 1.0
        elif iou_val > 0.75: return 0.75
        elif iou_val > 0.5: return 0.5
        elif iou_val > 0.25: return -0.5
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

def simple_lr_maskiou_reward(per_round_maskious, **kwargs):
    rewards = []

    for sample_rounds in per_round_maskious:
        if not sample_rounds:
            rewards.append(0.0)
            continue

        last_round_data = sample_rounds[-1]

        if isinstance(last_round_data, list):
            if not last_round_data:
                final_score = 0.0
            else:
                scores = [iou for iou in last_round_data]
                final_score = sum(scores) / len(scores)
        else:
            val = float(last_round_data) if last_round_data is not None else 0.0
            final_score = val

        rewards.append(final_score)

    return rewards

def simple_lr_keyframe_reward(per_round_keyframe_ious, **kwargs):
    rewards = []

    for sample_rounds in per_round_keyframe_ious:
        if not sample_rounds or len(sample_rounds) < 2:
            rewards.append(0.0)
            continue

        last_round_data = sample_rounds[-1]

        val = float(last_round_data) if last_round_data is not None else 0.0

        rewards.append(val)

    return rewards
