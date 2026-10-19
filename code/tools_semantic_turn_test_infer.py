#!/usr/bin/env python3
"""semantic-turn-taking (anyreach-ai/semantic-turn-taking, 0.5B Qwen2.5-Instruct fine-tune)
推理 3000 test 文本 → 输出 (N, 5) probs，列序 [C, NA, I, BC, T]

输入: --text_dir <dir>  目录下每个 *.json 是 {utterances: [{channel_id, text, ...}]}
输出: --output_probs / --output_ids
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# action token id 来自 added_tokens.json
ACTION_IDS = {
    "start_speaking":     151666,
    "continue_listening": 151665,
    "start_listening":    151667,
    "continue_speaking":  151668,
}

# 列序 [C, NA, I, BC, T]
# C 列没有直接对应（C = continuation by current speaker），仍用 1-(others 和) 或保留 0 让 fusion 兼容
# 实际：C 我们用 continue_listening + continue_speaking 之和（双向都"继续"=C）
# I = start_listening (interrupt by other speaker)
# BC = continue_speaking (backchannel)
# T = start_speaking (turn-shift)
# NA = 1 - 上面四个 max（剩余概率质量）

def fmt_prompt(utts: List[Dict]) -> str:
    msgs = []
    for u in utts:
        role = "user" if int(u.get("channel_id", 1)) == 1 else "assistant"
        text = str(u.get("text", "")).strip()
        if text:
            msgs.append(f"<|im_start|>{role}\n{text}<|im_end|>")
    head = "<|im_start|>system\nYou are predicting turn-taking actions.<|im_end|>\n"
    return head + "\n".join(msgs) + "\n<|predict|>"

def to_5col(action_probs: Dict[str, float]) -> List[float]:
    p_T  = action_probs["start_speaking"]
    p_NA = action_probs["continue_listening"]
    p_I  = action_probs["start_listening"]
    p_BC = action_probs["continue_speaking"]
    # C 列：当前说话方继续 ~= continue_listening + continue_speaking (任一通道继续说)
    # 但要避免 > 1, 取均值 + softmax 修正
    p_C = (p_NA + p_BC) / 2.0
    return [p_C, p_NA, p_I, p_BC, p_T]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--output_probs", required=True)
    ap.add_argument("--output_ids", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_length", type=int, default=2048)
    args = ap.parse_args()

    print(f"[semturn] loading model from {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.float16).to(args.device)
    model.eval()

    txt_dir = Path(args.text_dir)
    seg_ids = sorted([p.stem for p in txt_dir.glob("*.json")])
    print(f"[semturn] {len(seg_ids)} segments, device={args.device}", flush=True)

    probs_out = np.zeros((len(seg_ids), 5), dtype=np.float32)
    action_names = list(ACTION_IDS.keys())
    action_ids_t = torch.tensor([ACTION_IDS[n] for n in action_names], device=args.device)

    log_every = max(1, len(seg_ids) // 20)
    with torch.no_grad():
        for i, sid in enumerate(seg_ids):
            data = json.load(open(txt_dir / f"{sid}.json"))
            prompt = fmt_prompt(data.get("utterances", []))
            inp = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
            out = model(**inp)
            last_logits = out.logits[0, -1, :]
            action_logits = last_logits[action_ids_t]
            sm = torch.softmax(action_logits, dim=0).cpu().tolist()
            ap_d = dict(zip(action_names, sm))
            probs_out[i] = to_5col(ap_d)
            if (i + 1) % log_every == 0 or i == len(seg_ids) - 1:
                print(f"[semturn] {i+1}/{len(seg_ids)}", flush=True)

    Path(args.output_probs).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_probs, probs_out)
    with open(args.output_ids, "w") as f:
        for sid in seg_ids:
            f.write(sid + "\n")
    print(f"[semturn] wrote probs={args.output_probs} shape={probs_out.shape}", flush=True)

if __name__ == "__main__":
    main()
