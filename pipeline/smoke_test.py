"""
Local, CPU-only smoke test for the pilot harness logic used in
colab_budget_oracle_pilot.ipynb -- catches bugs in grading / feature
extraction / activation shapes *before* burning Colab GPU time.

Uses a tiny (a few MB) throwaway model, so results are mechanical
sanity checks only -- not expected to solve any real math.
"""
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "sshleifer/tiny-gpt2"
LAYER_FRACS = [0.25, 0.5, 0.8]

BOXED_RE = re.compile(r"\\boxed\{")


def extract_boxed(text: str):
    matches = list(BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip()


def normalize(ans):
    if ans is None:
        return ""
    a = ans.strip()
    a = a.replace("\\left", "").replace("\\right", "")
    a = a.replace(" ", "").replace("\\!", "")
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return a.rstrip(".")


def grade(generated_text, gold_answer):
    pred = extract_boxed(generated_text)
    if pred is None:
        m = re.search(r"final answer is[:\s]*\$?([^\n$]+)\$?", generated_text, re.IGNORECASE)
        pred = m.group(1) if m else None
    return normalize(pred) == normalize(gold_answer) and normalize(pred) != ""


def max_brace_depth(s):
    depth = maxd = 0
    for ch in s:
        if ch == "{":
            depth += 1
            maxd = max(maxd, depth)
        elif ch == "}":
            depth = max(0, depth - 1)
    return maxd


def surface_features(problem):
    return {
        "char_len": len(problem),
        "word_len": len(problem.split()),
        "latex_cmd_count": len(re.findall(r"\\[a-zA-Z]+", problem)),
        "dollar_count": problem.count("$"),
        "has_asy_figure": "[asy]" in problem,
        "brace_depth": max_brace_depth(problem),
        "digit_count": sum(ch.isdigit() for ch in problem),
        "eq_count": problem.count("="),
    }


def test_grading():
    cases = [
        ("blah blah Final answer: \\boxed{28}", "28", True),
        ("... $\\boxed{ 2 }$ vertical asymptotes", "2", True),
        ("The answer is \\boxed{\\frac{1}{2}}", "\\dfrac{1}{2}", True),
        ("I have no idea", "5", False),
        ("\\boxed{3}", "5", False),
    ]
    for text, gold, expected in cases:
        got = grade(text, gold)
        status = "OK" if got == expected else "FAIL"
        print(f"[{status}] grade({text!r}, {gold!r}) = {got} (expected {expected})")
        assert got == expected, f"grading logic bug on: {text!r} vs {gold!r}"
    print("grading logic: all cases passed\n")


def test_pipeline_mechanics():
    subset_path = DATA_DIR / "math_pool_large.json"
    problems = json.loads(subset_path.read_text(encoding="utf-8"))[:2]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    n_layers = model.config.n_layer if hasattr(model.config, "n_layer") else model.config.num_hidden_layers
    layer_idx = sorted(set(int(round(f * n_layers)) for f in LAYER_FRACS))
    print(f"tiny model has {n_layers} layers, sampling at {layer_idx}")

    for row in problems:
        phi = surface_features(row["problem"])
        assert phi["char_len"] > 0

        prompt = f"Problem:\n{row['problem']}\n\nSolution:"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        acts = {}
        for li in layer_idx:
            h = out.hidden_states[li][0]
            acts[f"layer_{li}"] = h.mean(dim=0).float().tolist()
            assert len(acts[f"layer_{li}"]) == model.config.n_embd if hasattr(model.config, "n_embd") else True

        gen_inputs = tokenizer([prompt] * 2, return_tensors="pt", padding=True)
        with torch.no_grad():
            gen_out = model.generate(
                **gen_inputs,
                max_new_tokens=16,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_only = gen_out[:, gen_inputs["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        successes = [grade(t, row["answer"]) for t in texts]

        print(f"id={row['id']} phi={phi} n_activation_layers={len(acts)} "
              f"gen_lens={[len(t) for t in texts]} graded={successes}")

    print("\npipeline mechanics: no crashes, shapes look right")


if __name__ == "__main__":
    test_grading()
    test_pipeline_mechanics()
