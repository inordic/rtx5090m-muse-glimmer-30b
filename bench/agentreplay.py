#!/usr/bin/env python3
"""agentreplay: an agent-shaped session against an OpenAI server. A long fixed system prompt (rules + reference
text, about TARGET tokens) and 20 tools; each step resends the whole history, as an agent client does. The model's
tool calls get simulated results (agenttest's workspace). Records per step: prompt/cached/new tokens, prefill time,
output tokens, decode rate, wall time, and whether the step made a tool call.
  python3 agentreplay.py --port 8091 --label L [--target 20000] [--steps 8] [--tasks 2] [--strength low]"""
import argparse, json, sys, time, urllib.request
sys.path.insert(0, "/root/bench")
import agenttest as at

def post(base, body):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    return out, time.time() - t

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8091)
ap.add_argument("--label", required=True); ap.add_argument("--target", type=int, default=20000)
ap.add_argument("--steps", type=int, default=8); ap.add_argument("--tasks", type=int, default=2)
ap.add_argument("--strength", default=None, help="sent as chat_template_kwargs.reasoning_strength (else server default)")
ap.add_argument("--max-tokens", type=int, default=2048)
a = ap.parse_args()
base = f"http://{a.host}:{a.port}"
doc = open("/opt/models/muse-glimmer/corpus/eval_neutral.txt").read()[: int(a.target * 4.2)]
system = ("You are Atlas, an autonomous operations agent for a small company. Follow the runbook below. Use tools to act; "
          "keep answers short. Never invent tool results.\n\n## Runbook and reference material\n" + doc)
rows = []
for ti in range(a.tasks):
    ws = at.Workspace()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": at.TASKS[ti]}]
    for step in range(a.steps):
        body = {"model": "x", "messages": msgs, "tools": at.TOOLS, "max_tokens": a.max_tokens, "temperature": 1.0, "top_k": 64, "top_p": 0.95, "min_p": 0.0, "seed": 100 * ti + step}
        if a.strength: body["chat_template_kwargs"] = {"reasoning_strength": a.strength}
        out, wall = post(base, body)
        m = out["choices"][0]["message"]; tm = out.get("timings") or {}; u = out.get("usage") or {}
        calls = m.get("tool_calls") or []
        rows.append({"task": ti, "step": step, "prompt": (tm.get("prompt_n") or 0) + (tm.get("cache_n") or 0), "cached": tm.get("cache_n"),
                     "new": tm.get("prompt_n"), "prefill_ms": round(tm.get("prompt_ms") or 0), "out": u.get("completion_tokens"),
                     "decode": round(tm.get("predicted_per_second") or 0, 1), "wall": round(wall, 2), "call": bool(calls),
                     "reasoning_chars": len(m.get("reasoning_content") or ""), "draft": tm.get("draft_n"), "acc": tm.get("draft_n_accepted")})
        r = rows[-1]
        print(f"task{ti} step{step}: prompt {r['prompt']} cached {r['cached']} new {r['new']} prefill {r['prefill_ms']} ms | out {r['out']} at {r['decode']} tok/s | wall {r['wall']} s | call {r['call']} | reasoning {r['reasoning_chars']} chars", flush=True)
        msgs.append({"role": "assistant", "content": m.get("content") or "", **({"tool_calls": calls} if calls else {})})
        if not calls:
            break
        for c in calls:
            args, problem, _ = at.validate(c["function"]["name"], c["function"].get("arguments")) if c["function"]["name"] in at.SCHEMAS else (None, "unknown tool", [])
            res = f"error: {problem}" if problem else ws.result(c["function"]["name"], args)
            msgs.append({"role": "tool", "tool_call_id": c.get("id", "c1"), "content": res})
first = [r for r in rows if r["step"] == 0]; later = [r for r in rows if r["step"] > 0]
avg = lambda xs, k: round(sum(x[k] or 0 for x in xs) / max(1, len(xs)), 2)
summary = {"label": a.label, "steps": len(rows), "first_wall": avg(first, "wall"), "later_wall": avg(later, "wall"), "later_prefill_ms": avg(later, "prefill_ms"),
           "later_cached_frac": round(sum((x["cached"] or 0) for x in later) / max(1, sum(x["prompt"] for x in later)), 3),
           "out_tokens_per_step": avg(rows, "out"), "decode": avg(rows, "decode"), "calls": sum(r["call"] for r in rows)}
print("REPLAY", json.dumps(summary), flush=True)
open("/root/bench/replay.jsonl", "a").write(json.dumps({**summary, "rows": rows}) + "\n")
