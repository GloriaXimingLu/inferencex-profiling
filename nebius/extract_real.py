#!/usr/bin/env python3
"""Build a REAL-TOKEN replay schedule from Sierra tau-bench trajectories.

Unlike the upstream toolkit (which keeps only call *lengths* and replays synthetic
tokens), this reconstructs the real prompt TEXT for each LLM call:
  prompt(call k) = system policy + serialized conversation history up to call k.
The model's natural text makes spec-decode acceptance realistic (vs ~19% on random
tokens), and the real shared-system-prompt + append-only history preserve the
prefix-cache structure.

Output: real_schedule/<model>__<domain>.jsonl, one object per simulation:
  {sim_id, model, domain, agent_system, user_system,
   turns: [{stream: agent|user, text, is_call: bool, output_len: int}]}
The replay reconstructs prompt(call i) = system(stream) + "".join(turn.text for
turns[:i]); the call's own message text is the OUTPUT, not part of the prompt.

Caveat: not byte-identical to the source model's exact rendering (tokenizer +
tool-schema serialization differ) — it is real, coherent agent text of ~the right
shape, which is what matters for spec-decode / cache measurement.

Usage: python extract_real.py <model-substr> <domain> [<domain> ...]
"""
import sys, os, json, urllib.request
BASE = "https://sierra-tau-bench-public.s3.us-west-2.amazonaws.com/submissions"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_schedule")

def gj(u):
    with urllib.request.urlopen(u, timeout=120) as r: return json.load(r)

def render_turn(m):
    """Real text a message contributes to the running conversation context."""
    role = m.get("role"); c = m.get("content")
    if role == "user":
        return "User: " + (c if isinstance(c, str) else json.dumps(c)) + "\n"
    if role == "tool":
        return "Tool result: " + (c if isinstance(c, str) else json.dumps(c)) + "\n"
    if role == "assistant":
        parts = []
        if isinstance(c, str) and c:
            parts.append(c)
        asg = m.get("audio_script_gold")
        if isinstance(asg, str) and asg:
            parts.append(asg)            # spoken-turn transcript
        tcs = m.get("tool_calls")
        if tcs:
            for tc in tcs:
                fn = (tc.get("function") or {})
                parts.append(f"[tool_call {fn.get('name','')}({fn.get('arguments','')})]")
        return "Assistant: " + (" ".join(parts) if parts else "") + "\n"
    return ""

def build(model_substr, domains):
    os.makedirs(OUT, exist_ok=True)
    subs = gj(f"{BASE}/manifest.json").get("submissions", [])
    sub = next((s for s in subs if model_substr in s), None)
    if not sub:
        print(f"no submission matches '{model_substr}'"); return
    model = sub.split("_sierra")[0].split("_")[0]
    tf = gj(f"{BASE}/{sub}/submission.json").get("trajectory_files", {})
    user_system = "You are a user-simulator interacting with a customer-service agent.\n"
    for dom in domains:
        if dom not in tf:
            print(f"  {dom}: not in {sub}"); continue
        traj = gj(f"{BASE}/{sub}/trajectories/{tf[dom]}")
        part = f"{model}__{dom}"
        path = os.path.join(OUT, f"{part}.jsonl")
        nsim = ncall = 0
        with open(path, "w") as out:
            for sim in traj.get("simulations", []):
                agent_system = sim.get("policy") or ""
                if not isinstance(agent_system, str):
                    agent_system = json.dumps(agent_system)
                turns = []
                for m in sim.get("messages", []):
                    txt = render_turn(m)
                    u = m.get("usage")
                    is_call = bool(u and m.get("role") in ("assistant", "user"))
                    stream = "agent" if m.get("role") == "assistant" else "user"
                    ol = int(u.get("completion_tokens") or 0) if u else 0
                    turns.append({"stream": stream, "text": txt,
                                  "is_call": is_call, "output_len": ol})
                if not any(t["is_call"] for t in turns):
                    continue
                ncall += sum(t["is_call"] for t in turns); nsim += 1
                out.write(json.dumps({"sim_id": sim.get("id", ""), "model": model,
                    "domain": dom, "agent_system": agent_system,
                    "user_system": user_system, "turns": turns}) + "\n")
        print(f"  wrote {path}  sims={nsim} calls={ncall}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: extract_real.py <model-substr> <domain> [domain ...]"); sys.exit(1)
    build(sys.argv[1], sys.argv[2:])
