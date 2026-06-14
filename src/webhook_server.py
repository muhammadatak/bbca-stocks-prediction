"""
Webhook Server — Menerima alert dari AlertManager lalu trigger GitHub Actions.
Alur: Prometheus Alert → AlertManager → webhook ini → GitHub API (repository_dispatch)
"""
import json
import os
import subprocess
from datetime import datetime

from fastapi import FastAPI, Request

app = FastAPI()
TRIGGER_LOG = os.path.join(os.path.dirname(__file__), "..", "trigger_log.jsonl")


def trigger_github_actions(reason: str, detail: str = ""):
    """Panggil GitHub API repository_dispatch, atau fallback ke gh CLI."""
    print(f"\n🚨 Trigger retrain — reason: {reason}")
    print(f"   detail: {detail}")

    # Coba gh CLI dulu (paling sederhana untuk demo)
    try:
        result = subprocess.run(
            ["gh", "workflow", "run", "retrain-on-alert.yml",
             "-f", f"trigger_reason={reason}"],
            capture_output=True, text=True, timeout=30
        )
        print(f"   gh exit: {result.returncode}")
        print(f"   stdout: {result.stdout.strip()}")
        if result.stderr:
            print(f"   stderr: {result.stderr.strip()}")
    except FileNotFoundError:
        print("   ⚠️  gh CLI not found — printing payload for manual trigger")
        print(f"   payload: {json.dumps({'trigger_reason': reason, 'detail': detail})}")

    # Log ke file
    with open(TRIGGER_LOG, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason,
            "detail": detail,
        }) + "\n")


@app.post("/alert")
async def receive_alert(request: Request):
    """Endpoint yang dipanggil AlertManager."""
    body = await request.json()
    print("\n📩 Alert received:")
    print(json.dumps(body, indent=2))

    alerts = body.get("alerts", [])
    for alert in alerts:
        labels = alert.get("labels", {})
        scenario = labels.get("scenario", "unknown")
        alertname = labels.get("alertname", "")
        status = alert.get("status", "")

        if status == "firing":
            trigger_github_actions(scenario, alertname)

    return {"status": "ok", "alerts_processed": len(alerts)}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/log")
def get_log():
    """Lihat history trigger."""
    if not os.path.exists(TRIGGER_LOG):
        return {"log": []}
    with open(TRIGGER_LOG) as f:
        return {"log": [json.loads(l) for l in f if l.strip()]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
