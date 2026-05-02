"""Wrapper léger Ollama → A2A spoke.

Pattern : ce process est un spoke A2A simple qui forward les tasks reçues
au Ollama local et retourne le résultat. Compatible avec le hub A2A qui
sera sur Ministar Linux (`100.98.18.76:3000`).

Usage :
  # Sur le host avec Ollama (DARKSTAR ou Ministar Linux ou autre)
  python scripts/ollama_a2a_spoke.py \\
      --hub http://100.98.18.76:3000 \\
      --ollama http://127.0.0.1:11434 \\
      --name darkstar-ollama \\
      --url http://100.73.222.64:11434 \\
      --models gemma4:26b qwen3:4b nomic-embed-text qwen3.6:35b-a3b-q4_K_M

Le wrapper :
1. Démarre, s'enregistre au hub via POST /api/a2a/agents/register avec son
   AgentCard (skills annoncées : 1 par modèle Ollama dispo).
2. Loop : heartbeat au hub toutes les 30s.
3. (V0) Pas de listening pour tasks entrantes — le hub appelle directement
   Ollama via l'URL annoncée. Le wrapper ne fait que register + heartbeat.
4. (V1+) Le wrapper exposerait son propre endpoint A2A /tasks/send qui
   wrappe Ollama avec lifecycle propre. Optionnel — direct Ollama call
   suffit pour V0.

Dépendances : `requests` (Python standard pour la plupart des installs),
`urllib3` implicit. Aucune autre.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

try:
    import requests
except ImportError:
    print("FATAL: requests not installed. pip install requests", file=sys.stderr)
    sys.exit(1)


def list_ollama_models(ollama_url: str) -> List[str]:
    """Retourne la liste des modèles dispos sur le Ollama local."""
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        print(f"warn: ollama tags fetch failed: {e}", file=sys.stderr)
        return []


def _normalize_model_id(model: str) -> str:
    """Sanitize Ollama model name for use as a skill id (no `:`, no `/`).

    `gemma4:26b-instruct` -> `gemma4-26b-instruct`.
    """
    return model.replace(":", "-").replace("/", "-").replace(".", "")


def build_agent_card(name: str, url: str, models: List[str], host: str | None = None) -> Dict[str, Any]:
    """Construit l'AgentCard A2A à partir des modèles Ollama dispos.

    Convention nomenclature (proposée par Claude/MINISTAR 2026-05-02) :
    `name = "ollama-<host>"` (e.g. "ollama-darkstar"), `skills.id` =
    "embed-<model_id>" ou "chat-<model_id>" pour qu'un router du hub
    puisse choisir par skill direct ou par host.
    """
    skills = []
    for m in models:
        mid = _normalize_model_id(m)
        if "embed" in m.lower():
            skills.append({
                "id": f"embed-{mid}",
                "name": f"Embeddings ({m})",
                "description": f"Génère des embeddings via Ollama {m}",
                "inputModes": ["text/plain"],
                "outputModes": ["application/json"],
            })
        else:
            skills.append({
                "id": f"chat-{mid}",
                "name": f"Chat ({m})",
                "description": f"Chat completion via Ollama {m}",
                "inputModes": ["text/plain"],
                "outputModes": ["text/plain"],
            })
    description = f"Ollama spoke ({len(models)} models)"
    if host:
        description += f" on {host}"
    return {
        "name": name,
        "description": description,
        "url": url,
        "version": "0.2.0",
        "skills": skills,
        "capabilities": {
            "streaming": True,   # Ollama supporte le streaming natif
            "pushNotifications": False,
        },
    }


_session = requests.Session()
_csrf_token: str | None = None


def _get_csrf(hub_url: str) -> str | None:
    """Fetch a CSRF token via GET /api/csrf-token (Double Submit Cookie pattern).

    Code Buddy server applies CSRF middleware even in --no-auth mode. Token must
    accompany POST requests in header X-CSRF-Token (and as cookie, but session
    handles that).
    """
    global _csrf_token
    try:
        r = _session.get(f"{hub_url}/api/csrf-token", timeout=5)
        if r.status_code == 200:
            data = r.json()
            _csrf_token = data.get("token") or data.get("csrfToken")
            return _csrf_token
    except requests.exceptions.RequestException:
        pass
    return None


def register(hub_url: str, name: str, url: str, card: Dict[str, Any]) -> bool:
    """POST au hub /api/a2a/agents/register avec CSRF token."""
    token = _get_csrf(hub_url)
    headers = {"X-CSRF-Token": token} if token else {}
    try:
        r = _session.post(
            f"{hub_url}/api/a2a/agents/register",
            json={"name": name, "url": url, "card": card},
            headers=headers,
            timeout=10,
        )
        if r.status_code == 200:
            print(f"[register] OK : {name} -> {hub_url}", flush=True)
            return True
        if r.status_code == 404:
            print(
                f"[register] hub {hub_url} n'a pas /agents/register (patch pas mergé) "
                f"— skip, on continue avec discovery passive",
                file=sys.stderr,
            )
            return False
        print(f"[register] FAIL {r.status_code}: {r.text[:200]}", file=sys.stderr)
        return False
    except requests.exceptions.RequestException as e:
        print(f"[register] connection failed: {e}", file=sys.stderr)
        return False


def heartbeat(hub_url: str, name: str) -> bool:
    """POST au hub /api/a2a/agents/:name/heartbeat avec CSRF."""
    token = _csrf_token or _get_csrf(hub_url)
    headers = {"X-CSRF-Token": token} if token else {}
    try:
        r = _session.post(
            f"{hub_url}/api/a2a/agents/{name}/heartbeat",
            json={},
            headers=headers,
            timeout=5,
        )
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hub", required=True, help="Hub A2A URL, ex http://100.98.18.76:3000")
    p.add_argument("--ollama", default="http://127.0.0.1:11434", help="Local Ollama URL")
    p.add_argument("--name", required=True, help="Nom unique du spoke (recommandé: ollama-<host>, ex ollama-darkstar)")
    p.add_argument("--host-tag", default=None, help="Tag du host (darkstar, ministar-linux, etc.) pour skills id et description")
    p.add_argument("--url", required=True, help="URL publique du Ollama local sur le tailnet")
    p.add_argument("--models", nargs="+", default=None,
                   help="Liste explicite de modèles à annoncer (sinon auto-détecte via /api/tags)")
    p.add_argument("--heartbeat-interval", type=int, default=30, help="Secondes entre heartbeats")
    args = p.parse_args()

    models = args.models or list_ollama_models(args.ollama)
    if not models:
        print("FATAL : aucun modèle Ollama disponible (auto-détection KO)", file=sys.stderr)
        sys.exit(2)
    print(f"[init] {len(models)} models annoncés : {', '.join(models)}", flush=True)

    card = build_agent_card(args.name, args.url, models, host=args.host_tag)
    registered = register(args.hub, args.name, args.url, card)

    if not registered:
        print(
            f"[init] hub register KO. Mode passif : on attend que le hub "
            f"discover via curl direct sur {args.url}/api/tags",
            file=sys.stderr,
        )
        # Sans register actif, le wrapper reste utile : le hub peut découvrir
        # passivement via discovery card si on expose une.
        sys.exit(0)

    # Heartbeat loop
    print(f"[loop] heartbeat every {args.heartbeat_interval}s", flush=True)
    try:
        while True:
            time.sleep(args.heartbeat_interval)
            ok = heartbeat(args.hub, args.name)
            if not ok:
                print(f"[heartbeat] FAIL — hub down ou register expired ?", file=sys.stderr)
                # Tentative de re-register
                register(args.hub, args.name, args.url, card)
            else:
                print(f"[heartbeat] OK ({time.strftime('%H:%M:%S')})", flush=True)
    except KeyboardInterrupt:
        print("\n[exit] interrupt — bye", flush=True)


if __name__ == "__main__":
    main()
