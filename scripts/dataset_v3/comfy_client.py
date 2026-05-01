"""Client minimal HTTP/WS pour piloter un serveur ComfyUI local.

API ComfyUI :
  POST /prompt        body {"prompt": <workflow_dict>, "client_id": <uuid>}  → {"prompt_id": ..., "number": ...}
  GET  /history/<id>                                                            → {"<id>": {"outputs": {<node_id>: {...}}, "status": ...}}
  GET  /view?filename=...&subfolder=...&type=output                             → bytes (PNG/JPG/MP4)
  WS   /ws?clientId=<uuid>                                                      → events (executing, executed, status, ...)

On utilise du polling /history pour la simplicité (suffit pour 1 server / queue séquentielle).
"""
from __future__ import annotations
import io
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", client_id: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())
        self.session = requests.Session()

    def submit(self, workflow: Dict[str, Any], timeout: float = 30.0) -> str:
        """POST /prompt → prompt_id."""
        body = {"prompt": workflow, "client_id": self.client_id}
        r = self.session.post(f"{self.base_url}/prompt", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()["prompt_id"]

    def wait(self, prompt_id: str, timeout: float = 600.0, poll_interval: float = 1.0) -> Dict[str, Any]:
        """Poll /history jusqu'à ce que le prompt soit terminé. Retourne le dict outputs."""
        start = time.time()
        while time.time() - start < timeout:
            r = self.session.get(f"{self.base_url}/history/{prompt_id}", timeout=10.0)
            if r.status_code == 200:
                payload = r.json()
                if prompt_id in payload:
                    entry = payload[prompt_id]
                    status = entry.get("status", {})
                    if status.get("completed", False):
                        return entry.get("outputs", {})
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"comfy error for {prompt_id}: {status}")
            time.sleep(poll_interval)
        raise TimeoutError(f"comfy timeout {timeout}s for {prompt_id}")

    def view(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        r = self.session.get(f"{self.base_url}/view", params=params, timeout=60.0)
        r.raise_for_status()
        return r.content

    def collect_images(self, outputs: Dict[str, Any]) -> List[Tuple[str, bytes]]:
        """Pour chaque output node qui a 'images', télécharge les bytes.

        Retourne une liste (filename, bytes) ordonnée par node_id puis par ordre dans 'images'.
        """
        out: List[Tuple[str, bytes]] = []
        for node_id in sorted(outputs.keys()):
            images = outputs[node_id].get("images", []) or outputs[node_id].get("gifs", [])
            for img in images:
                fn = img.get("filename")
                sub = img.get("subfolder", "")
                ftype = img.get("type", "output")
                data = self.view(fn, subfolder=sub, folder_type=ftype)
                out.append((fn, data))
        return out

    def alive(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/object_info", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False
