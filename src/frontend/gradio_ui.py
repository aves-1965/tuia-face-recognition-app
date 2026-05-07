from __future__ import annotations

import json
import os
from typing import Any

import cv2
import gradio as gr
import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def _client() -> httpx.Client:
    return httpx.Client(timeout=120.0)


def _abs_url(rel: str | None) -> str | None:
    if not rel:
        return None
    if rel.startswith("http://") or rel.startswith("https://"):
        return rel
    return f"{API_BASE}{rel}" if rel.startswith("/") else f"{API_BASE}/{rel}"


def upload_numpy_image(image: np.ndarray | None) -> str:
    if image is None:
        raise ValueError("Falta la imagen.")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen.")
    data = buf.tobytes()
    files = {"file": ("upload.jpg", data, "image/jpeg")}
    with _client() as c:
        r = c.post(f"{API_BASE}/upload", files=files)
        r.raise_for_status()
        body = r.json()
    return str(body["path"])


# InsightFace: k0 ojo izq, k1 ojo der, k2 nariz, k3 boca izq, k4 boca der (coords en espacio del recorte).
_KP_EDGES_5 = ((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4))

_KP_COLORS_BGR = (
    (40, 40, 220),
    (220, 40, 40),
    (60, 220, 60),
    (220, 220, 60),
    (220, 60, 220),
)


def _keypoints_crop_to_full(
    keypoints: dict[str, Any], x1: int, y1: int
) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for key, val in keypoints.items():
        if not isinstance(key, str) or not key.startswith("k"):
            continue
        try:
            idx = int(key[1:])
        except ValueError:
            continue
        if not isinstance(val, (list, tuple)) or len(val) < 2:
            continue
        px = int(round(float(val[0])))
        py = int(round(float(val[1])))
        out[idx] = (px, py)
    return out


def draw_boxes_on_bgr(image_bgr: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    for det in result.get("detections", []):
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        label = det.get("label", "?")
        score = det.get("score", 0.0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 220, 80), 2)
        txt = f"{label} ({score})"
        cv2.putText(
            vis,
            txt,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 220, 80),
            2,
        )
        raw_kp = det.get("keypoints") or {}
        if not isinstance(raw_kp, dict) or not raw_kp:
            continue
        pts = _keypoints_crop_to_full(raw_kp, x1, y1)
        # Solo conectamos la malla típica de 5 puntos (InsightFace); si hubiera más índices, solo puntos.
        m = max(pts.keys(), default=-1)
        edges = _KP_EDGES_5 if m <= 4 else ()
        line_color = (200, 230, 255)
        for a, b in edges:
            pa, pb = pts.get(a), pts.get(b)
            if pa is None or pb is None:
                continue
            cv2.line(vis, pa, pb, line_color, 1, cv2.LINE_AA)
        for i, (px, py) in sorted(pts.items()):
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))
            col = _KP_COLORS_BGR[i % len(_KP_COLORS_BGR)]
            cv2.circle(vis, (px, py), 4, col, -1, cv2.LINE_AA)
            cv2.circle(vis, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(
                vis,
                f"k{i}",
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def decode_image_bytes(content: bytes) -> np.ndarray | None:
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def start_predict(image: np.ndarray | None) -> tuple[str, str]:
    try:
        path = upload_numpy_image(image)
        with _client() as c:
            r = c.post(
                f"{API_BASE}/predict",
                json={"source_path": path, "source_type": "image"},
            )
            r.raise_for_status()
            job_id = r.json()["job_id"]
        msg = (
            f"Trabajo encolado. **job_id:** `{job_id}`\n\n"
            f"Enlaces del backend: `{API_BASE}`\n\n"
            "Pulsa **Consultar resultado de este job** o ve a **Estado y resultados**."
        )
        return job_id, msg
    except httpx.HTTPStatusError as exc:
        return "", f"Error HTTP: {exc.response.status_code} — {exc.response.text[:500]}"
    except Exception as exc:
        return "", f"Error: {exc}"


def start_register(identity: str, image: np.ndarray | None) -> tuple[str, str]:
    try:
        name = (identity or "").strip()
        if not name:
            return "", "Indica un nombre de identidad."
        path = upload_numpy_image(image)
        with _client() as c:
            r = c.post(
                f"{API_BASE}/insert",
                json={"identity": name, "image_path": path, "metadata": {"source": "frontend"}},
            )
            r.raise_for_status()
            job_id = r.json()["job_id"]
        msg = (
            f"Registro encolado. **job_id:** `{job_id}`\n\n"
            "Consulta el estado en **Estado y resultados**."
        )
        return job_id, msg
    except httpx.HTTPStatusError as exc:
        return "", f"Error HTTP: {exc.response.status_code} — {exc.response.text[:500]}"
    except Exception as exc:
        return "", f"Error: {exc}"


def consult_status(job_id: str) -> tuple[np.ndarray | None, str, str, str]:
    raw = (job_id or "").strip()
    if not raw:
        return None, "", "", "Ingresa un job_id."

    try:
        with _client() as c:
            r = c.get(f"{API_BASE}/status/{raw}")
            if r.status_code == 404:
                return None, "", "", "job_id no encontrado."
            r.raise_for_status()
            st = r.json()
    except httpx.HTTPError as exc:
        return None, "", "", f"No se pudo consultar el estado: {exc}"

    status = st.get("status")
    if status == "inProgress":
        return None, "", "", "**Estado:** en progreso…"

    if status == "failed":
        reason = st.get("reason") or "sin detalle"
        return None, "", "", f"**Estado:** fallido\n\n`{reason}`"

    artifact_url = st.get("artifact_url")
    source_image_url = st.get("source_image_url")
    link = st.get("link") or ""

    links_lines = []
    au = _abs_url(artifact_url)
    su = _abs_url(source_image_url)
    if au:
        links_lines.append(f"- [Descargar artefacto]({au})")
    if su:
        links_lines.append(f"- [Descargar imagen origen]({su})")
    links_md = "\n".join(links_lines) if links_lines else ""

    if not artifact_url and link not in ("", "none"):
        return (
            None,
            "",
            links_md,
            f"**Estado:** hecho, sin URL pública para: `{link}`",
        )

    if not artifact_url:
        return None, "", links_md, "**Estado:** hecho, pero sin enlace de descarga."

    try:
        with _client() as c:
            ar = c.get(_abs_url(artifact_url) or "")
            ar.raise_for_status()
            content = ar.content
            ctype = ar.headers.get("content-type", "")
    except httpx.HTTPError as exc:
        return None, "", links_md, f"Error al bajar artefacto: {exc}"

    looks_json = bool(artifact_url and str(artifact_url).lower().rstrip("/").endswith(".json"))
    if looks_json or "json" in ctype:
        try:
            data = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "", links_md, "No se pudo parsear el JSON de resultado."
        pretty = json.dumps(data, ensure_ascii=False, indent=2)
        people = ", ".join(data.get("detected_people") or []) or "(ninguno)"
        extra = f"**Personas detectadas:** {people}\n\n{links_md}"

        src_url = _abs_url(source_image_url)
        if not src_url:
            return None, pretty, extra, "**Estado:** completado (predicción); falta imagen origen."

        with _client() as c:
            ir = c.get(src_url)
            ir.raise_for_status()
        img_bgr = decode_image_bytes(ir.content)
        if img_bgr is None:
            return None, pretty, extra, "**Estado:** completado; no se decodificó la imagen origen."
        vis = draw_boxes_on_bgr(img_bgr, data)
        return vis, pretty, extra, "**Estado:** completado (predicción)."

    img_bgr = decode_image_bytes(content)
    if img_bgr is None:
        return None, "", links_md, "No se pudo decodificar la imagen de registro."
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    extra = f"Imagen de registro (recorte).\n\n{links_md}"
    return rgb, "", extra, "**Estado:** completado (registro)."


def build_ui() -> gr.Blocks:
    title = os.environ.get("APP_NAME", "Reconocimiento facial") + " — UI (externa)"
    with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "### Reconocimiento facial (cliente HTTP)\n"
            f"Backend configurado: `{API_BASE}` (`BACKEND_URL`). "
            "Sube imágenes, obtén **job_id** y consulta el estado. "
            "Las descargas usan los endpoints `/files/output/...` y `/files/data/...`. "
            "En predicción se dibujan **bounding boxes**, **keypoints** (k0…k4 en coordenadas del recorte, "
            "proyectados a la imagen completa) y aristas ojo–nariz–boca."
        )

        job_id_shared = gr.Textbox(label="Job ID (último o manual)", lines=1)

        with gr.Tab("Predecir"):
            gr.Markdown("Llama a `POST /upload` y `POST /predict`.")
            pred_in = gr.Image(label="Imagen", type="numpy", height=320)
            pred_btn = gr.Button("Iniciar predicción", variant="primary")
            pred_log = gr.Markdown()
            pred_quick = gr.Button("Consultar resultado de este job")

        with gr.Tab("Registrar identidad"):
            gr.Markdown("`POST /upload` y `POST /insert` (un rostro por imagen).")
            reg_name = gr.Textbox(label="Nombre / etiqueta", placeholder="ej. Ana")
            reg_in = gr.Image(label="Imagen", type="numpy", height=320)
            reg_btn = gr.Button("Registrar", variant="primary")
            reg_log = gr.Markdown()

        with gr.Tab("Estado y resultados"):
            gr.Markdown("`GET /status/{job_id}` — enlaces de descarga en el texto si aplica.")
            status_in = gr.Textbox(label="job_id a consultar", lines=1)
            status_btn = gr.Button("Consultar", variant="primary")
            status_line = gr.Markdown()
            vis_out = gr.Image(label="Vista: cajas, keypoints y malla facial", height=420)
            json_out = gr.Code(label="JSON (predicción)", language="json", lines=16)
            extra_md = gr.Markdown()

        def _on_pred(img: np.ndarray | None) -> tuple[str, str, str, None, str]:
            jid, msg = start_predict(img)
            return jid, msg, jid, None, ""

        def _on_reg(name: str, img: np.ndarray | None) -> tuple[str, str, str, None, str]:
            jid, msg = start_register(name, img)
            return jid, msg, jid, None, ""

        pred_btn.click(
            _on_pred,
            pred_in,
            [job_id_shared, pred_log, status_in, vis_out, json_out],
        )
        reg_btn.click(
            _on_reg,
            [reg_name, reg_in],
            [job_id_shared, reg_log, status_in, vis_out, json_out],
        )

        def _consult(from_id: str) -> tuple[Any, str, str, str]:
            vis, js, extra, line = consult_status(from_id)
            return vis, js, extra, line

        status_btn.click(
            _consult,
            status_in,
            [vis_out, json_out, extra_md, status_line],
        )
        pred_quick.click(
            _consult,
            job_id_shared,
            [vis_out, json_out, extra_md, status_line],
        )

    return demo
