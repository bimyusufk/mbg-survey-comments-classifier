"""Jalankan MLflow + JupyterLab lalu buka keduanya ke tim lewat Cloudflare Quick Tunnel.

Pemakaian (dari root proyek, venv aktif):
    python scripts/serve.py              # MLflow + JupyterLab + 2 tunnel
    python scripts/serve.py --no-jupyter # MLflow saja
    python scripts/serve.py --local      # tanpa tunnel (cek lokal)

Keamanan:
- Link *.trycloudflare.com acak dan berganti setiap kali skrip dijalankan ulang.
- MLflow dilindungi login (basic-auth), JupyterLab dilindungi token.
- Link + kredensial hanya ditulis ke .share/links.txt (di-.gitignore) dan ke terminal.
  Bagikan HANYA lewat grup privat tim, jangan di README/issue/PR.
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
MLFLOW_DIR = ROOT / ".mlflow"
SHARE_DIR = ROOT / ".share"
LOG_DIR = SHARE_DIR / "logs"
TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

MLFLOW_PORT = 5000
JUPYTER_PORT = 8888


# ---------------------------------------------------------------- .env
def read_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def ensure_secrets() -> dict[str, str]:
    """Buat kredensial acak sekali saja, simpan di .env (tidak di-commit)."""
    env = read_env()
    defaults = {
        "SERVE_JUPYTER_TOKEN": secrets.token_urlsafe(24),
        "MLFLOW_FLASK_SERVER_SECRET_KEY": secrets.token_urlsafe(32),
        "MLFLOW_TRACKING_URI": f"http://127.0.0.1:{MLFLOW_PORT}",
        "MLFLOW_TRACKING_USERNAME": "tim",
        "MLFLOW_TRACKING_PASSWORD": secrets.token_urlsafe(18),
    }
    missing = {k: v for k, v in defaults.items() if not env.get(k)}
    if missing:
        with ENV_FILE.open("a", encoding="utf-8") as f:
            f.write(f"\n# Dibuat otomatis oleh scripts/serve.py ({datetime.now():%Y-%m-%d %H:%M})\n")
            for k, v in missing.items():
                f.write(f"{k}={v}\n")
        env.update(missing)
    return env


# ---------------------------------------------------------------- proses
def find_cloudflared() -> str:
    found = shutil.which("cloudflared")
    if found:
        return found
    for p in (
        r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
        r"C:\Program Files\cloudflared\cloudflared.exe",
    ):
        if Path(p).exists():
            return p
    sys.exit(
        "cloudflared tidak ditemukan. Pasang dulu:\n"
        "  Windows : winget install --id Cloudflare.cloudflared\n"
        "  macOS   : brew install cloudflared\n"
        "  Linux   : https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    )


def spawn(name: str, cmd: list[str], env: dict[str, str] | None = None) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = (LOG_DIR / f"{name}.log").open("w", encoding="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    return subprocess.Popen(
        cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, **(env or {})}, creationflags=flags,
    )


def wait_port(port: int, proc: subprocess.Popen, name: str, timeout: float = 90) -> None:
    import socket

    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            sys.exit(f"{name} berhenti saat start. Lihat .share/logs/{name}.log")
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    sys.exit(f"{name} tidak merespons di port {port}. Lihat .share/logs/{name}.log")


def start_tunnel(cloudflared: str, name: str, port: int) -> tuple[subprocess.Popen, str]:
    log_path = LOG_DIR / f"tunnel_{name}.log"
    proc = spawn(f"tunnel_{name}", [
        cloudflared, "tunnel", "--no-autoupdate",
        "--url", f"http://127.0.0.1:{port}",
    ])
    end = time.time() + 60
    while time.time() < end:
        if proc.poll() is not None:
            break
        m = TUNNEL_RE.search(log_path.read_text(encoding="utf-8", errors="ignore"))
        if m:
            return proc, m.group(0)
        time.sleep(0.5)
    sys.exit(f"Gagal membuat tunnel {name}. Lihat {log_path}")


def stop_all(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                               capture_output=True)
            else:
                p.terminate()


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-jupyter", action="store_true", help="hanya jalankan MLflow")
    ap.add_argument("--local", action="store_true", help="tanpa Cloudflare tunnel")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, errors="replace")

    env = ensure_secrets()
    user, password = env["MLFLOW_TRACKING_USERNAME"], env["MLFLOW_TRACKING_PASSWORD"]
    token = env["SERVE_JUPYTER_TOKEN"]

    # --- konfigurasi auth MLflow (admin dibuat saat server pertama kali jalan)
    MLFLOW_DIR.mkdir(exist_ok=True)
    auth_ini = MLFLOW_DIR / "basic_auth.ini"
    auth_ini.write_text(
        "[mlflow]\n"
        "default_permission = EDIT\n"
        f"database_uri = sqlite:///{(MLFLOW_DIR / 'auth.db').as_posix()}\n"
        f"admin_username = {user}\n"
        f"admin_password = {password}\n"
        "authorization_function = mlflow.server.auth:authenticate_request_basic_auth\n",
        encoding="utf-8",
    )

    procs: list[subprocess.Popen] = []

    def on_exit(*_):
        print("\nMenghentikan semua proses...")
        stop_all(procs)
        sys.exit(0)

    signal.signal(signal.SIGINT, on_exit)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, on_exit)

    try:
        print("Menjalankan MLflow...")
        mlflow = spawn("mlflow", [
            sys.executable, "-m", "mlflow", "server",
            "--app-name", "basic-auth",
            "--host", "127.0.0.1", "--port", str(MLFLOW_PORT),
            "--backend-store-uri", f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}",
            "--artifacts-destination", (MLFLOW_DIR / "artifacts").as_uri(),
            "--allowed-hosts", f"*.trycloudflare.com,localhost:{MLFLOW_PORT},127.0.0.1:{MLFLOW_PORT}",
            "--cors-allowed-origins",
            f"https://*.trycloudflare.com,http://localhost:{MLFLOW_PORT},http://127.0.0.1:{MLFLOW_PORT}",
        ], env={
            "MLFLOW_AUTH_CONFIG_PATH": str(auth_ini),
            "MLFLOW_FLASK_SERVER_SECRET_KEY": env["MLFLOW_FLASK_SERVER_SECRET_KEY"],
        })
        procs.append(mlflow)
        wait_port(MLFLOW_PORT, mlflow, "mlflow")

        if not args.no_jupyter:
            print("Menjalankan JupyterLab...")
            jupyter = spawn("jupyter", [
                sys.executable, "-m", "jupyter", "lab", "--no-browser",
                "--ServerApp.ip=127.0.0.1", f"--ServerApp.port={JUPYTER_PORT}",
                "--ServerApp.port_retries=0",
                f"--IdentityProvider.token={token}",
                # Host header dari tunnel bukan localhost; akses tetap dijaga token.
                "--ServerApp.allow_remote_access=True",
                f"--ServerApp.root_dir={ROOT}",
            ])
            procs.append(jupyter)
            wait_port(JUPYTER_PORT, jupyter, "jupyter")

        links = {"mlflow": f"http://127.0.0.1:{MLFLOW_PORT}"}
        if not args.no_jupyter:
            links["jupyter"] = f"http://127.0.0.1:{JUPYTER_PORT}/lab"

        if not args.local:
            cf = find_cloudflared()
            print("Membuat Cloudflare Quick Tunnel...")
            p, url = start_tunnel(cf, "mlflow", MLFLOW_PORT)
            procs.append(p)
            links["mlflow"] = url
            if not args.no_jupyter:
                p, url = start_tunnel(cf, "jupyter", JUPYTER_PORT)
                procs.append(p)
                links["jupyter"] = f"{url}/lab"

        lines = [
            f"Dibuat: {datetime.now():%Y-%m-%d %H:%M}  (link berganti tiap restart)",
            "",
            f"MLflow (dashboard eksperimen) : {links['mlflow']}",
            f"  username : {user}",
            f"  password : {password}",
        ]
        if "jupyter" in links:
            lines += [
                "",
                f"JupyterLab (notebook bersama) : {links['jupyter']}",
                f"  token    : {token}",
            ]
        lines += [
            "",
            "Untuk anggota yang training di laptop sendiri, isi .env mereka:",
            f"  MLFLOW_TRACKING_URI={links['mlflow']}",
            f"  MLFLOW_TRACKING_USERNAME={user}",
            f"  MLFLOW_TRACKING_PASSWORD={password}",
        ]
        text = "\n".join(lines)
        (SHARE_DIR / "links.txt").write_text(text + "\n", encoding="utf-8")

        bar = "=" * 64
        print(f"\n{bar}\n{text}\n{bar}")
        print("Bagikan HANYA lewat grup privat tim. Tersimpan juga di .share/links.txt")
        print("Tekan Ctrl+C untuk menghentikan semuanya.\n")

        while True:
            for p in procs:
                if p.poll() is not None:
                    print(f"Proses berhenti (exit {p.returncode}): {p.args[:4]}. Cek .share/logs/")
                    on_exit()
            time.sleep(2)
    finally:
        stop_all(procs)


if __name__ == "__main__":
    main()
