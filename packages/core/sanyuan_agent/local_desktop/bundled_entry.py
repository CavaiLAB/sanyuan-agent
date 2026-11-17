"""Minimal PyInstaller entry point for the desktop-run protocol."""

from sanyuan_agent.local_desktop.stdio_guard import run_guarded_desktop_process


def main() -> int:
    return run_guarded_desktop_process()


if __name__ == "__main__":
    raise SystemExit(main())
