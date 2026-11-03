"""CLI: python -m app.pipeline [init|score|reasons|all]"""
import sys
import time

from .db import connect, init_schema


def run(step: str = "all", job_type: str = "pipeline") -> dict:
    from .generator import generate
    from .models import train_and_score
    from .reasons import generate_reasons

    init_schema()
    with connect() as c:
        job = c.execute("INSERT INTO job_status (job_type, status) VALUES (%s,'running') RETURNING job_id",
                        (job_type,)).fetchone()["job_id"]
        c.commit()
    out, t0 = {}, time.time()
    try:
        if step in ("init", "all"):
            out["generate"] = generate()
        if step in ("init", "all", "score"):
            out["score"] = train_and_score()
        if step in ("all", "score", "reasons"):
            out["reasons"] = generate_reasons()
        out["seconds"] = round(time.time() - t0, 1)
        status, detail = "done", str(out)
    except Exception as e:  # noqa: BLE001
        status, detail = "failed", f"{type(e).__name__}: {e}"
        out["error"] = detail
    with connect() as c:
        c.execute("UPDATE job_status SET status=%s, detail=%s, finished_at=now() WHERE job_id=%s", (status, detail[:4000], job))
        c.commit()
    return out


if __name__ == "__main__":
    print(run(sys.argv[1] if len(sys.argv) > 1 else "all"))
