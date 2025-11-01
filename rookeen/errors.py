from __future__ import annotations

import json
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RookeenError:
    code: int
    name: str
    message: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "error": {
                    "code": self.code,
                    "name": self.name,
                    "message": self.message,
                }
            },
            ensure_ascii=False,
        )


def emit_and_exit(err: RookeenError) -> None:
    sys.stderr.write(err.to_json() + "\n")
    raise SystemExit(err.code)


GENERIC = RookeenError(1, "GENERIC_ERROR", "Generic error")
USAGE = RookeenError(2, "USAGE_ERROR", "Invalid CLI arguments")
FETCH = RookeenError(3, "FETCH_ERROR", "Failed to fetch content")
MODEL = RookeenError(4, "MODEL_ERROR", "Missing or failed spaCy model")
