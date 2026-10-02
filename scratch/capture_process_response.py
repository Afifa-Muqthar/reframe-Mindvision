import os
import sys
import json
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

WORKPLACE_INJUSTICE_INPUT = (
    "My supervisor gave credit for my three-month project to another colleague in the team meeting. "
    "I'm furious and feeling completely powerless."
)

app.config["TESTING"] = True

with patch("app.main.threading.Thread") as mock_thread, \
     patch("app.main._run_tts", return_value="data/sessions/session_test.wav"):
    with app.test_client() as client:
        res = client.post("/process", data={"text": WORKPLACE_INJUSTICE_INPUT})
        data = res.get_json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
