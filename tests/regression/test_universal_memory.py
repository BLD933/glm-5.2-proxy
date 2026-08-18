"""test_universal_memory.py — Verify Universal Dual-Mode Memory for delta harnesses and stateless clients."""

import sys
import unittest
from unittest.mock import MagicMock, patch
from server import (
    ChatCompletionRequest, Message, _resolve_session_id, _commit_turn,
    _tool_state, SESSIONS, chat_completions
)


class TestUniversalMemory(unittest.TestCase):
    def setUp(self):
        SESSIONS.clear()

    def test_session_resolution_order(self):
        # 1. Body session_id
        req1 = ChatCompletionRequest(messages=[Message(role="user", content="hi")], session_id="sess_123")
        mock_req1 = MagicMock()
        self.assertEqual(_resolve_session_id(req1, mock_req1), "sess_123")

        # 2. Header x-session-id
        req2 = ChatCompletionRequest(messages=[Message(role="user", content="hi")])
        mock_req2 = MagicMock()
        mock_req2.headers.get.side_effect = lambda k: "hdr_456" if k == "x-session-id" else None
        self.assertEqual(_resolve_session_id(req2, mock_req2), "hdr_456")

        # 3. Header x-conversation-id
        mock_req3 = MagicMock()
        mock_req3.headers.get.side_effect = lambda k: "dcp_conv_789" if k == "x-conversation-id" else None
        self.assertEqual(_resolve_session_id(req2, mock_req3), "dcp_conv_789")

        # 4. Fallback client host
        mock_req4 = MagicMock()
        mock_req4.headers.get.return_value = None
        mock_req4.client.host = "127.0.0.1"
        sid = _resolve_session_id(req2, mock_req4)
        self.assertTrue(sid.startswith("v1_sess_") and sid != "v1_sess_127.0.0.1", sid)

    def test_commit_turn_and_retrieval(self):
        sess_id = "test_sess_abc"
        _commit_turn(sess_id, "who are you?", "I'm GLM-5.2 developed by Z.ai")

        self.assertIn(sess_id, SESSIONS)
        self.assertEqual(len(SESSIONS[sess_id]["history"]), 2)
        self.assertEqual(SESSIONS[sess_id]["history"][0], {"role": "user", "content": "who are you?"})
        self.assertEqual(SESSIONS[sess_id]["history"][1], {"role": "assistant", "content": "I'm GLM-5.2 developed by Z.ai"})

    def test_tool_state_delta_hydration(self):
        sess_id = "test_tool_delta"
        _commit_turn(sess_id, "who are you?", "I'm GLM-5.2")

        # Delta harness sends single message:
        req = ChatCompletionRequest(
            messages=[Message(role="user", content="what did i say in the previous message ?")],
            session_id=sess_id
        )
        state = _tool_state(req, token="fake_token", session_id=sess_id)
        # History must contain the prior turn!
        self.assertEqual(len(state["history"]), 2)
        self.assertEqual(state["history"][0]["content"], "who are you?")
        self.assertEqual(state["history"][1]["content"], "I'm GLM-5.2")

    def test_tool_state_stateless_explicit(self):
        sess_id = "test_stateless"
        _commit_turn(sess_id, "stale turn", "stale answer")

        # Stateless client sends explicit multi-turn array:
        req = ChatCompletionRequest(
            messages=[
                Message(role="user", content="explicit turn 1"),
                Message(role="assistant", content="explicit answer 1"),
                Message(role="user", content="explicit turn 2"),
            ],
            session_id=sess_id
        )
        state = _tool_state(req, token="fake_token", session_id=sess_id)
        # Explicit messages override cache
        self.assertEqual(len(state["history"]), 2)
        self.assertEqual(state["history"][0]["content"], "explicit turn 1")
        self.assertEqual(state["history"][1]["content"], "explicit answer 1")


if __name__ == "__main__":
    unittest.main()
