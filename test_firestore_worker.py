import asyncio
import os
import random
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import json
import os

os.environ.setdefault(
    "FIREBASE_KEYS",
    json.dumps({"tec-chat%d" % i: "test-key-%d" % i for i in range(1, 6)}),
)
import firestore_worker as worker
from firestore_http import error_detail, fetch, retry_after

P = worker.MESSAGE_SERVERS[0]


def src(project=P, room="general", mid="m1"):
    return "projects/%s/databases/(default)/documents/rooms/%s/messages/%s" % (project, room, mid)


class SerializationTests(unittest.TestCase):
    def test_round_trip(self):
        for value in [None, True, 0, 42, "hi", [], ["a", 1], {}, {"a": {"b": ["c"]}}]:
            with self.subTest(value=value):
                self.assertEqual(worker.unpack(worker.pack(value)), value)

    def test_unsupported_rejected(self):
        with self.assertRaises(RuntimeError):
            worker.unpack({"unknownValue": "no"})


class OutcomeTests(unittest.TestCase):
    def test_reply_and_silence(self):
        self.assertEqual(worker.outcome("[FINAL RESPONSE]Hello.[END FINAL RESPONSE]"), "Hello.")
        self.assertIsNone(worker.outcome("[NO RESPONSE][END NO RESPONSE]"))

    def test_invalid_rejected(self):
        for text in [
            "Hello.",
            "[FINAL RESPONSE][END FINAL RESPONSE]",
            "extra [FINAL RESPONSE]Hi[END FINAL RESPONSE]",
            "[FINAL RESPONSE]/ban x[END FINAL RESPONSE]",
            "[FINAL RESPONSE]@help[END FINAL RESPONSE]",
            "[FINAL RESPONSE]" + "x" * 1801 + "[END FINAL RESPONSE]",
            "[FINAL RESPONSE]A[END FINAL RESPONSE][FINAL RESPONSE]B[END FINAL RESPONSE]",
        ]:
            with self.subTest(text=text[:40]):
                with self.assertRaises(RuntimeError):
                    worker.outcome(text)

    def test_secret_overlap_rejected(self):
        secret = "example-secret-value-for-test-only"
        with patch.dict(os.environ, {"GROQ_API_KEY": secret}):
            with self.assertRaises(RuntimeError):
                worker.outcome("[FINAL RESPONSE]" + secret + "[END FINAL RESPONSE]")


class HttpTests(unittest.IsolatedAsyncioTestCase):
    def test_error_redaction(self):
        secret = "example-private-value-for-test"
        r = httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT",
            "message": secret + " https://example.com/private user@example.com"}})
        d = error_detail(r, secrets=(secret,))
        self.assertIn("HTTP 400", d)
        self.assertIn("INVALID_ARGUMENT", d)
        for bad in (secret, "https://example.com/private", "user@example.com"):
            self.assertNotIn(bad, d)

    def test_index_link_kept_only_on_real_host(self):
        path = "/v1/r/project/test/firestore/indexes?create_exemption=T"
        good = "https://console.firebase.google.com" + path
        bad = "https://console.firebase.google.com.example.com" + path
        for url, kept in ((good, True), (bad, False)):
            r = httpx.Response(400, json={"error": {"status": "FAILED_PRECONDITION", "message": url}})
            self.assertEqual(url in error_detail(r), kept)

    def test_retry_after(self):
        self.assertEqual(retry_after(httpx.Response(429, headers={"Retry-After": "12"})), 12)
        self.assertEqual(retry_after(httpx.Response(429, headers={"Retry-After": "x"})), 0)

    async def test_quota_and_writes_not_retried(self):
        for method, status in (("GET", 429), ("POST", 503)):
            client = AsyncMock()
            response = httpx.Response(status)
            client.request.return_value = response
            self.assertIs(await fetch(client, method, "https://firestore.googleapis.com/v1/x"), response)
            client.request.assert_awaited_once()


class SelectionTests(unittest.TestCase):
    def test_mentions(self):
        self.assertEqual(worker.mentioned("hey @Mira and dylan!"), ["mira", "dylan"])
        self.assertEqual(worker.mentioned("Mirage and Dylanesque"), [])
        self.assertEqual(worker.mentioned("Miko's hoodie"), ["miko"])

    def test_own_room_answers_humans_only(self):
        human = {"role": "member", "uid": "u1", "content": "hi"}
        ai = {"role": "ai", "uid": "ai-mira", "content": "hi"}
        self.assertEqual(worker.choose_responders("miko", human), ["miko"])
        self.assertEqual(worker.choose_responders("miko", ai), [])

    def test_general_named_always_and_never_self(self):
        msg = {"role": "ai", "uid": "ai-mira", "content": "Mira says hi to Dylan and Mira"}
        for seed in range(20):
            chosen = worker.choose_responders("general", msg, random.Random(seed))
            self.assertIn("dylan", chosen)
            self.assertNotIn("mira", chosen)

    def test_general_limit(self):
        msg = {"role": "member", "uid": "u", "content": "hello everyone"}
        with patch.object(worker, "AMBIENT_HUMAN", 1.0):
            chosen = worker.choose_responders("general", msg, random.Random(1))
        self.assertEqual(len(chosen), worker.MAX_RESPONDERS)

    def test_unknown_room_ignored(self):
        self.assertEqual(worker.choose_responders("other", {"content": "Mira"}), [])

    def test_chain_count(self):
        ctx = [{"role": "member"}, {"role": "ai"}, {"role": "ai"}]
        self.assertEqual(worker.trailing_ai(ctx), 2)


class SourceTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(worker.parse_source(P, src()), ("general", "m1"))

    def test_rejected(self):
        for bad in (src("other-project"), src(room="nope"),
                    "projects/%s/databases/(default)/documents/users/x" % P):
            with self.subTest(bad=bad):
                with self.assertRaises(RuntimeError):
                    worker.parse_source(P, bad)


class SendTests(unittest.IsolatedAsyncioTestCase):
    def make(self):
        w = worker.Worker(None)
        a = w.auth[P]
        a.token, a.uid, a.refresh = "tok", "anon-uid", "ref"
        a.expires = 1e18
        w.request = AsyncMock(return_value=httpx.Response(200))
        return w

    async def test_existing_reply_prevents_write(self):
        w = self.make()
        with patch.object(worker, "fetch", new_callable=AsyncMock, return_value=httpx.Response(200)):
            self.assertFalse(await w.send(P, src(), {}, "mira", "Hi"))
        w.request.assert_not_awaited()

    async def test_failed_dedup_prevents_write(self):
        w = self.make()
        with patch.object(worker, "fetch", new_callable=AsyncMock, return_value=httpx.Response(403)):
            with self.assertRaisesRegex(RuntimeError, "deduplication"):
                await w.send(P, src(), {}, "mira", "Hi")
        w.request.assert_not_awaited()

    async def test_cross_project_and_unknown_character(self):
        w = self.make()
        with self.assertRaisesRegex(RuntimeError, "project scope"):
            await w.send(P, src("other"), {}, "mira", "Hi")
        with self.assertRaisesRegex(RuntimeError, "Unknown character"):
            await w.send(P, src(), {}, "nobody", "Hi")

    async def test_commit_shape(self):
        w = self.make()
        with patch.object(worker, "fetch", new_callable=AsyncMock, return_value=httpx.Response(404)):
            self.assertTrue(await w.send(P, src(), {"name": "Pat", "content": "hello there"}, "dylan", "Yo!"))
        args, kwargs = w.request.await_args
        self.assertEqual(args[1], worker.base(P) + ":commit")
        write = kwargs["json"]["writes"][0]
        self.assertEqual(write["currentDocument"], {"exists": False})
        self.assertEqual(write["updateTransforms"][0]["fieldPath"], "createdAt")
        m = worker.fields(write["update"])
        self.assertEqual((m["name"], m["role"], m["uid"], m["authUid"]), ("Dylan", "ai", "ai-dylan", "anon-uid"))
        self.assertEqual(m["replyTo"]["name"], "Pat")
        self.assertLessEqual(len(m["name"]), 20)

    async def test_unknown_server_rejected(self):
        w = self.make()
        with self.assertRaisesRegex(RuntimeError, "Server not allowed"):
            await w.new_messages("unrelated-project", "general", "2020-01-01T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
