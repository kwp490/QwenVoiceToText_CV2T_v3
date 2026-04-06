import unittest
from unittest.mock import MagicMock, patch

from cv2t.text_processor import TextProcessor, _build_system_prompt, _sanitize_error


class BuildSystemPromptTests(unittest.TestCase):
    """Tests for dynamic system-prompt construction."""

    def test_all_flags_on(self):
        prompt = _build_system_prompt(True, True, True)
        self.assertIn("professional", prompt.lower())
        self.assertIn("grammar", prompt.lower())
        self.assertIn("punctuation", prompt.lower())

    def test_tone_only(self):
        prompt = _build_system_prompt(True, False, False)
        self.assertIn("professional", prompt.lower())
        self.assertNotIn("grammar", prompt.lower())
        self.assertNotIn("punctuation", prompt.lower())

    def test_grammar_only(self):
        prompt = _build_system_prompt(False, True, False)
        self.assertIn("grammar", prompt.lower())
        self.assertNotIn("professional", prompt.lower())

    def test_punctuation_only(self):
        prompt = _build_system_prompt(False, False, True)
        self.assertIn("punctuation", prompt.lower())
        self.assertNotIn("grammar", prompt.lower())

    def test_all_flags_off_returns_empty(self):
        prompt = _build_system_prompt(False, False, False)
        self.assertEqual(prompt, "")

    def test_custom_prompt_replaces_default_tone(self):
        prompt = _build_system_prompt(
            True, False, False, custom_prompt="Use a legal tone."
        )
        self.assertIn("legal tone", prompt.lower())
        self.assertNotIn("professional and neutral", prompt.lower())

    def test_custom_prompt_included_when_tone_off(self):
        prompt = _build_system_prompt(
            False, True, False, custom_prompt="Always use Oxford comma."
        )
        self.assertIn("oxford comma", prompt.lower())
        self.assertIn("grammar", prompt.lower())

    def test_vocabulary_appended(self):
        prompt = _build_system_prompt(
            True, True, True, vocabulary="Kubernetes, gRPC, OAuth2"
        )
        self.assertIn("kubernetes", prompt.lower())
        self.assertIn("grpc", prompt.lower())
        self.assertIn("preserve these terms", prompt.lower())

    def test_empty_vocabulary_ignored(self):
        prompt = _build_system_prompt(True, True, True, vocabulary="")
        self.assertNotIn("preserve these terms", prompt.lower())

    def test_all_flags_off_with_custom_prompt_returns_empty(self):
        prompt = _build_system_prompt(False, False, False, custom_prompt="")
        self.assertEqual(prompt, "")


class SanitizeErrorTests(unittest.TestCase):
    """Tests for API key redaction in error messages."""

    def test_key_redacted(self):
        result = _sanitize_error(Exception("Error with sk-abc123"), "sk-abc123")
        self.assertNotIn("sk-abc123", result)
        self.assertIn("***", result)

    def test_empty_key(self):
        result = _sanitize_error(Exception("Some error"), "")
        self.assertEqual(result, "Some error")


class TextProcessorProcessTests(unittest.TestCase):
    """Tests for TextProcessor.process() with mocked OpenAI client."""

    def _make_processor(self):
        proc = TextProcessor(api_key="sk-test-key", model="gpt-5.4-mini")
        proc._client = MagicMock()
        return proc

    def test_empty_text_returns_empty(self):
        proc = self._make_processor()
        self.assertEqual(proc.process(""), "")
        self.assertEqual(proc.process("   "), "   ")

    def test_all_flags_off_returns_original(self):
        proc = self._make_processor()
        result = proc.process(
            "Hello world",
            fix_tone=False,
            fix_grammar=False,
            fix_punctuation=False,
        )
        self.assertEqual(result, "Hello world")
        proc._client.chat.completions.create.assert_not_called()

    def test_successful_cleanup(self):
        proc = self._make_processor()
        mock_choice = MagicMock()
        mock_choice.message.content = "I am having a difficult day at work."
        proc._client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        result = proc.process(
            "I am having a horrible day at work and I am angry",
            fix_tone=True,
            fix_grammar=True,
            fix_punctuation=True,
        )
        self.assertEqual(result, "I am having a difficult day at work.")
        proc._client.chat.completions.create.assert_called_once()

    def test_api_error_returns_original(self):
        proc = self._make_processor()
        proc._client.chat.completions.create.side_effect = Exception("API down")

        result = proc.process("angry text", fix_tone=True)
        self.assertEqual(result, "angry text")

    def test_none_response_returns_original(self):
        proc = self._make_processor()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        proc._client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        result = proc.process("test input", fix_tone=True)
        self.assertEqual(result, "test input")

    def test_no_api_key_returns_original(self):
        proc = TextProcessor(api_key="", model="gpt-5.4-mini")
        result = proc.process("angry text", fix_tone=True)
        self.assertEqual(result, "angry text")

    def test_process_with_preset(self):
        from cv2t.pro_preset import ProPreset

        proc = self._make_processor()
        mock_choice = MagicMock()
        mock_choice.message.content = "Cleaned text."
        proc._client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        preset = ProPreset(
            name="Test",
            system_prompt="Be formal.",
            fix_tone=True,
            fix_grammar=True,
            fix_punctuation=True,
            vocabulary="API, REST",
        )
        result = proc.process("messy text", preset=preset)
        self.assertEqual(result, "Cleaned text.")
        proc._client.chat.completions.create.assert_called_once()
        # Verify the system prompt includes custom prompt and vocabulary
        call_args = proc._client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        system_msg = messages[0]["content"]
        self.assertIn("formal", system_msg.lower())
        self.assertIn("api", system_msg.lower())


class TextProcessorValidateKeyTests(unittest.TestCase):
    """Tests for TextProcessor.validate_key()."""

    def test_valid_key(self):
        proc = TextProcessor(api_key="sk-test-key", model="gpt-5.4-mini")
        proc._client = MagicMock()
        proc._client.models.list.return_value = []

        ok, msg = proc.validate_key()
        self.assertTrue(ok)
        self.assertIn("valid", msg.lower())

    def test_invalid_key(self):
        from openai import AuthenticationError

        proc = TextProcessor(api_key="sk-bad-key", model="gpt-5.4-mini")
        proc._client = MagicMock()

        # AuthenticationError requires specific args
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        proc._client.models.list.side_effect = AuthenticationError(
            message="Invalid API key",
            response=mock_response,
            body=None,
        )

        ok, msg = proc.validate_key()
        self.assertFalse(ok)
        self.assertIn("invalid", msg.lower())

    def test_no_key(self):
        proc = TextProcessor(api_key="", model="gpt-5.4-mini")
        ok, msg = proc.validate_key()
        self.assertFalse(ok)
        self.assertIn("no api key", msg.lower())

    def test_api_key_never_in_error_message(self):
        proc = TextProcessor(api_key="sk-secret-123", model="gpt-5.4-mini")
        proc._client = MagicMock()
        proc._client.models.list.side_effect = Exception(
            "Connection failed for sk-secret-123"
        )

        ok, msg = proc.validate_key()
        self.assertFalse(ok)
        self.assertNotIn("sk-secret-123", msg)


if __name__ == "__main__":
    unittest.main()
