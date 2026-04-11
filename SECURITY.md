# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 3.0.x   | Yes       |
| < 3.0   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in CV2T, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, email the maintainer directly or use [GitHub's private vulnerability reporting](https://github.com/kwp490/QwenVoiceToText_CV2T_v3/security/advisories/new).

### What to include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response timeline

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 1 week
- **Fix or mitigation**: depends on severity, typically within 2 weeks for critical issues

## Known Security Considerations

- **Keyboard hooks**: The `keyboard` library uses low-level hooks (`SetWindowsHookEx`) for global hotkeys. Antivirus or anti-malware software may flag this as suspicious — it is a false positive. CV2T only listens for the specific hotkey combinations you configure.
- **Administrator privileges**: The installer requires elevation to write to `C:\Program Files\CV2T`.
- **Defender exclusions**: The GUI installer automatically adds Windows Defender exclusions for the install directory and `cv2t.exe` to prevent false positives.
- **`uv.exe` false positives**: Some anti-malware tools (e.g. Malwarebytes) may quarantine `uv.exe` during source installs. If this happens, restore it and add it to your allow list. [uv](https://github.com/astral-sh/uv) is a widely used open-source Python package manager.
- **API key handling (Professional Mode)**: OpenAI API keys entered in Settings are held in memory only by default and are **never** written to `settings.json` or any log file. If "Remember API key" is enabled, the key is stored via Windows Credential Manager (protected by Windows DPAPI encryption). API keys are never displayed in the UI log panel, and all error messages are sanitized to redact key content.
- **Single-instance mutex**: A Windows named mutex (`Global\CV2TMutex`) prevents multiple CV2T processes from running simultaneously, avoiding resource conflicts.
- **Canary Bridge IPC**: The subprocess bridge (`CanaryBridgeEngine`) communicates with the Canary worker via JSON lines over stdin/stdout — local only, no network exposure.
- **Blackwell workarounds**: On RTX 50-series GPUs, environment variables `CUDA_LAUNCH_BLOCKING` and `TORCHDYNAMO_DISABLE` are set automatically for stability. These are security-neutral but modify the process environment.

## Privacy & Data Handling

**Audio**: Recorded audio is processed in memory and discarded after transcription. The Canary engine writes temporary WAV files to `%TEMP%\cv2t\` during processing; these are deleted immediately after use.

**Transcriptions**: Transcribed text is displayed in the UI and optionally copied to the clipboard. Transcription content is **not** written to log files — only character counts are logged. When **Professional Mode** is enabled, transcribed text is sent to the OpenAI API for cleanup (see Network below).

**Logs**: Application logs are stored at `C:\Program Files\CV2T\logs\` as rotating plaintext files (~6 MB max). Logs contain diagnostic information (engine status, GPU metrics, error traces) but no speech content. Logs are cleared on exit by default (`clear_logs_on_exit: true`).

**Network**: CV2T makes network requests **only** in two scenarios:

1. **Model downloads** — to HuggingFace Hub (public repos, no authentication required) when downloading speech engine models.
2. **Professional Mode** (when enabled) — transcribed text is sent to the OpenAI API (`api.openai.com`) for tone, grammar, and punctuation cleanup. This requires a user-provided API key and is **opt-in only** — disabled by default. No audio data is sent; only the transcribed text string is transmitted.

No telemetry, analytics, or usage data is collected or transmitted.

**Keyboard hooks**: Only the configured hotkey combinations are monitored — general keystrokes are not captured or logged.
