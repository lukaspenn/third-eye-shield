"""
Third Eye Shield — Audio Interface (Future-Ready Abstraction)
======================================================

Provides speech-to-text (STT) and text-to-speech (TTS) capabilities
for voice interaction with the LLM companion.

Currently operates in text-fallback mode. When a microphone and speaker
are available, enable audio mode for natural voice conversations.

Supported backends:
  STT: Whisper (openai-whisper) or MERaLiON-AudioLLM
  TTS: pyttsx3 (offline) or gTTS (online, multilingual)
"""
import sys


class AudioInterface:
    """
    Unified audio I/O for Third Eye Shield.

    When no audio hardware is available, gracefully falls back to text mode.
    """

    def __init__(self, enable_stt=False, enable_tts=False,
                 stt_model="base", tts_engine="pyttsx3", language="en"):
        """
        Args:
            enable_stt: Enable speech-to-text input.
            enable_tts: Enable text-to-speech output.
            stt_model: Whisper model size ('tiny', 'base', 'small').
            tts_engine: 'pyttsx3' (offline) or 'gtts' (online).
            language: Language code (e.g. 'en', 'zh', 'ms', 'ta').
        """
        self.stt_enabled = False
        self.tts_enabled = False
        self.language = language

        if enable_stt:
            self._init_stt(stt_model)
        if enable_tts:
            self._init_tts(tts_engine)

    def _init_stt(self, model_size):
        try:
            import whisper
            self._whisper = whisper.load_model(model_size)
            self.stt_enabled = True
            print(f"[AUDIO] STT ready (Whisper {model_size})")
        except ImportError:
            print("[AUDIO] Whisper not installed -- STT disabled")
            print("        Install: pip install openai-whisper")
        except Exception as e:
            print(f"[AUDIO] STT init failed: {e}")

    def _init_tts(self, engine):
        if engine == "pyttsx3":
            try:
                import pyttsx3
                self._tts = pyttsx3.init()
                self._tts.setProperty('rate', 140)  # slower for elderly
                self._tts_engine = 'pyttsx3'
                self.tts_enabled = True
                print("[AUDIO] TTS ready (pyttsx3)")
            except Exception as e:
                print(f"[AUDIO] pyttsx3 TTS failed: {e}")
        elif engine == "gtts":
            try:
                from gtts import gTTS
                self._tts_engine = 'gtts'
                self.tts_enabled = True
                print("[AUDIO] TTS ready (gTTS -- requires internet)")
            except ImportError:
                print("[AUDIO] gTTS not installed -- TTS disabled")

    def listen(self, duration_sec=5, audio_file=None):
        """
        Capture audio and transcribe to text.

        Args:
            duration_sec: Recording duration if using microphone.
            audio_file: Path to audio file (alternative to live mic).

        Returns:
            str: Transcribed text, or None if STT unavailable.
        """
        if not self.stt_enabled:
            return None

        if audio_file:
            result = self._whisper.transcribe(str(audio_file), language=self.language)
            return result.get("text", "").strip()

        # Live microphone recording
        try:
            import sounddevice as sd
            import numpy as np
            import tempfile, soundfile as sf

            print(f"[AUDIO] Listening for {duration_sec}s...")
            audio = sd.rec(int(duration_sec * 16000), samplerate=16000,
                          channels=1, dtype='float32')
            sd.wait()
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, audio, 16000)
                result = self._whisper.transcribe(f.name, language=self.language)
                import os
                os.unlink(f.name)
            return result.get("text", "").strip()
        except ImportError:
            print("[AUDIO] sounddevice/soundfile not installed for live recording")
            return None
        except Exception as e:
            print(f"[AUDIO] Recording error: {e}")
            return None

    def speak(self, text):
        """
        Convert text to speech and play it.

        Args:
            text: Text to speak.
        """
        if not self.tts_enabled or not text:
            return

        if self._tts_engine == 'pyttsx3':
            try:
                self._tts.say(text)
                self._tts.runAndWait()
            except Exception as e:
                print(f"[AUDIO] TTS error: {e}")
        elif self._tts_engine == 'gtts':
            try:
                from gtts import gTTS
                import tempfile, os
                tts = gTTS(text=text, lang=self.language[:2])
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                    tts.save(f.name)
                    os.system(f"mpg321 -q {f.name} 2>/dev/null || "
                              f"mpg123 -q {f.name} 2>/dev/null || "
                              f"aplay {f.name} 2>/dev/null")
                    os.unlink(f.name)
            except Exception as e:
                print(f"[AUDIO] gTTS error: {e}")

    def get_user_input(self, prompt="You: ", prefer_voice=True):
        """
        Get input from user — voice if available, text fallback.

        Args:
            prompt: Text prompt shown if falling back to text input.
            prefer_voice: Try voice input first.

        Returns:
            str: User's input text.
        """
        if prefer_voice and self.stt_enabled:
            text = self.listen()
            if text:
                print(f"  [Heard] {text}")
                return text
            print("  [AUDIO] Couldn't hear clearly, switching to text input")

        try:
            return input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    def deliver_response(self, text, also_print=True):
        """
        Deliver LLM response — speak if TTS available, always show text.

        Args:
            text: Response text.
            also_print: Print to terminal regardless.
        """
        if also_print:
            print(f"  [Third Eye Shield] {text}")
        if self.tts_enabled:
            self.speak(text)
