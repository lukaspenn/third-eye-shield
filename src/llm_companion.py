"""
Third Eye Shield — LLM Health Companion Service
==========================================

Context-aware LLM assistant for elderly wellness monitoring.
Runs on a laptop/server and exposes an HTTP API that the RPi calls.

Recommended models (Singapore National LLM Programme):
  - SEA-LION: aisingapore/llama-3-8b-cpt-sea-lionv3-instruct
  - MERaLiON: SUTD/MERaLiON-AudioLLM-Whisper-SEA-LION (with audio)

Usage — Start the companion server on a laptop:
    python3 src/llm_companion.py --port 5000
    python3 src/llm_companion.py --port 5000 --model aisingapore/llama-3-8b-cpt-sea-lionv3-instruct

Then point the wellness monitor to it:
    python3 scripts/wellness_monitor.py --llm-endpoint http://<laptop-ip>:5000/chat

API:
    POST /chat
    Body: {"context": {...wellness context...}, "user_message": "optional text"}
    Response: {"response": "LLM text response"}
"""
import json
import os
import sys
import argparse
from pathlib import Path

# System prompt for elderly care companion
SYSTEM_PROMPT = """You are Third Eye Shield, a caring and empathetic health companion for elderly individuals living alone in Singapore. You monitor their wellness through AI sensors and provide friendly check-ins.

Your personality:
- Warm, patient, and respectful — treat every user like a respected elder
- Speak simply and clearly — avoid medical jargon
- Multilingual: respond in the user's preferred language (English, Mandarin, Malay, Tamil, or Singlish)
- Culturally aware of Singapore's diverse communities

Your capabilities:
- You receive real-time wellness data (activity, posture, emotion if opted-in, sedentary time)
- You can remind users about medication, hydration, exercises
- You detect falls and guide users through safety checks
- You encourage social engagement and physical activity

Important rules:
- NEVER diagnose medical conditions — always suggest seeing a doctor for concerns
- If a fall is detected, calmly ask if the user is okay and offer to alert emergency contacts
- Respect privacy — don't mention camera details, just say "I noticed" naturally
- If emotion detection is off, do not reference emotions or facial expressions
- Keep responses concise (2-3 sentences for check-ins, more for conversations)
- Use encouraging language: "Well done!", "You're doing great!", "That's wonderful!"
"""


def build_context_prompt(context: dict, user_message: str = None) -> str:
    """Build a prompt from wellness context and optional user message."""
    parts = []

    event = context.get("event", "periodic_checkin")
    wl = context.get("wellness_level", 1)
    wn = context.get("wellness_name", "Normal")
    action = context.get("action", "")
    posture = context.get("posture_score")
    emotion = context.get("emotion")
    sed_min = context.get("sedentary_minutes", 0)
    emotion_en = context.get("emotion_enabled", False)

    parts.append(f"[Wellness System Update -- {event}]")
    parts.append(f"Current status: {wn} (level {wl}/4)")

    if action and action != "(idle)":
        parts.append(f"Activity: {action}")

    if posture is not None:
        quality = "good" if posture >= 65 else "fair" if posture >= 35 else "poor"
        parts.append(f"Posture quality: {quality} ({posture:.0f}/100)")

    if emotion_en and emotion:
        parts.append(f"Detected emotion: {emotion}")
    elif not emotion_en:
        parts.append("(Emotion detection is off -- user has not opted in)")

    if sed_min >= 5:
        parts.append(f"Inactive for: {sed_min:.0f} minutes")

    # Event-specific prompts
    if event == "fall_alert":
        parts.append("\nURGENT: A fall has been detected. "
                     "Ask the user if they are okay. Be calm and reassuring.")
    elif event == "concern":
        parts.append("\nA wellness concern was flagged. "
                     "Gently check in with the user.")
    elif event == "periodic_checkin":
        parts.append("\nTime for a friendly check-in. "
                     "Ask how they're doing and offer encouragement.")
    elif event == "sedentary_alert":
        parts.append("\nThe user has been inactive for a while. "
                     "Gently suggest some light activity or stretching.")

    if user_message:
        parts.append(f"\nUser says: {user_message}")
        parts.append("Respond naturally to what they said, incorporating the wellness context.")
    else:
        parts.append("Generate a brief, caring check-in message.")

    return "\n".join(parts)


class LLMCompanion:
    """
    LLM companion with pluggable backend.
    Supports: GGUF (llama-cpp-python), transformers (HuggingFace), or a simple rule-based fallback.
    """

    def __init__(self, model_name=None, device="auto", gguf_path=None):
        self.model_name = model_name
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self._llama = None  # llama-cpp-python model

        if gguf_path:
            self._init_gguf(gguf_path)
        elif model_name:
            self._init_transformers(model_name, device)
        else:
            print("[LLM] No model specified -- using rule-based fallback")

    def _init_gguf(self, gguf_path):
        try:
            from llama_cpp import Llama
            print(f"[LLM] Loading GGUF model: {gguf_path}")
            self._llama = Llama(
                model_path=gguf_path,
                n_ctx=2048,
                n_threads=os.cpu_count() or 4,
                verbose=False,
            )
            self.model_name = os.path.basename(gguf_path)
            print(f"[LLM] GGUF model loaded successfully")
        except Exception as e:
            print(f"[LLM] Failed to load GGUF model: {e}")
            print("[LLM] Falling back to rule-based responses.")
            self._llama = None

    def _init_transformers(self, model_name, device):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
            import torch

            print(f"[LLM] Loading model: {model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device if device != "auto" or torch.cuda.is_available() else "cpu",
                trust_remote_code=True,
            )
            self._pipeline = hf_pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            print(f"[LLM] Model loaded on {next(self._model.parameters()).device}")
        except Exception as e:
            print(f"[LLM] Failed to load {model_name}: {e}")
            print("[LLM] Falling back to rule-based responses.")
            self._pipeline = None

    def generate(self, context: dict, user_message: str = None) -> str:
        """Generate a response given wellness context and optional user text."""
        context_prompt = build_context_prompt(context, user_message)

        if self._llama:
            return self._generate_gguf(context_prompt)
        elif self._pipeline:
            return self._generate_llm(context_prompt)
        else:
            return self._generate_fallback(context, user_message)

    def _generate_gguf(self, context_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_prompt},
        ]
        response = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
        )
        return response["choices"][0]["message"]["content"].strip()

    def _generate_llm(self, context_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_prompt},
        ]
        # Try chat template first (for instruction-tuned models)
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"{SYSTEM_PROMPT}\n\n{context_prompt}\n\nAssistant:"

        outputs = self._pipeline(prompt, return_full_text=False)
        return outputs[0]['generated_text'].strip()

    def _generate_fallback(self, context: dict, user_message: str = None) -> str:
        """Rule-based fallback when no LLM model is available."""
        event = context.get("event", "periodic_checkin")
        wn = context.get("wellness_name", "Normal")
        action = context.get("action", "")
        posture = context.get("posture_score")
        sed_min = context.get("sedentary_minutes", 0)
        emotion = context.get("emotion") if context.get("emotion_enabled") else None

        if event == "fall_alert":
            return ("I noticed you may have had a fall. Are you alright? "
                    "Take your time, and let me know if you need help. "
                    "I can alert your emergency contact if needed.")

        if event == "concern":
            parts = ["I just wanted to check in with you."]
            if posture is not None and posture < 35:
                parts.append("Try sitting up a bit straighter if you can - it's good for your back!")
            if emotion in ('sad', 'fear'):
                parts.append("You seem a little down. Would you like to chat, or shall I play some music?")
            return " ".join(parts)

        if sed_min >= 30:
            return (f"You've been sitting for about {sed_min:.0f} minutes. "
                    "How about a short walk or some gentle stretches? "
                    "Even a few minutes can make a big difference!")

        if action in ('arm circles', 'clapping', 'kicking something'):
            return f"Great job staying active with {action}! Keep it up, you're doing wonderfully!"

        if user_message:
            return ("Thank you for sharing! I'm here whenever you need someone to talk to. "
                    "Remember to drink some water and take things at your own pace.")

        # Generic check-in
        return ("Hello! Just checking in - how are you feeling today? "
                "Remember to stay hydrated and take breaks when you need them.")


def run_server(companion, host="0.0.0.0", port=5000):
    """Run Flask HTTP API server for the wellness monitor to call."""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("[ERROR] Flask not installed. Install with: pip install flask")
        print("[INFO] You can also use the companion in direct mode:")
        print("       from src.llm_companion import LLMCompanion")
        sys.exit(1)

    app = Flask(__name__)

    @app.route('/chat', methods=['POST'])
    def chat():
        data = request.get_json(force=True)
        context = data.get('context', {})
        user_msg = data.get('user_message', None)
        response = companion.generate(context, user_msg)
        return jsonify({"response": response})

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            "status": "ok",
            "model": companion.model_name or "rule-based-fallback",
            "product": "Third Eye Shield LLM Companion",
        })

    print(f"\n[Third Eye Shield LLM] Server starting on {host}:{port}")
    print(f"  POST /chat  -- send wellness context, get response")
    print(f"  GET  /health -- check server status\n")
    app.run(host=host, port=port, debug=False)


def main():
    parser = argparse.ArgumentParser(description="Third Eye Shield LLM Companion Server")
    parser.add_argument('--model', type=str, default=None,
                        help='HuggingFace model name (e.g. aisingapore/llama-3-8b-cpt-sea-lionv3-instruct)')
    parser.add_argument('--gguf', type=str, default=None,
                        help='Path to a GGUF model file for llama-cpp-python backend')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for model: auto, cpu, cuda')
    args = parser.parse_args()

    companion = LLMCompanion(model_name=args.model, device=args.device, gguf_path=args.gguf)
    run_server(companion, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
