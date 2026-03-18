"""
Third Eye Shield -- Telegram Alert Notifier
=====================================

Sends wellness alerts to family members / caregivers via Telegram Bot API.
Uses only the standard library (urllib) -- no extra dependencies.

Setup:
  1. Create a bot via @BotFather on Telegram -> get BOT_TOKEN
  2. Get your chat ID: message the bot, then visit
     https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Set in config.yaml:
       wellness.telegram.bot_token: "123456:ABC..."
       wellness.telegram.chat_ids: ["987654321"]

Usage from wellness_monitor.py:
    from src.telegram_notifier import TelegramNotifier
    notifier = TelegramNotifier(bot_token, chat_ids)
    notifier.send_alert("Fall detected!", level="alert")
"""
import json
import time
import urllib.request
import urllib.error
import urllib.parse
import threading
from datetime import datetime


class TelegramNotifier:
    """
    Send Third Eye Shield wellness alerts to Telegram.
    Non-blocking: messages are sent in a background thread.
    """

    API_BASE = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, bot_token, chat_ids, cooldown_sec=60, enabled=True):
        """
        Args:
            bot_token: Telegram Bot API token from @BotFather.
            chat_ids: List of chat ID strings to notify.
            cooldown_sec: Minimum seconds between alerts of the same type.
            enabled: Master toggle.
        """
        self.bot_token = bot_token
        self.chat_ids = list(chat_ids) if chat_ids else []
        self.cooldown_sec = cooldown_sec
        self.enabled = enabled and bool(bot_token) and len(self.chat_ids) > 0
        self._last_alert_time = {}  # type -> timestamp
        self._send_queue = []

        if self.enabled:
            print(f"[TELEGRAM] Ready -- {len(self.chat_ids)} recipient(s)")
        else:
            if not bot_token:
                print("[TELEGRAM] Disabled (no bot_token configured)")
            elif not self.chat_ids:
                print("[TELEGRAM] Disabled (no chat_ids configured)")

    def send_alert(self, message, level="info", context=None):
        """
        Send an alert message to all configured chat IDs.
        Non-blocking -- runs in a daemon thread.

        Args:
            message: Alert text.
            level: 'info', 'concern', 'alert' -- controls emoji prefix.
            context: Optional dict of wellness context for rich formatting.
        """
        if not self.enabled:
            return

        # Cooldown check
        now = time.time()
        last = self._last_alert_time.get(level, 0)
        if now - last < self.cooldown_sec:
            return
        self._last_alert_time[level] = now

        text = self._format_message(message, level, context)
        thread = threading.Thread(
            target=self._send_to_all,
            args=(text,),
            daemon=True,
        )
        thread.start()

    def send_daily_summary(self, summary_dict):
        """Send a formatted daily wellness summary."""
        if not self.enabled:
            return

        lines = [
            "📊 *Third Eye Shield Daily Summary*",
            f"📅 {datetime.now().strftime('%A, %d %B %Y')}",
            "",
        ]

        falls = summary_dict.get('fall_alerts', 0)
        concerns = summary_dict.get('concerns', 0)
        posture_avg = summary_dict.get('posture_avg')
        sed_max = summary_dict.get('sedentary_max', 0)

        if falls > 0:
            lines.append(f"🚨 Falls detected: {falls}")
        else:
            lines.append("✅ No falls detected")

        if concerns > 0:
            lines.append(f"⚠️ Concerns flagged: {concerns}")

        if posture_avg is not None:
            emoji = "✅" if posture_avg >= 65 else "⚠️" if posture_avg >= 35 else "🔴"
            lines.append(f"{emoji} Avg posture: {posture_avg:.0f}/100")

        if sed_max >= 30:
            lines.append(f"🪑 Longest inactive: {sed_max:.0f} min")

        actions = summary_dict.get('actions', {})
        if actions:
            top = list(actions.items())[:3] if isinstance(actions, dict) else []
            if top:
                lines.append("")
                lines.append("*Activities:*")
                for act, cnt in top:
                    lines.append(f"  - {act}: {cnt}x")

        lines.append("")
        lines.append("_Sent by Third Eye Shield_")
        text = "\n".join(lines)

        thread = threading.Thread(
            target=self._send_to_all,
            args=(text, "Markdown"),
            daemon=True,
        )
        thread.start()

    def _format_message(self, message, level, context):
        """Format alert with emoji prefix and optional context."""
        prefix = {
            "info": "ℹ️",
            "concern": "⚠️",
            "alert": "🚨",
            "active": "💪",
            "sedentary": "🪑",
        }.get(level, "ℹ️")

        ts = datetime.now().strftime("%H:%M:%S")
        parts = [f"{prefix} *Third Eye Shield Alert* [{ts}]", "", message]

        if context:
            parts.append("")
            wn = context.get('wellness_name', '')
            if wn:
                parts.append(f"Status: {wn}")
            action = context.get('action', '')
            if action and action != '(idle)':
                parts.append(f"Activity: {action}")
            ps = context.get('posture_score')
            if ps is not None:
                parts.append(f"Posture: {ps:.0f}/100")
            sed = context.get('sedentary_minutes', 0)
            if sed >= 5:
                parts.append(f"Inactive: {sed:.0f} min")

        parts.append("")
        parts.append("_Sent by Third Eye Shield_")
        return "\n".join(parts)

    def _send_to_all(self, text, parse_mode=None):
        """Send text to all configured chat IDs (blocking, run in thread)."""
        for chat_id in self.chat_ids:
            try:
                self._send_message(chat_id, text, parse_mode)
            except Exception as e:
                print(f"[TELEGRAM] Send failed to {chat_id}: {e}")

    def _send_message(self, chat_id, text, parse_mode=None):
        """Send a single message via Telegram Bot API."""
        url = self.API_BASE.format(token=self.bot_token, method="sendMessage")
        payload = {
            "chat_id": str(chat_id),
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                if not result.get("ok"):
                    print(f"[TELEGRAM] API error: {result}")
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            print(f"[TELEGRAM] HTTP {e.code}: {body[:200]}")
        except urllib.error.URLError as e:
            print(f"[TELEGRAM] Network error: {e.reason}")

    def test_connection(self):
        """Send a test message to verify bot token and chat IDs work."""
        if not self.enabled:
            print("[TELEGRAM] Not enabled -- cannot test")
            return False
        try:
            self._send_to_all("✅ Third Eye Shield Telegram connection test successful!")
            print("[TELEGRAM] Test message sent")
            return True
        except Exception as e:
            print(f"[TELEGRAM] Test failed: {e}")
            return False
