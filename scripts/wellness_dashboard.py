#!/usr/bin/env python3
"""
Third Eye Shield — Wellness Dashboard (Daily Summary)
===============================================

Generates a daily wellness summary from CSV logs.
Can be run standalone or as a module imported by the wellness monitor.

Usage:
    python3 scripts/wellness_dashboard.py                       # today's summary
    python3 scripts/wellness_dashboard.py --date 2026-03-18     # specific date
    python3 scripts/wellness_dashboard.py --render summary.png  # save as image
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, date
from pathlib import Path

import numpy as np
import cv2


def load_wellness_logs(log_dir="logs/wellness", target_date=None):
    """
    Load all wellness CSV logs for a given date.

    Returns list of dicts with parsed fields.
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []

    if target_date is None:
        target_date = date.today()
    date_str = target_date.strftime('%Y%m%d')

    entries = []
    for csv_path in sorted(log_dir.glob(f"wellness_{date_str}*.csv")):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['_file'] = csv_path.name
                entries.append(row)

    return entries


def compute_daily_summary(entries):
    """
    Compute aggregate daily wellness metrics from log entries.

    Returns dict with summary statistics.
    """
    if not entries:
        return {"total_events": 0, "message": "No data for this date."}

    summary = {
        "total_events": len(entries),
        "fall_alerts": 0,
        "concerns": 0,
        "person_events": 0,
        "actions": Counter(),
        "emotions": Counter(),
        "posture_scores": [],
        "wellness_levels": Counter(),
        "sedentary_max": 0.0,
        "first_event": None,
        "last_event": None,
    }

    for e in entries:
        event = e.get('event', '')
        if event == 'FALL_ALERT':
            summary['fall_alerts'] += 1
        elif event == 'CONCERN':
            summary['concerns'] += 1
        elif event in ('PERSON_ENTERED', 'PERSON_LEFT'):
            summary['person_events'] += 1

        action = e.get('action', '').strip()
        if action and action != '(idle)':
            summary['actions'][action] += 1

        emotion = e.get('emotion', '').strip()
        if emotion:
            summary['emotions'][emotion] += 1

        ps = e.get('posture_score', '').strip()
        if ps:
            try:
                summary['posture_scores'].append(float(ps))
            except ValueError:
                pass

        wl = e.get('wellness_level', '').strip()
        if wl:
            summary['wellness_levels'][wl] += 1

        sed = e.get('sedentary_min', '').strip()
        if sed:
            try:
                summary['sedentary_max'] = max(summary['sedentary_max'], float(sed))
            except ValueError:
                pass

        ts = e.get('timestamp', '')
        if ts:
            if summary['first_event'] is None:
                summary['first_event'] = ts
            summary['last_event'] = ts

    # Aggregate posture
    if summary['posture_scores']:
        scores = summary['posture_scores']
        summary['posture_avg'] = np.mean(scores)
        summary['posture_min'] = np.min(scores)
        summary['posture_max'] = np.max(scores)
    else:
        summary['posture_avg'] = None

    return summary


def format_summary_text(summary, target_date=None):
    """Format summary as readable text."""
    if target_date is None:
        target_date = date.today()

    lines = [
        f"{'='*50}",
        f"  Third Eye Shield Daily Wellness Report",
        f"  Date: {target_date.strftime('%A, %d %B %Y')}",
        f"{'='*50}",
        "",
    ]

    if summary['total_events'] == 0:
        lines.append("  No wellness data recorded for this date.")
        return "\n".join(lines)

    lines.append(f"  Monitoring period: {summary['first_event']} -- {summary['last_event']}")
    lines.append("")

    # Alerts
    lines.append(f"  Fall Alerts:     {summary['fall_alerts']}")
    lines.append(f"  Concerns:        {summary['concerns']}")
    lines.append(f"  Person Events:   {summary['person_events']}")
    lines.append("")

    # Activity
    if summary['actions']:
        lines.append("  Activities Detected:")
        for action, count in summary['actions'].most_common(10):
            lines.append(f"    - {action}: {count}x")
        lines.append("")

    # Posture
    if summary['posture_avg'] is not None:
        lines.append(f"  Posture Score:   avg {summary['posture_avg']:.0f}/100  "
                      f"(range: {summary['posture_min']:.0f}-{summary['posture_max']:.0f})")
    lines.append(f"  Max Sedentary:   {summary['sedentary_max']:.0f} min")
    lines.append("")

    # Emotions (if any)
    if summary['emotions']:
        lines.append("  Emotions (opt-in):")
        for emo, count in summary['emotions'].most_common():
            lines.append(f"    - {emo}: {count}x")
        lines.append("")

    # Wellness level distribution
    if summary['wellness_levels']:
        from src.utils.wellness_features import WELLNESS_NAMES
        lines.append("  Wellness Level Distribution:")
        for level_str, count in sorted(summary['wellness_levels'].items()):
            try:
                name = WELLNESS_NAMES.get(int(level_str), level_str)
            except ValueError:
                name = level_str
            lines.append(f"    - {name}: {count} events")

    lines.append("")
    lines.append(f"{'='*50}")
    return "\n".join(lines)


def render_summary_image(summary, target_date=None, width=800, height=480):
    """Render summary as an OpenCV image for display/screenshot."""
    if target_date is None:
        target_date = date.today()

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 25, 20)  # dark background

    y = 35
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(img, "Third Eye Shield Daily Wellness Report", (20, y), font, 0.8, (255, 200, 80), 2)
    y += 30
    cv2.putText(img, target_date.strftime('%A, %d %B %Y'), (20, y), font, 0.6, (180, 180, 180), 1)
    y += 35

    if summary['total_events'] == 0:
        cv2.putText(img, "No data for this date.", (20, y), font, 0.7, (150, 150, 150), 1)
        return img

    # Alerts panel
    cv2.rectangle(img, (15, y-5), (width//2 - 10, y+80), (50, 40, 30), -1)
    cv2.putText(img, f"Falls: {summary['fall_alerts']}", (25, y+20), font, 0.6,
                (0, 0, 255) if summary['fall_alerts'] > 0 else (100, 100, 100), 2)
    cv2.putText(img, f"Concerns: {summary['concerns']}", (25, y+45), font, 0.6,
                (0, 140, 255) if summary['concerns'] > 0 else (100, 100, 100), 2)
    cv2.putText(img, f"Sedentary max: {summary['sedentary_max']:.0f} min", (25, y+70), font, 0.5,
                (0, 180, 255), 1)

    # Posture panel
    px = width // 2 + 10
    cv2.rectangle(img, (px, y-5), (width - 15, y+80), (50, 40, 30), -1)
    if summary.get('posture_avg') is not None:
        ps = summary['posture_avg']
        ps_col = (0, 200, 0) if ps >= 65 else (0, 180, 255) if ps >= 35 else (0, 0, 255)
        cv2.putText(img, f"Posture: {ps:.0f}/100", (px+10, y+20), font, 0.6, ps_col, 2)
        cv2.putText(img, f"Range: {summary['posture_min']:.0f}-{summary['posture_max']:.0f}",
                    (px+10, y+45), font, 0.5, (150, 150, 150), 1)
    else:
        cv2.putText(img, "Posture: N/A", (px+10, y+20), font, 0.6, (100, 100, 100), 1)

    y += 100

    # Activities
    if summary['actions']:
        cv2.putText(img, "Activities:", (20, y), font, 0.6, (200, 200, 200), 1)
        y += 25
        for action, count in summary['actions'].most_common(5):
            bar_len = min(count * 8, width - 200)
            cv2.rectangle(img, (160, y-12), (160 + bar_len, y+4), (0, 200, 100), -1)
            cv2.putText(img, f"{action}", (25, y), font, 0.45, (180, 180, 180), 1)
            cv2.putText(img, f"{count}x", (165 + bar_len, y), font, 0.4, (200, 200, 200), 1)
            y += 22

    # Emotions
    y += 10
    if summary['emotions']:
        cv2.putText(img, "Emotions (opt-in):", (20, y), font, 0.6, (200, 200, 200), 1)
        y += 25
        for emo, count in summary['emotions'].most_common(5):
            bar_len = min(count * 8, width - 200)
            cv2.rectangle(img, (160, y-12), (160 + bar_len, y+4), (255, 200, 0), -1)
            cv2.putText(img, f"{emo}", (25, y), font, 0.45, (180, 180, 180), 1)
            cv2.putText(img, f"{count}x", (165 + bar_len, y), font, 0.4, (200, 200, 200), 1)
            y += 22

    # Footer
    cv2.putText(img, "Third Eye Shield", (width - 120, height - 12), font, 0.5, (100, 100, 100), 1)
    return img


def main():
    parser = argparse.ArgumentParser(description="Third Eye Shield Daily Wellness Dashboard")
    parser.add_argument('--date', type=str, default=None,
                        help='Target date (YYYY-MM-DD). Default: today.')
    parser.add_argument('--log-dir', type=str, default='logs/wellness',
                        help='Wellness log directory')
    parser.add_argument('--render', type=str, default=None,
                        help='Save summary as image to this path')
    parser.add_argument('--display', action='store_true',
                        help='Show summary image in OpenCV window')
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        target_date = date.today()

    entries = load_wellness_logs(args.log_dir, target_date)
    summary = compute_daily_summary(entries)

    # Text summary to terminal
    print(format_summary_text(summary, target_date))

    # Image rendering
    if args.render or args.display:
        img = render_summary_image(summary, target_date)
        if args.render:
            cv2.imwrite(args.render, img)
            print(f"\n[SAVED] Summary image: {args.render}")
        if args.display:
            cv2.imshow("Third Eye Shield Daily Summary", img)
            print("[INFO] Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
