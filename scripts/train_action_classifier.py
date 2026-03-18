#!/usr/bin/env python3
"""
Train a RandomForest classifier on collected kinematic features.
Reads all *_kin.npy files from collections/ directory.
Naming convention: S{subject}_C{classid}_V{sample}_kin.npy

Usage:
    python3 scripts/train_action_classifier.py
    python3 scripts/train_action_classifier.py --data_dir collections --out models/action_rf.pkl
"""
import os, sys, re, argparse, pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The 10 selected actions (must match collect_action_data.py)
CLASSES = {
    0: "clapping",
    1: "arm circles",
    2: "drink water",
    3: "falling",
    4: "kicking something",
    5: "sit down and up",
    6: "pointing with finger",
    7: "phone call",
    8: "punching/slapping",
    9: "pushing other person",
}

def load_dataset(data_dir):
    """Load all _kin.npy files, parse class ID from filename.
    Supports both naming conventions:
        S01_C023_V001_kin.npy  (old)
        S01_A00_001_kin.npy    (new)
    """
    X, y, names = [], [], []
    old_pat = re.compile(r"S(\d+)_C(\d+)_V(\d+)_kin\.npy")
    new_pat = re.compile(r"S(\d+)_A(\d+)_(\d+)_kin\.npy")
    
    for f in sorted(Path(data_dir).glob("*_kin.npy")):
        m = new_pat.match(f.name) or old_pat.match(f.name)
        if not m:
            print(f"  [SKIP] {f.name} (bad naming)")
            continue
        subject, class_id, sample = int(m.group(1)), int(m.group(2)), int(m.group(3))
        feats = np.load(f)
        X.append(feats)
        y.append(class_id)
        names.append(f.name)
    
    return np.array(X), np.array(y), names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='collections', help='Directory with _kin.npy files')
    parser.add_argument('--out', default='models/action_rf.pkl', help='Output model path')
    args = parser.parse_args()
    
    print(f"[1/4] Loading data from {args.data_dir}/...")
    X, y, names = load_dataset(args.data_dir)
    
    if len(X) == 0:
        print("ERROR: No _kin.npy files found. Run collect_kinematics.py first.")
        sys.exit(1)
    
    unique_classes = np.unique(y)
    print(f"       Loaded {len(X)} samples, {len(unique_classes)} classes")
    print(f"       Feature dimension: {X.shape[1]}")
    for c in unique_classes:
        label = CLASSES.get(c, f"unknown_{c}")
        count = np.sum(y == c)
        print(f"         Class {c:3d} ({label}): {count} samples")
    
    # Cross-validation
    print(f"\n[2/4] Cross-validation (Stratified 5-Fold)...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,           
        min_samples_leaf=2,     
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    n_splits = min(5, min(np.bincount(y)[np.bincount(y) > 0]))
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=skf, scoring='f1_macro')
        print(f"       Macro-F1 per fold: {[f'{s:.3f}' for s in scores]}")
        print(f"       Mean Macro-F1: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
    else:
        print(f"       Skipping CV (need >= 2 samples per class, min={n_splits})")
    
    # Train final model on ALL data
    print(f"\n[3/4] Training final model on all {len(X)} samples...")
    clf.fit(X, y)
    
    # Quick accuracy on training set (sanity check, should be ~100%)
    train_pred = clf.predict(X)
    train_acc = np.mean(train_pred == y)
    print(f"       Training accuracy: {train_acc:.1%}")
    
    # Feature importances (top 10)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    print(f"       Top 10 feature indices: {top_idx.tolist()}")
    
    # Save
    print(f"\n[4/4] Saving model to {args.out}...")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump({
            'model': clf,
            'classes': CLASSES,
            'feature_dim': X.shape[1],
            'n_samples': len(X),
            'n_classes': len(unique_classes),
        }, f)
    
    print(f"\n===== DONE =====")
    print(f"Model saved: {args.out}")
    print(f"To use in inference: pickle.load(open('{args.out}', 'rb'))['model'].predict(features)")

if __name__ == '__main__':
    main()
