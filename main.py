"""
TRUTHSCAN AI - Command Line Interface (CLI) Entry Point
Enables data preparation, model training, test evaluation, and text inference from terminal.
"""

import os
import sys
import argparse

import config
from src.download_data import ensure_dataset
from src.dataset import prepare_data, get_dataset_statistics, check_dataset_exists
from src.train import train_pipeline
from src.evaluate import evaluate_models
from src.predict import NewsPredictor

# Top-level exports for Vercel discovery if root entrypoints are scanned
from api.index import handler, ApiHandler, app, application


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="TRUTHSCAN AI: Deep Learning-Based Fake News Detection Using CNN and LSTM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--download-data", action="store_true",
                        help="Download authentic ISOT dataset files if missing")
    parser.add_argument("--prepare-data", action="store_true",
                        help="Clean raw dataset and create stratified 80/10/10 train/val/test splits")
    parser.add_argument("--train", choices=["cnn", "lstm", "both"],
                        help="Train selected model(s): cnn, lstm, or both")
    parser.add_argument("--mode", choices=["quick", "normal", "full"], default="quick",
                        help="Training mode: quick (5k samples), normal (20k samples), full (all data)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate trained models on untouched test split and generate comparison plots")
    parser.add_argument("--predict", action="store_true",
                        help="Make a prediction on input text provided via --title and --text")
    parser.add_argument("--model", choices=["cnn", "lstm", "both"], default="both",
                        help="Model to use for prediction (default: both)")
    parser.add_argument("--title", type=str, default="",
                        help="Headline/Title of the news article for prediction")
    parser.add_argument("--text", type=str, default="",
                        help="Body text of the news article for prediction")
    parser.add_argument("--stats", action="store_true",
                        help="Display dataset balance and split statistics")
    parser.add_argument("--serve", action="store_true",
                        help="Launch the standard local web application server")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # If no flags passed, print help and summary
    if len(sys.argv) == 1:
        print("=" * 60)
        print("  TRUTHSCAN AI - Fake News Detection CLI")
        print("  Deep Learning-Based Fake News Detection Using CNN and LSTM")
        print("=" * 60)
        print("\nAvailable commands:")
        print("  python main.py --download-data      # Download ISOT Fake.csv & True.csv")
        print("  python main.py --stats              # View dataset statistics")
        print("  python main.py --prepare-data       # Preprocess & stratify dataset")
        print("  python main.py --train both --mode quick   # Train CNN and LSTM")
        print("  python main.py --evaluate           # Evaluate on test set & generate charts")
        print("  python main.py --predict --model both --title '...' --text '...'")
        print("\nTo launch the web application:")
        print("  python server.py")
        print("  (or: python main.py --serve)")
        print("=" * 60)
        return

    # 1. Download data
    if args.download_data:
        print("[TRUTHSCAN AI] Verifying and retrieving dataset...")
        ensure_dataset()
        print("[TRUTHSCAN AI] Dataset verification complete.")

    # 2. Stats
    if args.stats:
        stats = get_dataset_statistics()
        if not stats.get("available", False):
            print("\n" + stats.get("error", "Dataset not available."))
            return
        print("\n--- ISOT Dataset Statistics ---")
        print(f"Total Articles:      {stats['total_articles']}")
        print(f"Fake Articles:       {stats['fake_articles']}")
        print(f"Real Articles:       {stats['real_articles']}")
        print(f"Train Split (80%):   {stats['train_samples']}")
        print(f"Val Split (10%):     {stats['val_samples']}")
        print(f"Test Split (10%):    {stats['test_samples']}")
        print(f"Avg Article Length:  {stats['avg_article_length']} words\n")

    # 3. Prepare data
    if args.prepare_data:
        exists, err_msg = check_dataset_exists()
        if not exists:
            print("\n" + err_msg)
            sys.exit(1)
        print(f"[TRUTHSCAN AI] Preparing dataset splits in {args.mode.upper()} mode...")
        _, tokenizer = prepare_data(mode=args.mode)
        print(f"[TRUTHSCAN AI] Done. Built vocabulary with {tokenizer.get_vocab_size()} tokens.")

    # 4. Train
    if args.train:
        exists, err_msg = check_dataset_exists()
        if not exists:
            print("\n" + err_msg)
            sys.exit(1)
        print(f"[TRUTHSCAN AI] Starting training pipeline for {args.train.upper()} ({args.mode.upper()} mode)...")
        train_pipeline(target=args.train, mode=args.mode)
        print(f"[TRUTHSCAN AI] Training finished successfully.")

    # 5. Evaluate
    if args.evaluate:
        print("[TRUTHSCAN AI] Running test set evaluation...")
        evaluate_models()
        print("[TRUTHSCAN AI] Evaluation complete. Results saved to results/ directory.")

    # 6. Predict
    if args.predict:
        if not args.title and not args.text:
            print("Error: Please enter a headline or article before analyzing.")
            sys.exit(1)
        predictor = NewsPredictor()
        if args.model == "both":
            res = predictor.predict_both(args.title, args.text)
            print("\n--- Model Prediction Comparison ---")
            print(f"CNN  Prediction: {res['cnn']['prediction']} (Confidence: {res['cnn']['confidence']}%)")
            print(f"LSTM Prediction: {res['lstm']['prediction']} (Confidence: {res['lstm']['confidence']}%)")
        else:
            res = predictor.predict_single(args.title, args.text, model_type=args.model)
            print(f"\nModel:           {res['model']}")
            print(f"Model Prediction: {res['prediction']}")
            print(f"Confidence:       {res['confidence']}%")

    # 7. Serve Web App
    if args.serve:
        from server import run_server
        run_server()


if __name__ == "__main__":
    main()
