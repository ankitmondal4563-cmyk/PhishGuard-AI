"""Train both models:  python train_all.py"""
from src import train_text, train_url

if __name__ == "__main__":
    print("=== URL model ===")
    train_url.train()
    print("\n=== Text model ===")
    train_text.train()
