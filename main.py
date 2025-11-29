import os
import sys
from src.etl import load_data
from src.feature_eng import engineer_features
from agent.chat_bot import SpotifyAgent

def main():
    # Basic path setup. main.py sits in the project root,
    # so the raw data folder is simply data/raw.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(base_dir, "data", "raw")

    try:
        # Load raw JSON files
        df = load_data(raw_data_path)
        
        # Apply all feature engineering steps
        df = engineer_features(df)
        
        # Spin up the interactive agent with processed data
        my_agent = SpotifyAgent(df)

        print("\n" + "="*60) 
        print("SPOTIFY DATA AGENT IS LIVE!") 
        print("Try asking:") 
        print("What is Time listened in mins this (month, year)?'") 
        print("Who is my top artist in (year)?'") 
        print("Song count top 5 songs in (month) (year)'") 
        print("When do I listen on (Day), (Month) (Year)?'") 
        print("Top artist on 15th August 2024 (Specific Day)'") 
        print("(Type 'exit' to stop)") 
        print("="*60 + "\n")

        # Simple REPL loop
        while True:
            user_input = input("YOU: ").strip()
            
            if user_input.lower() in ["exit", "quit", "stop"]:
                print("Bye!")
                break

            response = my_agent.chat(user_input)
            print(f"AGENT: {response}\n")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Tip: Make sure your JSON files are inside 'data/raw/'")

if __name__ == "__main__":
    main()
