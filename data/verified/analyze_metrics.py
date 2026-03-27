import os
import glob
import pandas as pd

def analyze_gemini_results(csv_filepath):
    model_name = os.path.splitext(os.path.basename(csv_filepath))[0]
    
    # Load the master dataset
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"ERROR: Could not find {csv_filepath}. Check the filename!")
        return

    total_problems = len(df)
    
    print("-" * 50)
    print(f"Model Name: {model_name}")

    print("-" * 50)
    print("📊 1. OPTIMIZATION SUCCESS RATE (Big-O Shift)")
    print("-" * 50)
    # Check how many times the model actually improved the complexity
    success_df = df[df['complexity_improved'] == 'True']
    success_count = len(success_df)
    success_rate = (success_count / total_problems) * 100
    print(f"Successful $O(N)$ Reductions: {success_count} / {total_problems} ({success_rate:.1f}%)")
    
    failed_count = total_problems - success_count
    failed_rate = (failed_count / total_problems) * 100
    print(f"Failed to Optimize / Regression: {failed_count} / {total_problems} ({failed_rate:.1f}%)\n")

    print("-" * 50)
    print("⚖️ 2. FORMAL VERIFICATION VERDICTS (Equivalence)")
    print("-" * 50)
    # Calculate distribution of UNSAT, SAT, WARNING, ERROR
    verdict_counts = df['verdict'].value_counts()
    for verdict, count in verdict_counts.items():
        percentage = (count / total_problems) * 100
        print(f"{verdict.ljust(10)}: {count} cases ({percentage:.1f}%)")
    print("\n")

    print("-" * 50)
    print("⏱️ 3. PERFORMANCE & LATENCY")
    print("-" * 50)
    # Calculate mean reasoning tokens and latency
    avg_latency_sec = df['verify_latency_ms'].mean() / 1000
    max_latency_sec = df['verify_latency_ms'].max() / 1000
    avg_reasoning = df['reasoning_tokens'].mean()
    
    print(f"Average Verification Latency : {avg_latency_sec:.2f} seconds")
    print(f"Maximum Verification Latency : {max_latency_sec:.2f} seconds")
    print(f"Average Reasoning Tokens Used: {avg_reasoning:.0f} tokens")

if __name__ == "__main__":
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("No CSV files found in the current directory.")
    else:
        print("Available CSV files:")
        for i, file in enumerate(csv_files):
            print(f"{i + 1}. {file}")
        
        try:
            choice = input("\nEnter the number of the CSV file to analyze (or type 'all' to run all, or press Enter to cancel): ")
            if choice.strip() == "":
                print("Analysis cancelled.")
            elif choice.strip().lower() == "all":
                for file in csv_files:
                    analyze_gemini_results(file)
                    print()
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(csv_files):
                    selected_file = csv_files[choice_idx]
                    analyze_gemini_results(selected_file)
                else:
                    print("Invalid choice. Please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a valid number or 'all'.")