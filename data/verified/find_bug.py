import pandas as pd
import glob

print("🚀 Step 1: Loading all VERIFIED CSVs...")

# 1. Grab all verified files
file_list = glob.glob('*.csv') # Make sure this runs ONLY in the folder with verified CSVs
df_list = [pd.read_csv(file) for file in file_list]
df = pd.concat(df_list, ignore_index=True)

# 2. Clean the Data (No Mercy Rules)
df['verdict'] = df['verdict'].replace('TIMEOUT', 'ERROR') # Architecture compliance

print(f"✅ Master Verified Data Ready: {len(df)} rows.")

# 3. BUG EXTRACTION (The final piece of the paper)
print("\n🔍 HUNTING FOR A PREMIUM 'SAT' BUG...")
sat_bugs = df[df['verdict'] == 'SAT']

if not sat_bugs.empty:
    # Let's get a bug from Llama 3.1 or Gemini (The heavyweights)
    premium_bugs = sat_bugs[sat_bugs['model_name'].str.contains('llama|gemini', case=False)]
    
    if not premium_bugs.empty:
        bug = premium_bugs.iloc[0]
    else:
        bug = sat_bugs.iloc[0]

    print(f"\n🛑 FATAL LOGIC BUG CAUGHT!")
    print(f"Model: {bug['model_name']}")
    print(f"Problem ID: {bug['id']}")
    print("\n" + "="*50)
    print("📜 ORIGINAL CODE O(N^2) [TRUSTED]")
    print("="*50)
    print(bug['original_code'])
    print("\n" + "="*50)
    print("🤖 GENERATED CODE O(N) [HALLUCINATION]")
    print("="*50)
    print(bug['generated_code'])
    
    print("\n⚠️ Note: Since Z3 counterexample is missing from columns, just paste the codes here. I will find the bug manually!")
else:
    print("❌ No SAT bugs found! Data might be corrupted.")