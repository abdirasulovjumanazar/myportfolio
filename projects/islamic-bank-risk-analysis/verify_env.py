import sys
import os

print("\n" + "="*60)
print(" 🕌 Environment Verification Script 🕌")
print("="*60)
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Working Directory: {os.getcwd()}")
print("="*60 + "\n")

try:
    import pydantic
    import numpy
    import pandas
    import sklearn
    print("✅ All key libraries (pydantic, numpy, pandas, sklearn) found in this environment!")
except ImportError as e:
    print(f"❌ Missing library: {e}")
except Exception as e:
    print(f"❌ Unexpected error when importing: {e}")

print("\n" + "="*60)
print(" 🛠️  HOW TO FIX THE 'COULD NOT FIND IMPORT' ERRORS IN VS CODE:")
print("="*60)
print(f"1. Open the Command Palette: Ctrl + Shift + P")
print(f"2. Select: 'Python: Select Interpreter'")
print(f"3. Select the following path:")
print(f"   {sys.executable}")
print(f"4. Reload Window: Ctrl + Shift + P -> 'Developer: Reload Window'")
print("="*60 + "\n")
