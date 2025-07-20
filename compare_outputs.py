# compare_outputs.py
import sys
import numpy as np

# Read two files: C output and Python output
with open(sys.argv[1]) as f1, open(sys.argv[2]) as f2:
    vals1 = np.fromstring(f1.read(), sep=' ')
    vals2 = np.fromstring(f2.read(), sep=' ')

# Compare with tolerance
if np.allclose(vals1, vals2, atol=1e-4):
    print("✅ Outputs match")
    sys.exit(0)
else:
    print("❌ Outputs differ")
    print("C output:     ", vals1)
    print("Python output:", vals2)
    sys.exit(1)
