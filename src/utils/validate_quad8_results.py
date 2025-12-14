from pathlib import Path
import numpy as np
import pandas as pd

# =================================================
# CONFIGURATION — EDIT PATHS ONLY
# =================================================
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

FILES = {
    "Octave": PROJECT_ROOT / "../MATLAB_G62_AC/QUAD8/Results_quad8_V2.xlsx",
    "Python_CPU": HERE / "../../data/output/Results_quad8_CPU.xlsx",
    "Python_GPU": HERE / "../../data/output/Results_quad8_GPU.xlsx",   # optional
}

# FEM-appropriate tolerances: (atol, rtol)
TOL = {
    "u": (1e-10, 1e-12),
    "vel": (1e-9, 1e-9),
    "abs_vel": (1e-9, 1e-9),
    "pressure": (1e-8, 1e-8),
}

# =================================================
# HELPERS
# =================================================
def diff_stats(ref, test, name, atol, rtol):
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)

    diff = np.abs(test - ref)
    max_abs = np.nanmax(diff)

    mask = np.abs(ref) > atol
    max_rel = np.nanmax(diff[mask] / np.abs(ref[mask])) if np.any(mask) else 0.0

    status = (max_abs <= atol) or (max_rel <= rtol)

    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "status": status,
        "atol": atol,
        "rtol": rtol,
    }


def load_quantities(df):
    """
    Assumes agreed Excel layout:
    Row 0 = headers
    """
    return {
        "u":        df.iloc[1:, 1].to_numpy(),
        "velx":     df.iloc[1:, 4].to_numpy(),
        "vely":     df.iloc[1:, 5].to_numpy(),
        "abs_vel":  df.iloc[1:, 6].to_numpy(),
        "pressure": df.iloc[1:, 7].to_numpy(),
    }


# =================================================
# LOAD FILES
# =================================================
print("\nLoading result files...")
data = {}

for name, path in FILES.items():
    if path.exists():
        data[name] = load_quantities(pd.read_excel(path, header=None))
        print(f"  ✓ {name}: {path}")
    else:
        print(f"  ✗ {name}: {path} (missing)")

if len(data) < 2:
    raise RuntimeError("Need at least two result files to compare.")

# Reference = first entry
ref_name = list(data.keys())[0]
ref = data[ref_name]

print(f"\nUsing reference: {ref_name}\n")

# =================================================
# VALIDATION
# =================================================
overall_ok = True

for test_name, test in data.items():
    if test_name == ref_name:
        continue

    print(f"Comparing {test_name} → {ref_name}\n")

    r = diff_stats(ref["u"], test["u"], "u", *TOL["u"])
    print(f"Potential u (nodal):")
    print(f"  max abs error = {r['max_abs']:.3e}")
    print(f"  max rel error = {r['max_rel']:.3e}")
    print(f"  status        = {'PASS' if r['status'] else 'FAIL'}\n")
    overall_ok &= r["status"]

    r = diff_stats(ref["velx"], test["velx"], "velx", *TOL["vel"])
    print(f"Velocity x (element):")
    print(f"  max abs error = {r['max_abs']:.3e}")
    print(f"  max rel error = {r['max_rel']:.3e}")
    print(f"  status        = {'PASS' if r['status'] else 'FAIL'}\n")
    overall_ok &= r["status"]

    r = diff_stats(ref["vely"], test["vely"], "vely", *TOL["vel"])
    print(f"Velocity y (element):")
    print(f"  max abs error = {r['max_abs']:.3e}")
    print(f"  max rel error = {r['max_rel']:.3e}")
    print(f"  status        = {'PASS' if r['status'] else 'FAIL'}\n")
    overall_ok &= r["status"]

    r = diff_stats(ref["abs_vel"], test["abs_vel"], "|V|", *TOL["abs_vel"])
    print(f"|V| (element):")
    print(f"  max abs error = {r['max_abs']:.3e}")
    print(f"  max rel error = {r['max_rel']:.3e}")
    print(f"  status        = {'PASS' if r['status'] else 'FAIL'}\n")
    overall_ok &= r["status"]

    r = diff_stats(ref["pressure"], test["pressure"], "pressure", *TOL["pressure"])
    print(f"Pressure (element):")
    print(f"  max abs error = {r['max_abs']:.3e}")
    print(f"  max rel error = {r['max_rel']:.3e}")
    print(f"  status        = {'PASS' if r['status'] else 'FAIL'}\n")
    overall_ok &= r["status"]

    print("-" * 60)

# =================================================
# SUMMARY
# =================================================
print("=" * 60)
if overall_ok:
    print("OVERALL RESULT: PASS")
    print("All results match reference within FEM tolerances.")
else:
    print("OVERALL RESULT: FAIL")
    print("At least one comparison exceeded tolerance.")
print("=" * 60)
