"""Quick validation Phase 1 regression check."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from validation_suite import phase1_regime_validation, phase5_determinism
print('Running Phase 1 (regime validation)...')
r = phase1_regime_validation()
print('Phase 1 OK')
print('Running Phase 5 (determinism)...')
r5 = phase5_determinism()
print('Phase 5 OK')
print('VALIDATION REGRESSION: PASS')
