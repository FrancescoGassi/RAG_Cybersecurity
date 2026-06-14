import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_names import BODMASFeatureNames

def test_feature_count():
    names = BODMASFeatureNames.get_all_feature_names()
    print(f"Numero feature generato: {len(names)}")
    assert len(names) == 2381, f"ERRORE: trovato {len(names)}, dovrebbe essere 2381"
    print("✓ Conteggio feature corretto (2381).")
    print("\nPrime 10 feature:")
    for i, name in enumerate(names[:10]):
        print(f"  {i}: {name}")

if __name__ == "__main__":
    test_feature_count()