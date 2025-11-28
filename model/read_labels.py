import pandas as pd
from pathlib import Path


def read_file_labels(path: Path) -> pd.DataFrame:
    """
    Type:
      0: mouse_bite
      1: spur
      2: missing_hole
      3: short
      4: open_circuit
      5: spurious_copper
    X_max, X_min, Y_max, Y_min:
      - coords of detected objects
    """
    names = ['Type', 'X_max', 'X_min', 'Y_max', 'Y_min', ]
    df = pd.read_csv(path, sep=' ', names=names)
    print(df)
    return df
