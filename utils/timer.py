import time
from contextlib import contextmanager
import numpy as np

class Timer:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timer = {}

    def reset_timer(self):
        self._timer = {}

    def print_runtime_summary(self, prefix="", print_each_iteration=False):
        print(f"-- {prefix} runtime summary --")
        summary_dict = {}

        for key, vals in self._timer.items():
            if len(vals) > 1:
                vals = np.array(vals)
                if print_each_iteration:
                    for val_idx, val in enumerate(vals):
                        augmented_key = f"{key}_{val_idx}"
                        print(f'\t{augmented_key:30}: {val:.3f} s')

                stats = {
                    "count": len(vals),
                    "sum": vals.sum(),
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "min": vals.min(),
                    "max": vals.max()
                }
                summary_dict[key] = stats
                print(f'\t{key:30} ({len(vals)}): {vals.sum():.3f} s (mean: {vals.mean():.3f} s, std: {vals.std():.3f} s, min: {vals.min():.3f} s, max: {vals.max():.3f} s)')
            else:
                val = vals[0]
                summary_dict[key] = {
                    "count": 1,
                    "sum": val,
                    "mean": val,
                    "std": 0.0,
                    "min": val,
                    "max": val
                }
                print(f'\t{key:30}: {val:.3f} s')
        print("----------------------")
        return summary_dict
    
    @contextmanager
    def trace(self, lap_name, overwrite:bool = False, verbose:bool=False):
        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        if lap_name in self._timer and not overwrite:
            self._timer[lap_name].append(elapsed_time) 
        else:
            self._timer[lap_name] = [elapsed_time]
        if verbose:
            print(f'{lap_name:30}: {elapsed_time:.3f} s')

