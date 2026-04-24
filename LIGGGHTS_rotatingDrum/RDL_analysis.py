import numpy as np
import matplotlib.pyplot as plt
import os
import re



#Extract dt if available
base_dir_param = os.path.dirname(os.path.abspath(__file__))
file_param = os.path.join(base_dir_param, "DEM", "data", "parameter_global.txt")

dt = None
default_dt = 1e-80  # <-- choose a sensible fallback for your case

file_param = os.path.join(base_dir_param, "DEM", "data", "parameter_global.txt")

with open(file_param, "r") as f:
    for line in f:
        line = line.strip()

        # ONLY look for direct dt definition
        if line.startswith("variable dt equal"):
            parts = line.split()

            try:
                dt = float(parts[-1])
            except:
                dt = None
            break

# fallback
if dt is None:
    print("WARNING: dt not found, using default =", default_dt)
    dt = default_dt

print("dt =", dt)

#Stream Timestep specific Particle entry and exit times
base_dir = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(base_dir, "DEM", "post", "roi.dump")

entry_time = {}        # pid -> entry timestep
residence_times = []   # final results

with open(file, "r") as f:
    while True:
        line = f.readline()
        if not line:
            break

        if "ITEM: TIMESTEP" in line:
            t = int(f.readline().strip())

            f.readline()  # NUMBER OF ATOMS
            n = int(f.readline().strip())

            f.readline()  # BOX BOUNDS
            f.readline()
            f.readline()
            f.readline()

            f.readline()  # ATOMS header

            current_ids = set()

            for _ in range(n):
                parts = f.readline().split()
                pid = int(parts[0])
                current_ids.add(pid)

                # ENTRY event (first time seen)
                if pid not in entry_time:
                    entry_time[pid] = t

            # EXIT detection (particles previously seen but now missing)
            for pid in list(entry_time.keys()):
                if pid not in current_ids:
                    residence_times.append((t - entry_time[pid]) * dt)
                    del entry_time[pid]
final_time = t  # last timestep seen

for pid, t0 in entry_time.items():
    residence_times.append((final_time - t0) * dt)

residence_times = np.array(residence_times)

#Plot Histogram of RTD
plt.figure()
plt.hist(residence_times, bins=100, density=True, alpha=0.75)
#plt.xlabel("Residence time (timesteps)")
plt.xlabel("Residence time [s]")
plt.ylabel("Probability density")
plt.title("Residence Time Distribution (RTD)")
plt.show()