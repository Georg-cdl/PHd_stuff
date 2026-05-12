# import pyvista as pv
# from glob import glob
# import numpy as np
# import matplotlib.pyplot as plt

# # -------------------------------------------------------------------
# # Load VTK files
# # -------------------------------------------------------------------

# path = "../post/run-*.liggghts.vtk"
# files = sorted(glob(path))

# print(f"Found {len(files)} VTK files")

# # -------------------------------------------------------------------
# # Read latest timestep
# # -------------------------------------------------------------------

# mesh = pv.read(files[-1])

# # -------------------------------------------------------------------
# # Extract RTD fields
# # -------------------------------------------------------------------

# rtd = mesh["f_rtd[1]"]
# inside = mesh["f_rtd[2]"]

# # -------------------------------------------------------------------
# # Keep only particles currently inside ROI
# # ---------------------------------------------

# mask = inside > 0

# rtd_inside = rtd[mask]

# # -------------------------------------------------------------------
# # Basic statistics
# # -------------------------------------------------------------------

# print("\n--- RTD Statistics ---")

# print(f"Particles inside ROI: {len(rtd_inside)}")

# if len(rtd_inside) > 0:

#     print(f"Minimum RTD: {rtd_inside.min():.6f} s")
#     print(f"Maximum RTD: {rtd_inside.max():.6f} s")
#     print(f"Mean RTD:    {rtd_inside.mean():.6f} s")
#     print(f"Median RTD:  {np.median(rtd_inside):.6f} s")

# else:

#     print("No particles currently inside ROI")

# # -------------------------------------------------------------------
# # Plot RTD histogram
# # -------------------------------------------------------------------

# if len(rtd_inside) > 0:

#     plt.figure(figsize=(8,5))

#     plt.hist(rtd_inside, bins=30)

#     plt.xlabel("Residence Time in ROI [s]")
#     plt.ylabel("Particle Count")

#     plt.title("Residence Time Distribution")

#     plt.grid(True)

#     plt.show()


import pyvista as pv
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------

path = "../post/run-*.liggghts.vtk"
files = sorted(glob(path))

# ---------------------------------------------------------
# Storage arrays
# ---------------------------------------------------------

times = []
mean_rtds = []
max_rtds = []
n_particles = []

# ---------------------------------------------------------
# Loop over timesteps
# ---------------------------------------------------------

for file in files:

    mesh = pv.read(file)

    rtd = mesh["f_rtd[1]"]

    inside = mesh["f_rtd[2]"]

    mask = inside > 0

    rtd_inside = rtd[mask]

    # extract timestep from filename
    step = int(file.split("run-")[1].split(".")[0])

    # convert to physical time
    sim_time = step * 1.0416666666666666e-05

    times.append(sim_time)

    if len(rtd_inside) > 0:

        mean_rtds.append(rtd_inside.mean())
        max_rtds.append(rtd_inside.max())
        n_particles.append(len(rtd_inside))

    else:

        mean_rtds.append(0)
        max_rtds.append(0)
        n_particles.append(0)

# ---------------------------------------------------------
# Plot mean RTD evolution
# ---------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(times, mean_rtds)

plt.xlabel("Simulation Time [s]")
plt.ylabel("Mean RTD in ROI [s]")

plt.title("Evolution of Mean Residence Time")

plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Plot max RTD evolution
# ---------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(times, max_rtds)

plt.xlabel("Simulation Time [s]")
plt.ylabel("Maximum RTD in ROI [s]")

plt.title("Evolution of Maximum Residence Time")

plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Plot ROI occupancy
# ---------------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(times, n_particles)

plt.xlabel("Simulation Time [s]")
plt.ylabel("Particles inside ROI")

plt.title("ROI Occupancy Evolution")

plt.grid(True)

plt.show()