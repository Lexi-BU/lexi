import matplotlib.pyplot as plt
import pickle


# Get the data from global variables
counts = global_variables.data_lin["counts"]
xedges = global_variables.data_lin["xedges"]
yedges = global_variables.data_lin["yedges"]

# Save the array as an image
plt.figure()
plt.imshow(counts.T, cmap="gray", origin="lower")
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.title("Image of the moon")
plt.savefig("../figures/image.png", dpi=300)


# Save the data to a pickle file
with open(
    "../data/2022_04_28_1313_LEXI_Sci_unit1_mcp_unit1_eBox_1900V_counts.pickle", "wb"
) as f:
    pickle.dump(counts, f)
    pickle.dump(xedges, f)
    pickle.dump(yedges, f)

# Close the pickle file
f.close()

# Open the pickle file and read the data
with open(
    "../data/2022_04_28_1313_LEXI_Sci_unit1_mcp_unit1_eBox_1900V_counts.pickle", "rb"
) as f:
    counts = pickle.load(f)
    xedges = pickle.load(f)
    yedges = pickle.load(f)
