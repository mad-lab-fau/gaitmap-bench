# This script will generate representations of the different coordinate definitions used within gaitmap
# The resulting plots are used within the coordinate system guide

import matplotlib.pyplot as plt
import seaborn as sns
from gaitmap.example_data import get_healthy_example_imu_data, get_healthy_example_stride_borders
from gaitmap.preprocessing import sensor_alignment
from gaitmap.utils.consts import *
from gaitmap.utils.coordinate_conversion import convert_to_fbf

sns.set_context("talk", font_scale=1)

# Load gaitmap example dataset
example_dataset = get_healthy_example_imu_data()
example_stride_borders = get_healthy_example_stride_borders()
sampling_rate_hz = 204.8

# Align dataset to gravity: resulting in sensor frame representation
dataset_sf = sensor_alignment.align_dataset_to_gravity(example_dataset, sampling_rate_hz)

# Convert dataset to body frame: resulting in body frame representation
dataset_bf = convert_to_fbf(dataset_sf, right=["right_sensor"], left=["left_sensor"])

colors = ["#FFB81C", "#4C3B70", "#009B77"]


# helper to plot different coordinate frames
def plot_stride(data, column_names, sensor_id, stride_id, export_name):
    fig, axs = plt.subplots(2, figsize=(7, 7), sharex=True)
    start = example_stride_borders[sensor_id].iloc[stride_id].start
    end = example_stride_borders[sensor_id].iloc[stride_id].end
    axs[0].axhline(0, c="k", ls="--", lw=0.7)
    for i, col in enumerate(column_names[3:]):
        axs[0].plot(data[sensor_id][col].to_numpy()[start:end], label=col, color=colors[i])
    axs[0].set_title(sensor_id.split("_")[0].capitalize() + " Foot")
    # axs[0].legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
    axs[0].set_ylim([-600, 600])
    axs[0].set_ylabel("Rate of \nRotation [deg/s]")

    axs[1].axhline(-9.81, c="k", ls="--", lw=0.5)
    axs[1].axhline(+9.81, c="k", ls="--", lw=0.5)
    axs[1].axhline(0, c="k", ls="--", lw=0.7)
    for i, col in enumerate(column_names[:3]):
        axs[1].plot(data[sensor_id][col].to_numpy()[start:end], label=col, color=colors[i])
    axs[1].set_ylabel("Acceleration [m/s²]")
    axs[1].set_xlabel("Samples [#]")
    # axs[1].legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
    axs[1].set_ylim([-50, 50])
    plt.tight_layout()
    fig.savefig(export_name, bbox_inches="tight")


# %%
# Plot "Stride-Template" in Sensor Frame
plot_stride(dataset_sf, SF_COLS, "left_sensor", 5, "left_sensor_sensor_frame_template.pdf")
plot_stride(dataset_sf, SF_COLS, "right_sensor", 18, "right_sensor_sensor_frame_template.pdf")

# %%
# Plot "Stride-Template" in Body Frame
plot_stride(dataset_bf, BF_COLS, "left_sensor", 5, "left_sensor_body_frame_template.pdf")
plot_stride(dataset_bf, BF_COLS, "right_sensor", 18, "right_sensor_body_frame_template.pdf")

# %%
# Just plot a legend with x,y,z in the right order
fig, axs = plt.subplots(1, figsize=(7, 1))
for i, col in enumerate(["x", "y", "z"]):
    axs.plot([0, 1], [0, 0], color=colors[i], label=col)
axs.set_xlim([0, 1])
axs.set_ylim([0, 1])
axs.axis("off")
axs.legend(loc="center", ncols=3)
plt.tight_layout()
fig.savefig("legend_sensor_frame.pdf", bbox_inches="tight")

# %%
# At the same for the body frame
fig, axs = plt.subplots(1, figsize=(7, 1))
for i, col in enumerate(BF_ACC):
    axs.plot([0, 1], [0, 0], color=colors[i], label=col.split("_")[1])
axs.set_xlim([0, 1])
axs.set_ylim([0, 1])
axs.axis("off")
axs.legend(loc="center", ncols=3)
plt.tight_layout()
fig.savefig("legend_body_frame.pdf", bbox_inches="tight")
