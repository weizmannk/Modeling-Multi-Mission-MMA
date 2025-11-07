import glob
from astropy.table import QTable
import numpy as np
import logging

# GW parameters used to simulate lightcurve
output = "ouptput_detlimit"
table = QTable.read("./data/uvex_bns_O5.ecsv")
run = ["O5"]
m4opt_ouput_dir = "data/uvex-limmag"

event_ids_with_png = []  # List to store event_ids that have .png files
event_data = []

for event_file in glob.glob(f"{m4opt_ouput_dir}/*.ecsv"):
    try:
        data = QTable.read(event_file)
        data_args = data.meta["args"]
        event_id_str = data_args["skymap"].split("/")[-1].split(".")[0]
        # IMPORTANT: Convert to int to avoid string/int comparison issues
        event_id = int(event_id_str)
    except ValueError as exc:
        logging.error(f"Failed to parse event ID from {event_file}: {exc}")
        continue
    except Exception as exc:
        logging.error(f"Failed to read event file {event_file}: {exc}")
        continue

    # Check if there are .png files in the directory
    outdir = f"{output}/{event_id}/*_lightcurves.png"
    png_files = glob.glob(outdir)
    
    if png_files:
        # If .png files exist, save the event_id
        event_ids_with_png.append(event_id)
        print(f"event_id {event_id} has {len(png_files)} lightcurves file(s)")


        # Filter table by run and event_id
        mask = np.isin(table["run"], run) & (table["simulation_id"] == event_id)
        if mask.any():
            row = table[mask][0]
            mass1 = row["mass1"]
            mass2 = row["mass2"]
            distance = row["distance"]
            
            event_data.append({
                "event_id": event_id,
                "source_mass1": mass1,
                "source_mass2": mass2,
                "luminosity_distance": distance
            })
            print(f"  mass1={mass1}, mass2={mass2}, distance={distance}")
        else:
            logging.warning(f"Event ID {event_id} not found in main table for run {run}")
    else:
        print(f"No .png files found for event_id {event_id}")


# Save to a text file
with open("event_ids_with_png_7000.txt", "w") as f:
    f.write("event_id,mass1,mass2,luminosity_distance\n")
    for data_dict in event_data:
        f.write(f"{data_dict['event_id']},{data_dict['source_mass1']},{data_dict['source_mass2']},{data_dict['luminosity_distance']}\n")

# Display summary
print(f"\nSummary: {len(event_ids_with_png)} event_id(s) with .png files")
print(f"Event IDs: {event_ids_with_png}")
print(f"Saved {len(event_data)} event records to event_ids_with_png.txt")