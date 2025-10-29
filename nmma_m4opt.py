from astropy.table import QTable
from astropy import units as u
import shutil
import subprocess
from astropy.table import Table
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy.time import Time
import sys
import os
import logging
import numpy as np
import glob
import json
import shlex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

model_name = "HoNa2020"
prior_file = "nmma/priors/HoNa2020.prior"
injection_file = "./data/uvex_bns_O5.ecsv"
eos_file = "nmma/example_files/eos/ALF2.dat"
output = "ouptput"
injecjon_file_name = "HoNa2020_injection"

interpolation_type = "tensorflow"

# cmd =  f"nmma-create-injection --prior-file {prior_file} --injection-file {injection_file} --eos-file {eos_file} --binary-type BNS --extension json -f {output}/{injecjon_file_name} --generation-seed 42 --aligned-spin"

cmd = [
    "nmma-create-injection",
    "--prior-file", prior_file,
    "--injection-file", injection_file,
    "--eos-file", eos_file,
    "--binary-type", "BNS",
    "--extension", "json",
    "-f", f"{output}/{injecjon_file_name}",
    "--generation-seed", "42",
    "--aligned-spin"
]

print(' '.join(shlex.quote(arg) for arg in cmd))
try:
    completed = subprocess.run(
        cmd, 
        check=False,
        capture_output=True,
        text=True
    )
except Exception as exc:
    logging.error("Execution failed: %s", exc)
    sys.exit(1)

if completed.stdout:
    sys.stdout.write(completed.stdout)
if completed.stderr:
    sys.stderr.write(completed.stderr)

# GW parameters used to simulate lightcurve
table = QTable.read("./data/uvex_bns_O5.ecsv")
run = ["O5"]
m4opt_ouput_dir = "data/uvex-limmag"

# Filter BNS with NS max of 3 solar mass
ns_max_mass = 3.0
z = z_at_value(cosmo.luminosity_distance, table["distance"] * u.Mpc).to_value()
zp1 = 1 + z
source_mass1 = table["mass1"] / zp1
source_mass2 = table["mass2"] / zp1
table = table[(source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)]

simulation_id = [int(idx) for idx in table["simulation_id"][np.where(np.isin(table["run"], run))]]

for event_file in glob.glob(f"{m4opt_ouput_dir}/*.ecsv"):

    try:
        data = QTable.read(event_file)
        data_args = data.meta["args"]
        event_id_str = data_args["skymap"].split("/")[-1].split(".")[0]
        # IMPORTANT: Convert to int to avoid string/int comparison issues
        event_id = int(event_id_str)
    except ValueError as exc:
        logging.error(f"Failed to parse event ID from {event_file}: {exc}")
    except Exception as exc:
        logging.error(f"Failed to read event file {event_file}: {exc}")
    
    if event_id != 14:
        continue  # Process only event_id 14 for demonstration

    # Check if the event_id is in the simulation_id list
    if int(event_id) in simulation_id:
        mission = data_args["mission"]
        bandpass = data_args["bandpass"]
        exptime_min = data_args["exptime_min"]
        
        observations = data[data["action"] == "observe"].filled()
        
        # Extract RA and DEC
        ra = observations["target_coord"].ra.deg
        dec = observations["target_coord"].dec.deg
        
        # Create tuples (ra, dec) to identify unique positions
        positions = np.array([(r, d) for r, d in zip(ra, dec)])
        
        # Dictionary to store the index of the first occurrence of each position
        seen_positions = {}
        first_visit_indices = []
        
        for i, pos in enumerate(positions):
            pos_tuple = tuple(pos)  # Convert to tuple to use as dictionary key
            if pos_tuple not in seen_positions:
                seen_positions[pos_tuple] = i
                first_visit_indices.append(i)
        
        # Convert to numpy array
        first_visit_indices = np.array(first_visit_indices)
        
        # Filter observations to keep only first visits
        first_visits = observations[first_visit_indices]
        
        # Extract information for these first visits
        exposure_times = np.array(first_visits["duration"])
        detection_limits = np.array(first_visits["limmag"])
        ra_unique = np.array(first_visits["target_coord"].ra.deg)
        dec_unique = np.array(first_visits["target_coord"].dec.deg)
        times_unique = np.array(first_visits["start_time"])
        
        logging.info(f"Event {event_id}: {len(observations)} observations -> {len(first_visits)} unique positions")
        
        # Display information for the first visit of each position
        for i, (r, d, t, exp, lim) in enumerate(zip(ra_unique, dec_unique, times_unique, exposure_times, detection_limits)):
            logging.info(f"  Position {i+1}: RA={r:.6f}, DEC={d:.6f}, Time={t}, Duration={exp:.2f}s, Limmag={lim:.3f}")
        

        # Detection limit and filters strings
        # detection_limit_str = ', '.join([str(lim) for lim in detection_limits])
        # filter_str = ','.join([bandpass for _ in detection_limits])

        # detection_limit_dict = {f"{bandpass}" : lim for lim in detection_limits}
        # detection_limit_json = json.dumps(detection_limit_dict, separators=(",", ":"))

        detection_limit_dict = {f"{bandpass}" : np.max(detection_limits)}
        detection_limit_json = json.dumps(detection_limit_dict, separators=(",", ":"))

        # Build command for analysis
        cmd_analysis = [
            "lightcurve-analysis",
            "--model", model_name,
            "--label", "t",
            "--prior", prior_file,
            "--injection", f"{output}/{injecjon_file_name}.json",
            "--tmin", "0.1",
            "--tmax", "10",
            "--dt-inj", "1",
            "--injection-num", str(event_id),
            "--outdir", output,
            "--nlive", "2048",
            "--filters", bandpass,
            "--detection-limit", detection_limit_json,
            "--plot",
            "--generation-seed", "42",
            #"--sampler", "dynesty",
            "--interpolation-type", interpolation_type
        ]

        print(shlex.join(cmd_analysis))
        logging.info(f"Running analysis for event {event_id}")

        # Run the analysis command
        try:
            completed = subprocess.run(
                cmd_analysis,
                check=False,
                capture_output=True,
                text=True
            )
        except Exception as exc:
            logging.error("Execution failed: %s", exc)
            sys.exit(1)

        if completed.stdout:
            sys.stdout.write(completed.stdout)
        if completed.stderr:
            sys.stderr.write(completed.stderr)



# lightcurve-analysis --model HoNa2020 --label HoNa2020_injection --prior nmma/priors/HoNa2020.prior --injection ouptput/HoNa2020_injection.json --tmin 0. --tmax 2 --dt-inj 1 --injection-num 14 --outdir ouptput/14 --remove-nondetections --nlive 2048 --filters NUV,NUV,NUV --detection-limit 26.226225757951685,25.457474964255017,26.22700152470179 --plot --generation-seed 42
