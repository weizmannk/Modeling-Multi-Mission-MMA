"""
NMMA Light Curve Analysis Pipeline for Binary Neutron Star Mergers

This script processes gravitational wave event data from UVEX observations,
creates injections, and performs kilonova light curve analysis using the NMMA framework.

Author: [Your Name]
Date: 2025
"""

import logging
import os
import sys
import glob
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
from astropy.table import QTable, Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy.time import Time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class NMMAAnalysisPipeline:
    """
    Pipeline for NMMA kilonova light curve analysis.
    
    This class handles the complete workflow from injection creation to
    light curve analysis for binary neutron star merger events.
    """
    
    def __init__(
        self,
        model_name: str = "HoNa2020",
        prior_file: str = "HoNa2020.prior",
        injection_file: str = "./data/uvex_bns_O5.ecsv",
        eos_file: str = "nmma/example_files/eos/ALF2.dat",
        output_dir: str = "output/HoNa2020_injection",
        m4opt_output_dir: str = "data/uvex-limmag",
        ns_max_mass: float = 3.0
    ):
        """
        Initialize the NMMA analysis pipeline.
        
        Parameters
        ----------
        model_name : str
            Name of the kilonova model to use (default: HoNa2020)
        prior_file : str
            Path to the prior file
        injection_file : str
            Path to the injection table file
        eos_file : str
            Path to the equation of state file
        output_dir : str
            Directory for output files
        m4opt_output_dir : str
            Directory containing M4OPT observation data
        ns_max_mass : float
            Maximum neutron star mass in solar masses (default: 3.0)
        """
        self.model_name = model_name
        self.prior_file = prior_file
        self.injection_file = injection_file
        self.eos_file = eos_file
        self.output_dir = output_dir
        self.m4opt_output_dir = m4opt_output_dir
        self.ns_max_mass = ns_max_mass
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def create_injection(self, generation_seed: int = 42) -> bool:
        """
        Create NMMA injection file for kilonova analysis.
        
        Parameters
        ----------
        generation_seed : int
            Random seed for reproducibility (default: 42)
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        cmd = (
            f"nmma-create-injection "
            f"--prior-file {self.prior_file} "
            f"--injection-file {self.injection_file} "
            f"--eos-file {self.eos_file} "
            f"--binary-type BNS "
            f"--extension json "
            f"-f {self.output_dir} "
            f"--generation-seed {generation_seed} "
            f"--aligned-spin"
        )
        
        logger.info("Creating NMMA injection...")
        return self._run_command(cmd)
    
    def filter_bns_events(self, table: QTable, runs: List[str] = ["O5"]) -> Tuple[QTable, List[int]]:
        """
        Filter binary neutron star events based on mass constraints.
        
        Parameters
        ----------
        table : QTable
            Input table containing GW event parameters
        runs : List[str]
            List of observing runs to include (default: ["O5"])
        
        Returns
        -------
        Tuple[QTable, List[int]]
            Filtered table and list of simulation IDs
        """
        logger.info(f"Filtering BNS events with NS mass < {self.ns_max_mass} Sun Mass...")
        
        # Calculate source-frame masses
        z = z_at_value(cosmo.luminosity_distance, table["distance"] * u.Mpc).to_value()
        zp1 = 1 + z
        source_mass1 = table["mass1"] / zp1
        source_mass2 = table["mass2"] / zp1
        
        # Apply mass filter
        mass_filter = (source_mass1 < self.ns_max_mass) & (source_mass2 < self.ns_max_mass)
        filtered_table = table[mass_filter]
        
        # Extract simulation IDs for specified runs
        run_filter = np.isin(filtered_table["run"], runs)
        simulation_ids = [int(idx) for idx in filtered_table["simulation_id"][run_filter]]
        
        logger.info(f"Filtered {len(table)} events -> {len(filtered_table)} BNS events")
        logger.info(f"Found {len(simulation_ids)} events in runs {runs}")
        
        return filtered_table, simulation_ids
    
    def extract_unique_positions(self, observations: QTable) -> Tuple[np.ndarray, ...]:
        """
        Extract unique observation positions (first visit only).
        
        Parameters
        ----------
        observations : QTable
            Table of observations with coordinates
        
        Returns
        -------
        Tuple[np.ndarray, ...]
            Arrays of RA, DEC, times, exposure times, and detection limits
            for unique positions (first visits only)
        """
        ra = observations["target_coord"].ra.deg
        dec = observations["target_coord"].dec.deg
        
        # Find first occurrence of each unique position
        positions = np.array([(r, d) for r, d in zip(ra, dec)])
        seen_positions = {}
        first_visit_indices = []
        
        for i, pos in enumerate(positions):
            pos_tuple = tuple(pos)
            if pos_tuple not in seen_positions:
                seen_positions[pos_tuple] = i
                first_visit_indices.append(i)
        
        # Filter to first visits only
        first_visits = observations[np.array(first_visit_indices)]
        
        return (
            np.array(first_visits["target_coord"].ra.deg),
            np.array(first_visits["target_coord"].dec.deg),
            np.array(first_visits["start_time"]),
            np.array(first_visits["duration"]),
            np.array(first_visits["limmag"])
        )
    
    def run_light_curve_analysis(
        self,
        event_id: int,
        bandpass: str,
        detection_limits: np.ndarray,
        nlive: int = 2048,
        tmin: float = 0.0,
        tmax: float = 2.0,
        generation_seed: int = 42
    ) -> bool:
        """
        Run NMMA light curve analysis for a single event.
        
        Parameters
        ----------
        event_id : int
            Event simulation ID
        bandpass : str
            Filter bandpass name
        detection_limits : np.ndarray
            Array of detection limit magnitudes
        nlive : int
            Number of live points for nested sampling (default: 2048)
        tmin : float
            Minimum time for analysis in days (default: 0.0)
        tmax : float
            Maximum time for analysis in days (default: 2.0)
        generation_seed : int
            Random seed for reproducibility (default: 42)
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        # Build filter and detection limit strings
        filter_str = ','.join([bandpass] * len(detection_limits))
        detection_limit_str = ','.join([f"{lim:.3f}" for lim in detection_limits])
        
        cmd = (
            f"lightcurve-analysis --model {self.model_name} "
            f"--label {self.model_name}_injection "
            f"--prior {self.prior_file} "
            f"--injection {self.output_dir}_{event_id}.json "
            f"--tmin {tmin} "
            f"--tmax {tmax} "
            f"--dt-inj 0 "
            f"--injection-num {event_id} "
            f"--outdir ModelUvex "
            f"--remove-nondetections "
            f"--nlive {nlive} "
            f"--filters {filter_str} "
            f"--detection-limit {detection_limit_str} "
            f"--plot "
            f"--generation-seed {generation_seed}"
        )
        
        logger.info(f"Running light curve analysis for event {event_id}...")
        return self._run_command(cmd)
    
    def _run_command(self, cmd: str) -> bool:
        """
        Execute a shell command and handle output.
        
        Parameters
        ----------
        cmd : str
            Command to execute
        
        Returns
        -------
        bool
            True if command executed successfully, False otherwise
        """
        try:
            result = subprocess.run(
                cmd.split(),
                check=False,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.stdout:
                sys.stdout.write(result.stdout)
            if result.stderr:
                sys.stderr.write(result.stderr)
            
            if result.returncode != 0:
                logger.error(f"Command failed with return code {result.returncode}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Command timed out after 1 hour")
            return False
        except Exception as exc:
            logger.error(f"Command execution failed: {exc}")
            return False
    
    def process_events(self):
        """
        Main processing loop for all events.
        
        This method:
        1. Creates injections
        2. Filters BNS events
        3. Processes each observation file
        4. Runs light curve analysis
        """
        # Create injections
        if not self.create_injection():
            logger.error("Injection creation failed. Exiting.")
            sys.exit(1)
        
        # Load and filter GW parameters
        table = QTable.read(self.injection_file)
        filtered_table, simulation_ids = self.filter_bns_events(table)
        
        # Process each observation file
        event_files = glob.glob(f"{self.m4opt_output_dir}/*.ecsv")
        logger.info(f"Found {len(event_files)} observation files to process")
        
        for event_file in event_files:
            self._process_single_event(event_file, simulation_ids)
    
    def _process_single_event(self, event_file: str, simulation_ids: List[int]):
        """
        Process a single event observation file.
        
        Parameters
        ----------
        event_file : str
            Path to the event observation file
        simulation_ids : List[int]
            List of valid simulation IDs to process
        """
        data = QTable.read(event_file)
        data_args = data.meta["args"]
        event_id = int(data_args["skymap"].split("/")[-1].split(".")[0])
        
        # Skip if event not in simulation list
        if event_id not in simulation_ids:
            logger.debug(f"Skipping event {event_id} (not in simulation list)")
            return
        
        logger.info(f"Processing event {event_id}")
        
        # Extract observation parameters
        mission = data_args["mission"]
        bandpass = data_args["bandpass"]
        exptime_min = data_args["exptime_min"]
        
        # Filter for actual observations
        observations = data[data["action"] == "observe"].filled()
        
        # Extract unique positions (first visits)
        ra, dec, times, exposure_times, detection_limits = self.extract_unique_positions(observations)
        
        logger.info(
            f"Event {event_id}: {len(observations)} observations "
            f"-> {len(ra)} unique positions"
        )
        
        # Log details of unique positions
        for i in range(len(ra)):
            logger.info(
                f"  Position {i+1}: RA={ra[i]:.6f}°, DEC={dec[i]:.6f}°, "
                f"Time={times[i]}, Duration={exposure_times[i]:.2f}s, "
                f"Limmag={detection_limits[i]:.3f}"
            )
        
        # Run light curve analysis
        self.run_light_curve_analysis(
            event_id=event_id,
            bandpass=bandpass,
            detection_limits=detection_limits
        )


def main():
    """Main entry point for the script."""
    pipeline = NMMAAnalysisPipeline()
    pipeline.process_events()


if __name__ == "__main__":
    main()