from astropy.table import QTable


simulation_id = QTable.read("./data/uvex_bns_O5.ecsv")["simulation_id"]
simulation_id = [int(idx) for idx in simulation_id]



data =  QTable.read("./data/4678_with_limmag.ecsv")


observations = data[data["action"] == "observe"].filled()
exposuretimes = observations["duration"]