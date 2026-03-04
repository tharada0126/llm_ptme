import pyRAPL
import tracemalloc
from abc import ABC

class EnergyProfiler(ABC):
    def __init__(self):
        pyRAPL.setup(devices=None, socket_ids=None)
        self.measure = pyRAPL.Measurement('energy_profile')
        self._started = False
    
    def start(self):
        if self._started:
            raise RuntimeError("Energy profiler has already been started. Call stop() before starting again.")
        self.measure.begin()
        tracemalloc.start()
        self._started = True
    
    def stop(self):
        if not self._started:
            raise RuntimeError("Energy profiler has not been started. Call start() before stop().")
        self.measure.end()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if self.measure.result.pkg is None or self.measure.result.dram is None or self.measure.result.duration is None:
            cpu_energy = None
            dram_energy = None
            time = None
        else:
            cpu_energy = self.measure.result.pkg[0] * 1e-6 # Convert to Joules
            dram_energy = self.measure.result.dram[0] * 1e-6 # Convert to Joules
            time = self.measure.result.duration * 1e-6 # Convert to seconds
        self._started = False
        return {
            "cpu_energy": cpu_energy,
            "dram_energy": dram_energy,
            "time": time,
            "memory_current": current,
            "memory_peak": peak
        }