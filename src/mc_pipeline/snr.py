from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np

import bilby
from bilby.gw.detector import InterferometerList
from bilby.gw.waveform_generator import WaveformGenerator

Injection = Dict[str, float]


@dataclass(frozen=True)
class GWSettings:
    duration: float = 8.0
    sampling_frequency: float = 2048.0
    waveform_minimum_frequency: float = 20.0
    waveform_reference_frequency: float = 50.0
    approximant: str = "IMRPhenomPv2"
    detectors: Tuple[str, str, str] = ("H1", "L1", "V1")


@dataclass
class GWContext:
    settings: GWSettings
    wfg: WaveformGenerator = field(init=False)
    ifos_psd: InterferometerList = field(init=False)

    def __post_init__(self) -> None:
        self.wfg = WaveformGenerator(
            duration=self.settings.duration,
            sampling_frequency=self.settings.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            waveform_arguments=dict(
                waveform_approximant=self.settings.approximant,
                reference_frequency=self.settings.waveform_reference_frequency,
                minimum_frequency=self.settings.waveform_minimum_frequency,
            ),
        )

        self.ifos_psd = InterferometerList(list(self.settings.detectors))
        self.ifos_psd.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.settings.sampling_frequency,
            duration=self.settings.duration,
            start_time=0,
        )
        for ifo in self.ifos_psd:
            ifo.minimum_frequency = self.settings.waveform_minimum_frequency

    def network_optimal_snr(self, injection: Dict[str, float]) -> float:
        """
        Calculate the signal-to-noise ratio of an injection.

        Args:
            inj (Dict[str, float]): Injection parameters in Bilby format.

        Returns:
            float: SNR that best fits the simulated detector response.
        """
        polarizations = self.wfg.frequency_domain_strain(parameters=dict(injection))
        squared_snr = 0.0
        for ifo in self.ifos_psd:
            h = ifo.get_detector_response(waveform_polarizations=polarizations, parameters=dict(injection))
            squared_snr += float(ifo.optimal_snr_squared(h))
        return float(np.real(np.sqrt(squared_snr)))