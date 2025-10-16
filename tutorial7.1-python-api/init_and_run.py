"""Demonstrates initializing and running a WESTPA + OpenMM simulation.

"""
import functools
import logging
import os
import shutil

import mdtraj
import numpy as np
import openmm.app
import westpa
from openmm import unit

logger = logging.getLogger(__name__)


def main():
    clean_up_previous_output()
    logging.basicConfig(filename='west.log', level=logging.INFO)

    initial_state = westpa.State(ref=os.path.abspath('bstate.xml'))

    simulation = westpa.Simulation(
        datafile='west.h5',
        resampler=resampler(),
        propagator=propagator(),
        pcoord_calculator=pcoord_calculator(),
        source=westpa.Source(states=[initial_state]),
        sink=westpa.Sink(indicator=lambda segment: segment.pcoord[-1, 0] < 2.6),
    )

    simulation.initialize(initial_states=[initial_state] * 5)
    simulation.run(10)


def resampler():
    return westpa.HuberKimResampler(
        bin_mapper=westpa.RectilinearBinMapper(
            boundaries=[
                [0, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, np.inf]
            ],
        ),
        bin_target_counts=5,
    )


def propagator():
    forcefield = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    topology = openmm.app.PDBFile('bstate.pdb').getTopology()
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=openmm.app.PME,
        nonbondedCutoff=1 * unit.nanometer,
        constraints=openmm.app.HBonds,
    )
    system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond
    )

    reports = [
        westpa.OpenMMReport(
            reporter_type=openmm.app.XTCReporter,
            filename='traj.xtc',
            report_interval=500,
        ),
        westpa.OpenMMReport(
            reporter_type=openmm.app.StateDataReporter,
            filename='log.csv',
            report_interval=100,
            options=dict(step=True, potentialEnergy=True, kineticEnergy=True, temperature=True),
        ),
    ]

    return westpa.OpenMMPropagator(
        topology=topology,
        system=system,
        integrator=integrator,
        steps=1000,
        reports=reports,
    )


def calculate_pcoord(segment, topology):
    traj = mdtraj.load_xml(segment.final_state.ref, top=topology)
    distances = mdtraj.compute_distances(traj, atom_pairs=[[0, 1]])
    segment.pcoord = distances * 10  # nanometer -> angstrom
    return segment


def pcoord_calculator():
    topology = mdtraj.load_topology('bstate.pdb')
    return functools.partial(calculate_pcoord, topology=topology)


def clean_up_previous_output():
    for file in ['west.h5', 'west.log']:
        if os.path.exists(file):
            os.remove(file)
    if os.path.exists('traj_segs'):
        shutil.rmtree('traj_segs')


if __name__ == '__main__':
    main()
