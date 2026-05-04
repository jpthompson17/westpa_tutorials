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

    topology = openmm.app.PDBFile('topology.pdb').getTopology()
    initial_state = westpa.State(file=os.path.abspath('initial_state.xml'))

    simulation = westpa.Simulation(
        datafile='west.h5',
        propagator=propagator(topology),
        pcoord_calculator=pcoord_calculator(topology),
        bin_mapper=bin_mapper(),
        bin_target_counts=5,
        source=westpa.Source(states=[initial_state]),
        sinks=westpa.Sink(indicator=is_bound, label='bound'),
    )

    simulation.initialize(states=[initial_state] * 5)
    simulation.run(3)


def propagator(topology):
    forcefield = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=openmm.app.PME,
        nonbondedCutoff=1 * unit.nanometer,
        constraints=openmm.app.HBonds,
    )
    system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))
    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond
    )

    propagator = westpa.OpenMMPropagator(
        topology=topology,
        system=system,
        integrator=integrator,
        steps=1000,
    )
    propagator.add_reporter(openmm.app.XTCReporter, 'traj.xtc', 500)
    propagator.add_reporter(
        openmm.app.StateDataReporter,
        filename='log.csv',
        report_interval=100,
        options={
            'step': True,
            'potentialEnergy': True,
            'kineticEnergy': True,
            'temperature': True,
        },
    )

    return propagator


def get_pcoord(segment, top):
    traj = mdtraj.load_xml(segment.final_state.file, top=top)
    distances = mdtraj.compute_distances(traj, atom_pairs=[[0, 1]])
    segment.pcoord = distances * 10  # nanometer -> angstrom
    return segment


def pcoord_calculator(topology):
    top = mdtraj.Topology.from_openmm(topology)
    return functools.partial(get_pcoord, top=top)


def bin_mapper():
    return westpa.RectilinearBinMapper(
        boundaries=[
            [0, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, np.inf]
        ],
    )


def is_bound(segment):
    return segment.pcoord[-1, 0] < 2.6


def clean_up_previous_output():
    for file in ['west.h5', 'west.log']:
        if os.path.exists(file):
            os.remove(file)
    if os.path.exists('traj_segs'):
        shutil.rmtree('traj_segs')


if __name__ == '__main__':
    main()
