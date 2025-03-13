"""
NPT MD simulation pipeline

INPUT: cleaned protein structure 
(Hydrogens + heavy atoms + missing residues added)

- Read in structure
- Solvate structure using OPC water model
- Paramaterise system using ff19SB forcefield with Li/Merz ion parameters
- Parameterise the DCAF16 Zn ion with ZAFF treatment
- Perform energy minimisation
- Slowly ramp up temperature to our target temperature of 300 K
- Perform 1 ns NVT equilibration
- Perform 50 ns NPT MD simulation (implement Monte Carlo barostat)

OUTPUT: .pdb files containing output trajectories + .csv files containing
energy + temperature date

References:
https://openmm.github.io/openmm-cookbook/latest/notebooks/tutorials/protein_in_water.html
http://www.mdtutorials.com/gmx/lysozyme/index.html

"""
from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np

print('Imports have been completed sucessfully...')

dt = 2.0000*femtoseconds #Set global 2 fs timestep

print('Timestep set to 2 fs')

print('Loading in the system...')

# Load param and coord files
prmtop = AmberPrmtopFile('DCAF16_ZAFF.prmtop')
inpcrd = AmberInpcrdFile('DCAF16_ZAFF.inpcrd')

print('Amber input files read in')

"""Generate system"""
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.5*nanometer, constraints=HBonds)  

"""Use Langevin integrator with 1/ps friction coefficient, with initial temp = 0 K"""
integrator = LangevinMiddleIntegrator(0.0000*kelvin, 1.0000/picosecond, dt)

print('System and integrator initialised')

"""Set up simulation""" 
simulation = Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)

print('Simulation initialised')

simulation.minimizeEnergy() #Minimise energy

print('Energy minimisation complete')

"""Initialise print reporter for whole simulation"""

print('\nStarting simulation...\n')

totalSteps = 2.75E7
reportInterval = int(totalSteps/1000) 

simulation.reporters.append(DCDReporter('trajectory.dcd', reportInterval))

simulation.reporters.append(StateDataReporter('system.csv', reportInterval, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
    temperature=True, volume=True, density=True, progress=True, elapsedTime=False, totalSteps=totalSteps))

# simulation.reporters.append(StateDataReporter(stdout, reportInterval, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
#     temperature=True, volume=True, density=True, progress=True, elapsedTime=False, totalSteps=totalSteps))

# START SIMULATION

simulation.step(25000) #Perform 100 ps test simulation at 0 K

"""
Ramp up temperature slowly from 0 K to 300 K. 
Increase temperature by 1 K every 1 ps
"""
print('\nStarting heating...\n')

for temp in range(1, 301):

    integrator.setTemperature(temp * kelvin) #set new temperature
    simulation.step(500) #2 ps simulation at given temp

"""
NVT equilibration at 300 K
"""
integrator.setTemperature(300.0000 * kelvin) #set to final temperature

print('\nStarting NVT equilibration at 300 K\n')

simulation.step(250000) #Perform equilibration at 300 K

"""
NPT simulation 
"""
print('\nStarting NPT simulation at 300 K, 1 bar\n')

system.addForce(MonteCarloBarostat(1*bar, 300.0000*kelvin))
simulation.context.reinitialize(preserveState=True)
simulation.step(2.5E7) #50 ns 

# Save the final frame
outfile = 'final_frame_equil.pdb'
PDBFile.writeFile(simulation.topology.coords, outfile)

print("\nSimulation complete!")

    


