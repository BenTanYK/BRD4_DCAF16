"""
Perform radial separation of DCAF16-BRD4, 
saving initial configurations for the various US windows.
"""

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import sys
import os
import pandas as pd

"""Read global params from params.in"""

def read_param(param_str):
    """
    Read in a specific parameter and assign the parameter value to a variable
    """
    with open('params.in', 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith(param_str):
                parts = line.split(' = ')
                part = parts[1].strip()

    values = part.split(' ')
    value = values[0].strip()

    # Attempt int conversion
    try:
        value = int(value)
    except:
        value = str(value)

    return value

timestep = read_param('timestep')

"""r0 values to save"""

windows0 = np.arange(0.90, 2.11, 0.05)
windows1 = np.arange(2.2, 3.51, 0.1)
windows = np.append(windows0, windows1)

"""System setup"""

dt = timestep*unit.picoseconds

# Load param and coord files
prmtop = app.AmberPrmtopFile('structures/DCAF16_CYM.prmtop')
inpcrd = app.AmberInpcrdFile('structures/DCAF16_CYM.inpcrd')

system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, hydrogenMass=1.5*unit.amu, constraints=app.HBonds)  
integrator = mm.LangevinMiddleIntegrator(0.0000*unit.kelvin, 1.0000/unit.picosecond, dt)

simulation = app.Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)

# Add reporters to output data
simulation.reporters.append(app.DCDReporter('results/separation/sep_traj.dcd', 1000))
simulation.reporters.append(app.StateDataReporter('results/separation/separation.csv', 1000, step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))

# Minimise energy 
simulation.minimizeEnergy()

"""System heating"""

for i in range(50):
    integrator.setTemperature(6*(i+1)*unit.kelvin)
    simulation.step(1000)

simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
simulation.step(10E3)

"""RMSD Restraints"""

reference_positions = inpcrd.positions

receptor_atoms = [
    atom.index for atom in simulation.topology.atoms()
    if atom.residue.index in range(1, 172) and atom.element.symbol != 'H'
]

ligand_atoms = [
    atom.index for atom in simulation.topology.atoms()
    if atom.residue.index in range(173, 392) and atom.element.symbol != 'H'
]

# Add restraining forces for receptor and ligand rmsd
receptor_rmsd_force = mm.CustomCVForce('0.5*k_rec*rmsd^2')
receptor_rmsd_force.addGlobalParameter('k_rec', 30 * unit.kilocalories_per_mole / unit.angstrom**2)
receptor_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, receptor_atoms))
system.addForce(receptor_rmsd_force)

ligand_rmsd_force = mm.CustomCVForce('0.5*k_lig*rmsd^2')
ligand_rmsd_force.addGlobalParameter('k_lig', 30 * unit.kilocalories_per_mole / unit.angstrom**2)
ligand_rmsd_force.addCollectiveVariable('rmsd', mm.RMSDForce(reference_positions, ligand_atoms))
system.addForce(ligand_rmsd_force)

simulation.context.reinitialize(preserveState=True)

"""Radial separation CV"""

rec_group = [4, 18, 37, 56, 81, 96, 107, 126, 136, 160, 177, 193, 215, 226, 245, 264, 286, 307, 318, 332, 346, 400, 406, 425, 447, 453, 521, 610, 629, 649, 655, 666, 688, 694, 710, 727, 789, 941, 1872, 1899, 1905, 1920, 1941, 1960, 1999, 2026, 2057, 2068, 2084, 2095, 2102, 2112, 2123, 2133, 2140, 2164, 2183, 2197, 2219, 2463]
lig_group = [3032, 3054, 3071, 3088, 3333, 3360, 3366, 3378, 3442, 4051, 4072, 4091, 4112, 4126, 4156, 4162, 4169, 4181, 4193, 4212, 4228, 4247, 4264, 4274, 4289, 4824, 4834, 4856, 4878, 4895, 4905, 4915, 4936, 4946, 4978, 5025, 5055, 5061, 5077, 5089, 5105, 5120, 5130, 5149, 5156, 5175, 5192, 5204, 5890, 5911, 5933, 5953, 5965, 5982, 5997, 6013, 6039]

# Define radial distance as collective variable which we will vary
cv = mm.CustomCentroidBondForce(2, "distance(g1,g2)")
cv.addGroup(np.array(rec_group))
cv.addGroup(np.array(lig_group))

# Specify bond groups
bondGroups = [0, 1]
cv.addBond(bondGroups)

r_0 = 1.15 * unit.nanometers #Set initial separation of 11.5 Angstrom

# Define biasing potential
bias_pot = mm.CustomCVForce('0.5 * k_r * (cv-r_0)^2')
bias_pot.addGlobalParameter('k_r', 100 * unit.kilocalories_per_mole / unit.angstrom**2)
bias_pot.addGlobalParameter('r_0', r_0)

bias_pot.addCollectiveVariable('cv', cv)
system.addForce(bias_pot)

simulation.context.reinitialize(preserveState=True)

"""Boresch restraints"""

# Boresch_residues = [13, 8, 142, 370, 332, 244]

# Define anchor points
idx_a = 193
idx_b = 107
idx_c = 2233
idx_A = 5982
idx_B = 5368
idx_C = 3947

group_a = rec_group
group_b = [idx_b]
group_c = [idx_c]
group_A = lig_group
group_B = [idx_B]
group_C = [idx_C]

# Equilibrium values of Boresch dof
theta_A_0 = 1.29
theta_B_0 = 1.93
phi_A_0 = -2.65
phi_B_0 = 0.81
phi_C_0 = -2.36

k_Boresch = 100 * unit.kilocalories_per_mole / unit.radians**2 #Set global force constant

theta_A_pot = mm.CustomCentroidBondForce(3, '0.5 * k_Boresch * (angle(g1,g2,g3)-theta_A_0)^2')
theta_A_pot.addGlobalParameter('theta_A_0', theta_A_0)
theta_A_pot.addGlobalParameter('k_Boresch', k_Boresch)

# Add the particle groups
theta_A_pot.addGroup([idx_b])
theta_A_pot.addGroup(np.array(rec_group))
theta_A_pot.addGroup(np.array(lig_group))

# Add the centroid angle bond
theta_A_pot.addBond([0, 1, 2])

system.addForce(theta_A_pot)

theta_B_pot = mm.CustomCentroidBondForce(3, '0.5 * k_Boresch * (angle(g1,g2,g3)-theta_B_0)^2')
theta_B_pot.addGlobalParameter('theta_B_0', theta_B_0)
theta_B_pot.addGlobalParameter('k_Boresch', k_Boresch)

# Add the particle groups
theta_B_pot.addGroup(np.array(rec_group))
theta_B_pot.addGroup(np.array(lig_group))
theta_B_pot.addGroup([idx_B])

# Add the centroid angle bond
theta_B_pot.addBond([0, 1, 2])

system.addForce(theta_B_pot)

phi_A_pot = mm.CustomCentroidBondForce(4, '0.5 * k_Boresch * (dihedral(g1,g2,g3,g4)-phi_A_0)^2')
phi_A_pot.addGlobalParameter('phi_A_0', phi_A_0)
phi_A_pot.addGlobalParameter('k_Boresch', k_Boresch)

# Add the particle groups
phi_A_pot.addGroup([idx_c])
phi_A_pot.addGroup([idx_b])
phi_A_pot.addGroup(np.array(rec_group))
phi_A_pot.addGroup(np.array(lig_group))

# Add the centroid angle bond
phi_A_pot.addBond([0, 1, 2, 3])

system.addForce(phi_A_pot)

phi_B_pot = mm.CustomCentroidBondForce(4, '0.5 * k_Boresch * (dihedral(g1,g2,g3,g4)-phi_B_0)^2')
phi_B_pot.addGlobalParameter('phi_B_0', phi_B_0)
phi_B_pot.addGlobalParameter('k_Boresch', k_Boresch)

# Add the particle groups
phi_B_pot.addGroup([idx_b])
phi_B_pot.addGroup(np.array(rec_group))
phi_B_pot.addGroup(np.array(lig_group))
phi_B_pot.addGroup([idx_B])

# Add the centroid angle bond
phi_B_pot.addBond([0, 1, 2, 3])

system.addForce(phi_B_pot)

phi_C_pot = mm.CustomCentroidBondForce(4, '0.5 * k_Boresch * (dihedral(g1,g2,g3,g4)-phi_C_0)^2')
phi_C_pot.addGlobalParameter('phi_C_0', phi_C_0)
phi_C_pot.addGlobalParameter('k_Boresch', k_Boresch)

# Add the particle groups
phi_C_pot.addGroup(np.array(rec_group))
phi_C_pot.addGroup(np.array(lig_group))
phi_C_pot.addGroup([idx_B])
phi_C_pot.addGroup([idx_C])

# Add the centroid angle bond
phi_C_pot.addBond([0, 1, 2, 3])

system.addForce(phi_C_pot)

simulation.context.reinitialize(preserveState=True)

"""Reducing RMSD equilibrium value"""

n_red = 500 # Total number of steps to reduce r_0 over
dr = (0.5/n_red) * unit.nanometers # Bring separation down

for i in range(1, n_red+1):

    print(f'Iteration {i}')
    
    simulation.step(100) # Run short equilibration
    current_cv_value = bias_pot.getCollectiveVariableValues(simulation.context)

    r_0 = r_0 - dr
    simulation.context.setParameter('r_0', r_0)

    if i % 10 == 0:
        print(f"r_0 is {r_0}")
        print(f"The radial separation is {current_cv_value}")

"""Protein-protein unbinding"""

# Total number of steps
total_steps = 250000 

# Number of steps to run between incrementing r_0
increment_steps = 100

r_increment = ((np.max(windows)-np.min(windows)+0.2) / (total_steps // increment_steps)) * unit.nanometers

# During the pulling loop we will save specific configurations corresponding to the windows
window_coords = []
window_index = 0

# SMD pulling loop
for i in range(total_steps//increment_steps):

    if len(window_coords)==len(windows):
        break
    
    simulation.step(increment_steps)
    current_cv_value = bias_pot.getCollectiveVariableValues(simulation.context)

    if (i * increment_steps) % 1000 == 0:
        print("\nIteration " + str(i))
        print("r_0 = ", r_0, ", distance  = ", current_cv_value)
    
    # Increment the location of the CV
    r_0 = r_0 + r_increment
    simulation.context.setParameter('r_0', r_0)

    # Check if we should save this config as a window starting structure
    if (window_index < len(windows) and current_cv_value >= windows[window_index]):
        window_coords.append(simulation.context.getState(getPositions=True).getPositions())

        print(f"Configuration saved for window {window_index}")
        window_index += 1

    # Break condition
    if len(window_coords) == len(windows):
        break

"""Save input configurations for windows"""

for i in range(len(windows)):
    try:
        r0 = np.round(windows[i], 3)

        # Make directory if it doesn't exist
        dirname = f"windows/{r0}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        outfile = open(f'windows/{r0}/{r0}.pdb', 'w')
        app.PDBFile.writeFile(simulation.topology, window_coords[i], outfile)
        outfile.close()
    except:
        print(f'\nError encountered when saving configuration for window {i}')

