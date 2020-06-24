import platform
import os

class Constants:
    # Simulation control parameters
    n_nodes_per_layer = 50
    maximize_acceleration=True
    maximize_efficiency=True
    minimize_CoT=True

    randomness_exploration_policy=False

    DQL=False
    genetic_algorithm=True
    simple_run=False

    undulatory_movement = True
    flapping_movement = True

    plot_ideal_structure = False
    plot_structure_matrix = False
    plot_structure_vs_time = False

    print_structural_errors = False
    ajust_trailing_edge=True
    delete=False

    #Genetic Algorithm:
    number_of_generations=3
    deaths_per_generation=1
    population_per_generation=3
    random_genetic_parameter=0.1

    # Operative system:
    windows = False
    linux = False
    if platform.system() == 'Windows':
        windows = True
    if platform.system() == 'Linux':
        linux = True

    # Location of the files:
    windows_path = os.getcwd()
    linux_path= os.getcwd()
    # Structure:
    # mass=0.000001
    zebrafish_density=1.35
    fluid_density=1
    reference_velocity=1
    average_thickness = 0.02046804618550428*2
    perimeter=1.0258500882861628
    surface=1
    volume=0.02
    n_nodes_simplified_structure=n_nodes_per_layer/2+1
    L = 1
    omega_h = 0.04 * L
    s_b = 0.04 * L
    s_t = 0.95 * L
    omega_t = 0.01 * L
    x_translation = 6
    y_translation = 7.5


    # Mesh
    x_length = 30
    dense_x_length = 20
    y_length = 15
    dense_y_length = 12
    n=10








    x_elem=2**n+1
    x_elem_dense=0
    x_elem_non_dense=0
    unities=1
    while x_elem_dense==0:
        if int(x_elem/unities)==0:
            x_elem_dense=int(x_elem/(unities/10))*unities/10
            x_elem_non_dense=x_elem-x_elem_dense
        else:
            unities*=10
    dx_non_dense_1 = ((x_length-dense_x_length)/2)/(int(x_elem_non_dense/2))
    dx_non_dense_2=((x_length-dense_x_length)/2)/(int(x_elem_non_dense/2+1))
    dx = dense_x_length/x_elem_dense
    unities = 1
    y_elem = 2 ** (n - 1) + 1
    y_elem_dense = 0
    while y_elem_dense == 0:
        if int(y_elem / unities) == 0:
            y_elem_dense = int(y_elem/(unities/10))*unities/10
            y_elem_non_dense = y_elem - y_elem_dense
        else:
            unities *= 10
    dy_non_dense_1 = ((y_length - dense_y_length) / 2) / (int(y_elem_non_dense / 2))
    dy_non_dense_2 = ((y_length - dense_y_length) / 2) / (int(y_elem_non_dense / 2+1))
    dy = dense_y_length / y_elem_dense
    epsilon = 1.3  # Controls the space between in the non dense grid zone
    width=0.001


    # Velocity field:
    dt = 0.01
    tf = 1
    range_amplitude = [0, 0.15 * L]
    range_wave_length = [L, 5 * L]
    range_flapping_movement = [0.25, 0.5, 0.75, 1,'lineal',1.25, 1.5, 1.75, 2, 'exp',3,4,5,6,7,8,9]
    frequency = 1  # Hz
    distance_tolerance = 0
    undulatory_variables=['c1', 'c2', 'Wave Length']
    flapping_variables=['Amplitude', 'Tail Movement Node', 'Tail Movement Type']

    #RL:
    gamma=0.99
    eps=0.95
    alpha=0.05
    steps_per_episode=100
    episodes=50
    variation=2
    constant_variations_dql=True
    actions_available = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                actions_available.append([a,b,c])
    troncal_files=['canonical_body_in.dat', 'fort.41', 'input.dat','unstructure_surface_in.dat' , 'xgrid.dat', 'ygrid.dat', 'ibm.3d-1.1', 'Main.py', 'ConstantsPy.py', 'Utils.py','results.txt']
    input_files=['canonical_body_in.dat', 'input.dat', 'xgrid.dat', 'ygrid.dat', 'ibm3d-1.1','probe_in.dat' ]
    simulation_folder_name="Simulation_Folder_"
    simulation_completed_folder_name="completed_"
    best_simulation_name="Best_Simulation"
    input_files_single_simulation = ['canonical_body_in.dat', 'input.dat', 'xgrid.dat', 'ygrid.dat', 'ibm3d-1.1', 'probe_in.dat', 'unstruc_surface_in.dat', 'fort.41']

