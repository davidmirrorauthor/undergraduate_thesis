import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
import datetime
import math
import shutil
from Utils import plotFigure, createTxt, readTxtReturnMatrix, readFilesThatContainAString, guessOperatingSystem, \
    addNewLine, deleteFile, getFilesWithAPartOfTheName
from ConstantsPy import Constants
import random
import tensorflow as tf
import os
import subprocess
import keras.layers
import keras.models
from keras.utils import to_categorical
import concurrent.futures


def plotGeometry():
    perimeter = 0
    L = Constants.L
    s_b = Constants.s_b
    s_t = Constants.s_t
    x1 = np.linspace(start=0, stop=s_b, num=1000)
    x2 = np.linspace(start=s_b, stop=s_t, num=1000)
    x3 = np.linspace(start=s_t, stop=L, num=1000)
    y1 = getImage(x1, first_stage=True)
    y2 = getImage(x2, second_stage=True)
    y3 = getImage(x3, third_stage=True)
    x = np.concatenate((x1, x2, x3))
    y = np.concatenate((y1, y2, y3))
    # previous_x=x[0]
    # previous_y=y[0]
    # for x_element, y_element in zip(x,y):
    #     perimeter+=np.sqrt((x_element-previous_x)**2+(y_element-previous_y)**2)
    #     previous_x=x_element
    #     previous_y=y_element
    # print("The perimeter is: ", perimeter)
    plotFigure(x, y, symmetry=True)
    plt.xlabel('Length (m)')
    plt.ylabel('Width (m)')
    time.sleep(1000)


def changeNameFolders(gen, name, index=-1, number_of_generations=0):
    goToMainRepository()
    files = os.listdir(".")
    folders = []
    for file in files:
        if os.path.isdir(file):
            folders.append(file)
    for folder in folders:
        if Constants.simulation_folder_name in folder or Constants.simulation_completed_folder_name in folder:
            sim = folder.split("_")[-1]
            if index==-1:
                if Constants.simulation_folder_name in folder:
                    new_name = name + "gen_" + str(gen + 1) + "_sim_" + str(sim)
                    if new_name in folders:
                        deleteSpecificFolder(new_name)
                    os.rename(folder, new_name)
            else:
                if Constants.simulation_completed_folder_name in folder:
                    generation=folder.split("_")[2]
                    if int(sim) == index and int(generation) == number_of_generations:
                        os.rename(folder, name+"_gen_"+generation+"_sim_"+sim)
                        break
        else:
            pass


def getImage(x, first_stage=False, second_stage=False, third_stage=False):
    L = Constants.L
    omega_h = Constants.omega_h
    s_b = Constants.s_b
    s_t = Constants.s_t
    omega_t = Constants.omega_t
    y = 0
    if first_stage:
        if not Constants.ajust_trailing_edge:
            y = np.sqrt(2 * omega_h * x - x ** 2)
        else:
            parameter = (omega_h + omega_t) / (np.sqrt(2 * omega_h * Constants.s_b - Constants.s_b ** 2))
            y = np.sqrt(2 * omega_h * x - x ** 2) * parameter
    if second_stage:
        if not Constants.ajust_trailing_edge:
            y = omega_h - (omega_h - omega_t) * (x - s_b) / (s_t - s_b)
        else:
            y = omega_h - (omega_h - omega_t) * (x - s_b) / (s_t - s_b) + omega_t
    if third_stage:
        if not Constants.ajust_trailing_edge:
            y = omega_t * (L - x) / (L - s_t)
        else:
            y = np.sqrt((1 - (x - s_t) ** 2 / (L - s_t) ** 2) * omega_t ** 2) * 2
    return y


def putSameXYCoordinates(structure_matrix, node, x, y, n_nodes_per_layer, symetry=False):
    structure_matrix[node + 2][1] = Constants.x_translation + x
    structure_matrix[node + 2][2] = Constants.y_translation + y
    structure_matrix[node + n_nodes_per_layer + 2][1] = Constants.x_translation + x
    structure_matrix[node + n_nodes_per_layer + 2][2] = Constants.y_translation + y
    structure_matrix[node + 2 * n_nodes_per_layer + 2][1] = Constants.x_translation + x
    structure_matrix[node + 2 * n_nodes_per_layer + 2][2] = Constants.y_translation + y
    return structure_matrix


def deletePreviousFolders(generation):
    if generation == 0:
        folders_name_to_delete = [Constants.simulation_folder_name, Constants.simulation_completed_folder_name, Constants.best_simulation_name]
    else:
        folders_name_to_delete = [Constants.simulation_folder_name]
    path = getPath()
    files = os.listdir(".")
    folders = []
    for name in files:
        if os.path.isdir(name):
            folders.append(name)
    for folder in folders:
        for folder_to_delete in folders_name_to_delete:
            if folder_to_delete in folder:
                shutil.rmtree(os.path.join(path, folder))


def deleteSpecificFolder(folder_name):
    path = getPath()
    files = os.listdir(".")
    folders = []
    for name in files:
        if os.path.isdir(name):
            folders.append(name)
    if folder_name in folders:
        shutil.rmtree(os.path.join(path, folder_name))


def deleteFolders():
    goToMainRepository()
    folders_name_to_delete = [Constants.simulation_completed_folder_name, Constants.simulation_folder_name]
    path = getPath()
    files = os.listdir(".")
    folders = []
    for name in files:
        if os.path.isdir(name):
            folders.append(name)
    for folder_name_to_delete in folders_name_to_delete:
        folders_to_delete = []
        for folder in folders:
            if folder_name_to_delete in folder:
                folders_to_delete.append(folder)
        for folder_to_delete in folders_to_delete:
            if folder_to_delete in folders:
                shutil.rmtree(os.path.join(path, folder_to_delete))


def fillFirstStage(structure_matrix, n_nodes_stage, n_nodes_per_layer):
    structure_matrix = putSameXYCoordinates(structure_matrix, 1, 0, 0, n_nodes_per_layer)
    nodes_without_symmetry = math.trunc(n_nodes_stage / 2)
    x_increments = Constants.s_b / nodes_without_symmetry
    for index in range(nodes_without_symmetry):
        structure_matrix = putSameXYCoordinates(structure_matrix, index + 2, x_increments * (index + 1),
                                                getImage(x_increments * (index + 1), first_stage=True),
                                                n_nodes_per_layer)
        structure_matrix = putSameXYCoordinates(structure_matrix, n_nodes_per_layer - index, x_increments * (index + 1),
                                                -getImage(x_increments * (index + 1), first_stage=True),
                                                n_nodes_per_layer)
    return structure_matrix, nodes_without_symmetry + 1


def fillSecondStage(structure_matrix, n_nodes_stage, n_nodes_per_layer, last_node_first_stage):
    nodes_without_symmetry = int(math.trunc(n_nodes_stage) / 2)
    if nodes_without_symmetry > 0:
        x_increments = (Constants.s_t - Constants.s_b) / (nodes_without_symmetry + 1)
        for index in range(nodes_without_symmetry):
            structure_matrix = putSameXYCoordinates(structure_matrix, index + 1 + last_node_first_stage,
                                                    x_increments * (index + 1) + Constants.s_b,
                                                    getImage(x_increments * (index + 1) + Constants.s_b,
                                                             second_stage=True),
                                                    n_nodes_per_layer)
            structure_matrix = putSameXYCoordinates(structure_matrix,
                                                    n_nodes_per_layer + 1 - last_node_first_stage - index,
                                                    x_increments * (index + 1) + Constants.s_b,
                                                    -getImage(x_increments * (index + 1) + Constants.s_b,
                                                              second_stage=True),
                                                    n_nodes_per_layer)
    return structure_matrix, nodes_without_symmetry + last_node_first_stage


def fillThirdStage(structure_matrix, n_nodes_stage, n_nodes_per_layer, last_node_second_stage):
    tail_node = last_node_second_stage + math.trunc(n_nodes_stage / 2) + 1
    structure_matrix = putSameXYCoordinates(structure_matrix, tail_node, Constants.L, 0, n_nodes_per_layer)
    nodes_without_symmetry = math.trunc(n_nodes_stage / 2)
    x_increments = (Constants.L - Constants.s_t) / nodes_without_symmetry
    for index in range(nodes_without_symmetry):
        structure_matrix = putSameXYCoordinates(structure_matrix, index + 1 + last_node_second_stage,
                                                x_increments * index + Constants.s_t,
                                                getImage(x_increments * index + Constants.s_t, third_stage=True),
                                                n_nodes_per_layer)
        structure_matrix = putSameXYCoordinates(structure_matrix,
                                                n_nodes_per_layer + 1 - last_node_second_stage - index,
                                                x_increments * index + Constants.s_t,
                                                -getImage(x_increments * index + Constants.s_t, third_stage=True),
                                                n_nodes_per_layer)
    return structure_matrix


def runMultipleTimesOneFunctionsVaryingTheAttributes(name_function, list_of_attributes, threads=False):
    final_results = []
    start = time.perf_counter()
    if threads:
        with concurrent.futures.ThreadPoolExecutor()as executor:
            results = executor.map(name_function, list_of_attributes)
    else:
        with concurrent.futures.ProcessPoolExecutor()as executor:
            results = executor.map(name_function, list_of_attributes)
    for result in results:
        # They are printed in the order that where started
        final_results.append(result)
    finish = time.perf_counter()
    print("Multiprocess finished in " + str(round(finish - start, 2)) + " second(s).")
    return (final_results)


def fillNodes(structure_matrix, n_nodes, width, n_nodes_per_layer):
    # There are 3 stages to put the nodes, we are going to do it for one face and then apply the same for the others:
    for index in range(n_nodes):
        structure_matrix[index + 3][0] = int(index + 1)
        if index < n_nodes / 3:
            structure_matrix[index + 3][3] = 0
        elif n_nodes * 2 / 3 > index >= n_nodes / 3:
            structure_matrix[index + 3][3] = width / 2
        else:
            structure_matrix[index + 3][3] = width
    to_share_nodes = n_nodes_per_layer - 6
    n_nodes_first_stage = 3
    n_nodes_second_stage = 0
    n_nodes_third_stage = 3
    if to_share_nodes > 0:
        to_share_pairs_of_nodes = to_share_nodes / 2
        first_share_round = math.trunc(to_share_pairs_of_nodes / 3)
        second_share_round = to_share_pairs_of_nodes % 3
        n_nodes_first_stage += first_share_round * 2
        n_nodes_second_stage += first_share_round * 2
        n_nodes_third_stage += first_share_round * 2
        if second_share_round == 0:
            pass
        elif second_share_round == 1:
            n_nodes_first_stage += 2
        else:
            n_nodes_first_stage += 2
            n_nodes_second_stage += 2
    structure_matrix, last_node_first_stage = fillFirstStage(structure_matrix, n_nodes_first_stage, n_nodes_per_layer)
    structure_matrix, last_node_second_stage = fillSecondStage(structure_matrix, n_nodes_second_stage,
                                                               n_nodes_per_layer,
                                                               last_node_first_stage)
    structure_matrix = fillThirdStage(structure_matrix, n_nodes_third_stage, n_nodes_per_layer, last_node_second_stage)
    # print(n_nodes_first_stage, n_nodes_second_stage, n_nodes_third_stage, n_nodes_first_stage+n_nodes_second_stage+n_nodes_third_stage)
    x = []
    y = []
    z = []
    for index in range(n_nodes):
        x.append(structure_matrix[index + 3][1])
        y.append(structure_matrix[index + 3][2])
        z.append(structure_matrix[index + 3][3])
    # plotFigure(x, y)
    # plotFigure(y, z)
    # plotFigure(x, z)
    return structure_matrix


def getPath():
    return os.getcwd()


def goToMainRepository():
    if Constants.linux:
        current_location = str(subprocess.run('pwd', shell=True, stdout=subprocess.PIPE).stdout)
        folder = current_location.split("/")[-1].split("\\")[0]
        if folder != "zebrafish":
            path_not_desired = str(getPath().split("/")[-1])
            path_desired = "/"+getPath().strip("/" + path_not_desired)
            os.chdir(path_desired)
    else:
        os.chdir(Constants.windows_path)


def fillElements(structure_matrix, n_nodes, n_nodes_per_layer, n_elements):
    for index in range(n_elements):
        structure_matrix[index + n_nodes + 4][0] = int(index + 1)
    element = 1
    for column in range(1, n_nodes_per_layer + 1):
        for subindex in range(1, 5):
            z = column + 1
            if z > n_nodes_per_layer:
                z = 1
            if subindex == 1:
                structure_matrix[element + 3 + n_nodes][1] = int(column)
                structure_matrix[element + 3 + n_nodes][2] = int(z)
                structure_matrix[element + 3 + n_nodes][3] = int(z + n_nodes_per_layer)
            if subindex == 2:
                structure_matrix[element + 3 + n_nodes][1] = int(column)
                structure_matrix[element + 3 + n_nodes][2] = int(z + n_nodes_per_layer)
                structure_matrix[element + 3 + n_nodes][3] = int(column + n_nodes_per_layer)
            if subindex == 3:
                structure_matrix[element + 3 + n_nodes][1] = int(column + n_nodes_per_layer)
                structure_matrix[element + 3 + n_nodes][2] = int(z + n_nodes_per_layer)
                structure_matrix[element + 3 + n_nodes][3] = int(2 * n_nodes_per_layer + z)
            if subindex == 4:
                structure_matrix[element + 3 + n_nodes][1] = int(column + n_nodes_per_layer)
                structure_matrix[element + 3 + n_nodes][2] = int(2 * n_nodes_per_layer + z)
                structure_matrix[element + 3 + n_nodes][3] = int(2 * n_nodes_per_layer + column)
            # print(structure_matrix[element + 3 + n_nodes][0], structure_matrix[element + 3 + n_nodes][1],
            #       structure_matrix[element + 3 + n_nodes][2], structure_matrix[element + 3 + n_nodes][3])
            element += 1

    return structure_matrix


def EndElement(n_rows, structure_matrix):
    for index in range(3):
        structure_matrix[n_rows - 1, index] = float(-10.000)
    return structure_matrix


def getPowerVector(increment, epsilon, length, start, inverse=False):
    power = int(np.trunc(math.log(length / increment, epsilon)))
    vector = [start]
    for index in range(1, power):
        value = (increment * epsilon ** index - increment)
        vector.append(value)
    vector.append(length)
    if inverse:
        new_vector = np.zeros(len(vector))
        for index, value in enumerate(vector):
            new_vector[-index - 1] = length - value
        new_vector[-1] = new_vector[-1] + length
        vector = new_vector
    for index, value in enumerate(vector):
        vector[index] = value + start
    if not inverse:
        vector[-1] = start + length
        vector[0] = start
    else:
        vector = vector[:-2]
    return vector


def checkSimulation(long_wait=False):
    simulation_finished = False
    if Constants.linux:
        output_repeated = False
        previous_output = None
        while not output_repeated:
            p = subprocess.Popen(['ls', '-l', 'out'], stdout=subprocess.PIPE)
            output, err = p.communicate()
            output = output.decode('ascii')
            output = str(output).split(" ")[4]
            print("Simulation process: ", output)
            if output == previous_output:
                print("Simulation finished succesfully.")
                output_repeated = True
                simulation_finished = True
            else:
                previous_output = output
                time.sleep(30)
                if long_wait:
                    time.sleep(300)
    return simulation_finished


def getVector(length, dx, start, end=False):
    vector = [0]
    n_divisions = math.trunc(length / dx)
    for index in range(1, n_divisions):
        vector.append(length / n_divisions * index)
    for index, value in enumerate(vector):
        vector[index] = value + start
    if end:
        vector.append(length + start)
    return np.asarray(vector)


def getAxisMesh(incr, length, dense_length, name, incr_non_dense_1, incr_non_dense_2):
    non_dense_length = length - dense_length
    # non_dense_vector_1 = getPowerVector(incr, Constants.epsilon, non_dense_length / 2, start=0, inverse=True)
    non_dense_vector_1 = getVector(non_dense_length / 2, incr_non_dense_1, 0)
    dense_vector = getVector(dense_length, incr, non_dense_length / 2)
    non_dense_vector_2 = getVector(non_dense_length / 2, incr_non_dense_1, dense_length + non_dense_length / 2,
                                   end=True)
    # non_dense_vector_2 = getPowerVector(incr, Constants.epsilon, non_dense_length / 2,
    #                                     start=non_dense_length / 2 + dense_length, inverse=False)
    vector = np.concatenate((non_dense_vector_1, dense_vector, non_dense_vector_2))
    n_rows = int(len(vector))
    grid = np.zeros((n_rows, 2))
    for index, value in enumerate(vector):
        grid[index][0] = index + 1
        grid[index][1] = value
    createTxt(grid, name)
    return vector


def extractForces(files, path=None):
    a_x_vector = []
    v_x_vector = []
    efficiency_vector = []
    P_thrust = []
    P_def = []
    F_x = []
    CoT_vector = []
    F_shear_vector = []
    F_pressure_vector = []
    for file in files:
        if path:
            forces_matrix = readTxtReturnMatrix(os.path.join(path, file))
        else:
            forces_matrix = readTxtReturnMatrix(file)
        for row in forces_matrix:
            F_shear_vector.append(float(row[3] + row[6]))
            F_pressure_vector.append(float(row[2] + row[5]))
            F_x_i = float(row[3])
            F_x.append(F_x_i)
            a_x_vector.append(
                F_x_i * 0.5 * Constants.fluid_density * Constants.reference_velocity ** 2 * Constants.surface / (
                        Constants.zebrafish_density * Constants.volume))
    v_x = 0
    auxiliar_a_vector = []
    auxiliar_a_vector.append(0)
    for a_x in a_x_vector:
        auxiliar_a_vector.append(a_x)
        v_x += np.trapz(auxiliar_a_vector, dx=Constants.dt)
        v_x_vector.append(v_x)
    for index, v_x in enumerate(v_x_vector):
        P_thrust.append(F_x[index] * v_x)
    # In order to get the deformation power:
    y_velocity_matrix = readTxtReturnMatrix('fort.41')
    n_nodes = Constants.n_nodes_per_layer * 3
    v_def_vector = []
    for instant in range(len(v_x_vector)):
        aux_vector = []
        initial_saving_row = 2 + n_nodes * instant
        final_saving_row = (instant + 1) * n_nodes + 2 + instant
        row = initial_saving_row
        while row <= final_saving_row:
            aux_vector.append(abs(y_velocity_matrix[row][1]))
            row += 1
        v_def_vector.append(np.sum(aux_vector))
    for index in range(len(v_def_vector)):
        P_def_i = float(v_def_vector[index] * F_pressure_vector[index] / n_nodes)
        P_def.append(P_def_i)
        if P_def_i <= 0:
            efficiency_vector.append(1)
            CoT_vector.append(0)
        else:
            efficiency_vector.append(P_thrust[index] / (P_thrust[index] + P_def_i))
            CoT_vector.append(P_def_i / v_x_vector[index])
    return a_x_vector, efficiency_vector, CoT_vector


def createMesh():
    print("Creating the mesh: ")
    dx = Constants.dx
    dy = Constants.dy
    x_length = Constants.x_length
    y_length = Constants.y_length
    dense_x_length = Constants.dense_x_length
    dense_y_length = Constants.dense_y_length
    dy_non_dense_1 = Constants.dy_non_dense_1
    dy_non_dense_2 = Constants.dy_non_dense_2
    dx_non_dense_1 = Constants.dx_non_dense_1
    dx_non_dense_2 = Constants.dx_non_dense_2
    getAxisMesh(dx, x_length, dense_x_length, 'xgrid', dx_non_dense_1, dx_non_dense_2)
    getAxisMesh(dy, y_length, dense_y_length, 'ygrid', dy_non_dense_1, dy_non_dense_2)


def goToNewRepository(folder_name, path):
    if Constants.linux:
        current_location = str(subprocess.run('pwd', shell=True, stdout=subprocess.PIPE).stdout)
        folder = current_location.split("/")[-1].split("\\")[0]
        if folder != folder_name:
            os.chdir(path + '/' + folder_name)
    else:
        os.chdir(path + '/' + folder_name)


def copyFiles(folder_name, path):
    files = os.listdir(".")
    folders = []
    for name in files:
        if os.path.isdir(name):
            folders.append(name)
    if folder_name in folders:
        shutil.rmtree(path + "/" + folder_name)
    os.makedirs(path + "/" + folder_name)
    for file_name in files:
        full_file_name = os.path.join(path, file_name)
        if os.path.isfile(full_file_name) and file_name in Constants.input_files:
            shutil.copy(full_file_name, path + "/" + folder_name)


def getVy(t):
    return 2 * math.pi * Constants.frequency * Constants.amplitude * math.cos(2 * math.pi * Constants.frequency * t)


def postProcess(folder_name=None, main_path=None):
    if folder_name and main_path:
        path = os.path.join(main_path, folder_name)
    else:
        path = getPath()
    try:
        files = readFilesThatContainAString(path, 'drag_lift')
        if files:
            a_x_vector, efficiency_vector, CoT_vector = extractForces(files, path=path)
            if a_x_vector and efficiency_vector and CoT_vector:
                a_x_mean = np.mean(a_x_vector)
                mean_efficiency = np.mean(efficiency_vector)
                mean_CoT = np.mean(CoT_vector)
            else:
                a_x_mean = 0
                mean_efficiency = 0
                mean_CoT = 1000
                print("Error reading the force file. Possibly some divergence happened!")
        else:
            print("File not found.")
    except:
        a_x_mean = 0
        mean_efficiency = 0
        mean_CoT = 1000
        print("Error reading the force file. Possibly some divergence happened!")
    print("Mean acceleration in the x axis: ", a_x_mean)
    print("Mean efficiency: ", mean_efficiency)
    print("Mean CoT: ", mean_CoT)
    return a_x_mean, mean_efficiency, 1 / mean_CoT


def getHardcodedVelocities(t, coor):
    x = coor[0] - Constants.x_translation
    y = coor[1] - Constants.y_translation
    z = coor[2]
    Vx = 0
    Vz = 0
    Vy = 2 * math.pi * Constants.frequency * Constants.amplitude * math.cos(2 * math.pi * Constants.frequency * t) * x
    return Vx, Vy, Vz


def orderByRelevance(iterations, vector):
    best_combinations_index = []
    for index in range(iterations):
        max_value = -1
        for index, value in enumerate(vector):
            if index not in best_combinations_index:
                if value > max_value:
                    max_value = value
                    max_index = index
        best_combinations_index.append(max_index)
    return best_combinations_index


class Agent:
    def __init__(self):
        c1 = 0.05
        c2 = 0.01
        wave_length = 3
        amplitude = 0.09
        flapping_node = int(Constants.n_nodes_simplified_structure * 0.1)
        flapping_movement = 'exp'
        # Simulation
        self.activate_undulatory = Constants.undulatory_movement
        self.activate_flapping = Constants.flapping_movement
        self.activate_DQL = Constants.DQL
        self.activate_genetic_algorithm = Constants.genetic_algorithm
        self.activate_simple_run = Constants.simple_run
        self.activate_maximize_acceleration = Constants.maximize_acceleration
        self.activate_maximize_efficiency = Constants.maximize_efficiency
        self.activate_maximize_CoT_inverse = Constants.minimize_CoT
        self.activate_number_of_generations = Constants.number_of_generations
        self.activate_population_per_generation = Constants.population_per_generation
        self.activate_deaths_per_generation = Constants.deaths_per_generation
        if guessOperatingSystem() == "Linux":
            print("Select using 0/1 the simulation options for the zebrafish:")
            self.activate_undulatory = int(input("Undulatory movement: "))
            self.activate_flapping = int(input("Flapping movement: "))
            if self.activate_flapping or self.activate_undulatory:
                self.activate_maximize_acceleration = int(input("Maximize acceleration: "))
                self.activate_maximize_efficiency = int(input("Maximize efficiency: "))
                self.activate_maximize_CoT_inverse = int(input("Minimize Cost of Transport: "))
                self.activate_genetic_algorithm = int(input("Run a genetic simulation: "))
                if not self.activate_genetic_algorithm:
                    self.activate_simple_run = int(input("Run a single simulation: "))
                    # print("Introduce the value of the parameters: ")
                    if self.activate_simple_run:
                        if self.activate_undulatory:
                            c1 = float(input("Introduce undulatory movement parameter c1: "))
                            c2 = float(input("Introduce undulatory movement parameter c2: "))
                            wave_length = float(input("Introduce undulatory movement parameter wave length: "))
                        if self.activate_flapping:
                            amplitude = float(input("Introduce the amplitude of the flapping movement: "))
                            flapping_node = int(input(
                                "Introduce the node where to start the flapping movement from a total of " + str(
                                    int(Constants.n_nodes_simplified_structure)) + " nodes: "))
                            flapping_movement = input("Introduce the flapping movement: ")
                    elif not (self.activate_flapping and self.activate_undulatory):
                        self.activate_DQL = int(input("Run a Deep Q Learning simulation: "))
                else:
                    good_inputs = False
                    while not good_inputs:
                        self.activate_population_per_generation = int(
                            input("Introduce the number of individuals per generation: "))
                        self.activate_deaths_per_generation = int(
                            input("Introduce the number of deaths per generation: "))
                        if not self.activate_population_per_generation <= self.activate_deaths_per_generation:
                            good_inputs = True
                        else:
                            print("Incorrect inputs!")
                    self.activate_number_of_generations = int(input("Introduce the number of generations: "))
            else:
                print("No movement selected for the simulation")

        self.items_to_maximize = 0
        if self.activate_maximize_acceleration:
            self.items_to_maximize += 1
        if self.activate_maximize_efficiency:
            self.items_to_maximize += 1
        if self.activate_maximize_CoT_inverse:
            self.items_to_maximize += 1

        L = Constants.L
        n_nodes = Constants.n_nodes_per_layer * 3
        # DEFAULT ACTIONS:
        if self.activate_undulatory:
            # Here it goes a connexion to the AI algorithm
            self.range_c1 = Constants.range_amplitude
            self.c1 = c1
            self.range_c2 = [0, 0.15 * L - self.c1]
            self.c2 = c2
            self.range_wave_length = Constants.range_wave_length
            self.wave_length = wave_length
            self.wave_frequency = 1
        if self.activate_flapping:
            self.tail_frequency = 1
            self.range_amplitude = Constants.range_amplitude
            self.tail_amplitude = amplitude
            self.range_flapping_node = [1, n_nodes - 1]
            self.range_tail_movement_node = [1, int(Constants.n_nodes_simplified_structure)]
            self.tail_movement_node = flapping_node
            self.range_tail_movement_type = Constants.range_flapping_movement
            self.tail_movement_type = flapping_movement
        if self.activate_flapping and self.activate_undulatory:
            self.num_var = 6
        else:
            self.num_var = 3

        # Places to keep information
        self.structure_matrix = []
        self.velocity_matrix = []
        self.simplified_nodal_structure = []  # Will consist just on the first layer (z=0) and on a line of nodes with the same y coordinate
        self.simplified_position_vs_time = []  # Dictionary with index t and attributes position vector
        self.distance_between_nodes = []

    def getNodeLocation(self, node):
        x = self.structure_matrix[node + 2][1]
        y = self.structure_matrix[node + 2][2]
        z = self.structure_matrix[node + 2][3]
        coordinates = [x, y, z]
        return coordinates

    def createStructureFile(self, path=None, printing=True):
        """
        The structure should have a matrix of 4 columns:
        1. Blank space.
        2. Number of nodes - Number of elements.
        3. Node number - x - y - z
        4. Blank space.
        5. Number of element - node 1 - node 2 - node 3.
        6. Blank space.
        7. -10.0 - -10.0 - -10.0 (for ending the element)
        8. Repeat process for next element (if there is another one, of course)
        """
        # Fixed Inputs:
        if printing:
            print("Creating the structure file: ")
        width = Constants.width
        n_nodes_per_layer = Constants.n_nodes_per_layer  # Always a pair number, minimum 6 nodes because of the different geometry stages.
        n_nodes_rows = 3  # For 2D do not change
        n_nodes = n_nodes_per_layer * n_nodes_rows
        n_elements = int(4 * n_nodes / 3)  # Valid for a fish geometry in 2D
        n_rows = int(6 + n_nodes + n_elements)
        n_columns = 4
        structure_matrix = np.zeros((n_rows, n_columns))

        # Write the matrix
        structure_matrix[1][0] = n_nodes
        structure_matrix[1][1] = n_elements
        structure_matrix = fillNodes(structure_matrix, n_nodes, width, n_nodes_per_layer)
        structure_matrix = fillElements(structure_matrix, n_nodes, n_nodes_per_layer, n_elements)
        structure_matrix = EndElement(n_rows, structure_matrix)
        if self.activate_undulatory:
            structure_matrix = self.modifyStructureMatrixInitialPosition(structure_matrix)
        self.structure_matrix = structure_matrix
        createTxt(structure_matrix, 'unstruc_surface_in', line_elements=4 + n_nodes, path=path)

    def modifyStructureMatrixInitialPosition(self, matrix):
        x_vector = []
        y_vector = []
        for row, content in enumerate(matrix):
            if 3 + Constants.n_nodes_per_layer * 3 > row > 2:
                # node=row-2
                # n_nodes_per_layer=Constants.n_nodes_per_layer
                # while node>n_nodes_per_layer:
                #     node=n_nodes_per_layer-node
                x = content[1] - Constants.x_translation
                y = content[2] - Constants.y_translation
                y_undulatory, Vy = self.undulatory_equation(x, 0)
                matrix[row][2] = y + Constants.y_translation + y_undulatory
                if 2 < row < Constants.n_nodes_per_layer + 4:
                    x_vector.append(matrix[row][1])
                    y_vector.append(matrix[row][2])
        if Constants.plot_structure_matrix:
            plotFigure(x_vector, y_vector, title='Initial position', x_axis='x coordinates', y_axis='y coordinates')
        return matrix

    def getSimplifiedNodalStructure(self):
        self.simplified_nodal_structure = []
        n_nodes = Constants.n_nodes_per_layer
        n_tail = 0
        x_tail = 0
        for index in range(1, n_nodes + 1):
            x_candidate = self.getNodeLocation(index)[0]
            if x_candidate > x_tail:
                x_tail = x_candidate
                n_tail = index
            else:
                break
        for index in range(1, n_tail + 1):
            self.simplified_nodal_structure.append([self.getNodeLocation(index)[0] - Constants.x_translation, 0])
            self.simplified_position_vs_time[0][index - 1] = [self.getNodeLocation(index)[0] - Constants.x_translation,
                                                              0]
        for index, element in enumerate(self.simplified_nodal_structure):
            x = element[0]
            y = 0
            if self.activate_undulatory:
                y, Vy = self.undulatory_equation(x, 0)
            self.simplified_nodal_structure[index] = [x, y]
            self.simplified_position_vs_time[0][index] = [x, y]
            # Length is being modified!

        for index in range(1, len(self.simplified_nodal_structure)):
            x_difference = self.simplified_nodal_structure[index][0] - self.simplified_nodal_structure[index - 1][0]
            y_difference = self.simplified_nodal_structure[index][1] - self.simplified_nodal_structure[index - 1][1]
            distance = math.sqrt(x_difference ** 2 + y_difference ** 2)
            self.distance_between_nodes.append(distance)

    def enterVariables(self, variables):
        if variables and self.activate_undulatory:
            self.c1 = variables[0]
            self.c2 = variables[1]
            self.wave_length = variables[2]
            if self.activate_flapping:
                self.tail_amplitude = variables[3]
                self.tail_movement_node = variables[4]
                self.tail_movement_type = variables[5]
        elif variables and self.activate_flapping:
            self.tail_amplitude = variables[0]
            self.tail_movement_node = variables[1]
            self.tail_movement_type = variables[2]

    def runSimulationsGeneticAlgorithm(self, population, generation):
        def runSingleGeneticSimulation(inputs):
            results = []
            folder_name = inputs[0]
            maximize_acceleration = inputs[1]
            maximize_efficiency = inputs[2]
            maximize_CoT_inverse = inputs[3]
            number_of_simulation = int(folder_name.split("_")[-1])
            time.sleep(number_of_simulation)
            print("Running ", folder_name)
            goToMainRepository()
            main_path = getPath()
            path = getPath() + str("/") + folder_name
            goToNewRepository(folder_name, main_path)
            # subprocess.run('killall /dev/null', shell=True)
            if Constants.linux:
                try:
                    if getPath() == path:
                        # print("Starting to simulate in ", folder_name)
                        subprocess.run('nohup ./ibm3d* > out &', shell=True)
                    else:
                        print("Not change of folder for simulation ", folder_name)
                        while getPath() != path:
                            time.sleep(1)
                            goToNewRepository(folder_name, main_path)
                            if getPath() == path:
                                print("Finally starting to simulate in ", folder_name)
                                subprocess.run('nohup ./ibm3d* > out &', shell=True)
                    if not getPath() == path:
                        goToNewRepository(folder_name, main_path)
                    if checkSimulation(long_wait=True):
                        a_x_mean, mean_efficiency, mean_CoT_inverse = postProcess(folder_name=folder_name,
                                                                                  main_path=main_path)
                        if maximize_acceleration:
                            results.append(a_x_mean)
                        if maximize_efficiency:
                            results.append(mean_efficiency)
                        if maximize_CoT_inverse:
                            results.append(mean_CoT_inverse)
                except:
                    if maximize_acceleration:
                        results.append(0)
                    if maximize_efficiency:
                        results.append(0)
                    if maximize_CoT_inverse:
                        results.append(0)
            else:
                aux = 0
                if maximize_efficiency:
                    aux += 1
                if maximize_acceleration:
                    aux += 1
                if maximize_CoT_inverse:
                    aux += 1
                for index in range(aux):
                    results.append(np.random.random())
            time.sleep(1)
            return results

        inputs_simulation = []
        deletePreviousFolders(generation)
        for index, individual in enumerate(population):
            folder_name = Constants.simulation_folder_name + str(index + 1)
            inputs_simulation.append(
                [folder_name, self.activate_maximize_acceleration, self.activate_maximize_efficiency,
                 self.activate_maximize_CoT_inverse])
            goToMainRepository()
            self.enterVariables(individual)
            path = getPath()
            copyFiles(folder_name, path)
            goToNewRepository(folder_name, path)
            self.createStructureFile(path=os.path.join(path, folder_name), printing=False)
            self.createVelocityFile(path=os.path.join(path, folder_name), printing=False)
        # results=runSingleGeneticSimulation(inputs_simulation[0])
        all_results = []
        simulation = 0
        number_of_cpu = os.cpu_count()
        if generation == 0:
            print("System has a total of ", number_of_cpu, " CPU's able to simulate simultaneously.")
        if Constants.linux:
            while simulation < len(inputs_simulation):
                inputs_to_process = []
                for index in range(number_of_cpu):
                    if simulation == len(inputs_simulation):
                        break
                    else:
                        inputs_to_process.append(inputs_simulation[simulation])
                        simulation += 1
                results = runMultipleTimesOneFunctionsVaryingTheAttributes(runSingleGeneticSimulation,
                                                                           inputs_to_process,
                                                                           threads=True)
                for result in results:
                    all_results.append(result)
        else:
            for _ in range(len(inputs_simulation)):
                results = []
                if self.activate_maximize_acceleration:
                    results.append(np.random.random())
                if self.activate_maximize_efficiency:
                    results.append(np.random.random())
                if self.activate_maximize_CoT_inverse:
                    results.append(np.random.random())
                all_results.append(results)
        changeNameFolders(generation, Constants.simulation_completed_folder_name)
        if Constants.delete == True and generation<self.activate_number_of_generations-1:
            deleteFolders()
        goToMainRepository()
        return all_results

    def getXCoordinateWithNodeInSimplifiedStructure(self, node):
        x = None
        for index, element in enumerate(self.simplified_nodal_structure, start=1):
            if node == index:
                x = element[0]
        return x

    def calculateFlappingMovement(self, t, node, movement):
        first_node = self.tail_movement_node
        last_node = len(self.simplified_nodal_structure)
        x_node = self.getXCoordinateWithNodeInSimplifiedStructure(node)
        x_first_node = self.getXCoordinateWithNodeInSimplifiedStructure(first_node)
        x_last_node = self.getXCoordinateWithNodeInSimplifiedStructure(last_node)
        if movement == 'exp':
            movement = math.exp(1)
        if movement == 'lineal':
            movement = 1
        ratio = ((x_node - x_first_node) / (x_last_node - x_first_node)) ** movement

        Vy_tail_flapping = 2 * math.pi * self.tail_frequency * self.tail_amplitude * math.cos(
            2 * math.pi * self.tail_frequency * t) * ratio
        return Vy_tail_flapping

    def undulatory_equation(self, x, t):
        y = (x * self.c1 + self.c2 * x ** 2) * math.sin(2 * math.pi * (self.wave_frequency * t + x / self.wave_length))
        Vy = (x * self.c1 + self.c2 * x ** 2) * 2 * math.pi * self.wave_frequency * math.cos(
            2 * math.pi * (self.wave_frequency * t + x / self.wave_length))
        return y, Vy

    def CalculateUndulatoryMovement(self, t, node):
        x = self.simplified_nodal_structure[node - 1][0]
        y, Vy = self.undulatory_equation(x, t)
        return Vy

    def getVelocityVy(self, node, t, step):
        """
        Vz is 0 because is a 2D analysis Vy is the main variable with whom we are going to play Vx is just corrective
        for the displacements. Being the first node fixed, it will displace the nodes to maintain length.
        """
        Vy_undulatory_movement = 0
        Vy_tail_flapping = 0
        y_start = self.simplified_nodal_structure[node - 1][1]
        if self.activate_undulatory:
            Vy_undulatory_movement = self.CalculateUndulatoryMovement(t, node)
        if self.activate_flapping and node > self.tail_movement_node:
            Vy_tail_flapping = self.calculateFlappingMovement(t, node, self.tail_movement_type)
        Vy = Vy_tail_flapping + Vy_undulatory_movement

        if node > 1:
            y = Vy * Constants.dt + y_start
            y_previous_node = self.simplified_nodal_structure[node - 2][1]
            inc_y = y - y_previous_node
            if inc_y > self.distance_between_nodes[node - 2]:
                y = 0.999 * self.distance_between_nodes[node - 2] + y_previous_node
                Vy = (y - y_start) / Constants.dt
            elif inc_y < -self.distance_between_nodes[node - 2]:
                y = -0.999 * self.distance_between_nodes[node - 2] + y_previous_node
                Vy = (y - y_start) / Constants.dt
            self.simplified_nodal_structure[node - 1][1] = y
            self.simplified_position_vs_time[step][node - 1] = [self.simplified_position_vs_time[step - 1][node - 1][0],
                                                                y]
        else:
            self.simplified_position_vs_time[step][node - 1] = [0, 0]
            self.simplified_nodal_structure[node - 1][1] = Vy * Constants.dt
        return Vy

    def fillVelocity(self, index, row, Vx, Vy, Vz):
        n_nodes_per_layer = Constants.n_nodes_per_layer
        self.velocity_matrix[row + index][0] = index
        self.velocity_matrix[row + index][1] = Vx
        self.velocity_matrix[row + index][2] = Vy
        self.velocity_matrix[row + index][3] = Vz
        if index != 1 and index != len(self.simplified_nodal_structure):
            self.velocity_matrix[n_nodes_per_layer + 2 - index + row][0] = n_nodes_per_layer + 2 - index
            self.velocity_matrix[n_nodes_per_layer + 2 - index + row][1] = Vx
            self.velocity_matrix[n_nodes_per_layer + 2 - index + row][2] = Vy
            self.velocity_matrix[n_nodes_per_layer + 2 - index + row][3] = Vz

    def fillVelocityOtherLayers(self, row):
        for multiplier in range(1, 3):
            for index in range(1, Constants.n_nodes_per_layer + 1):
                self.velocity_matrix[row + index + Constants.n_nodes_per_layer * multiplier][
                    0] = index + Constants.n_nodes_per_layer * multiplier
                self.velocity_matrix[row + index + Constants.n_nodes_per_layer * multiplier][1] = \
                    self.velocity_matrix[row + index][1]
                self.velocity_matrix[row + index + Constants.n_nodes_per_layer * multiplier][2] = \
                    self.velocity_matrix[row + index][2]
                self.velocity_matrix[row + index + Constants.n_nodes_per_layer * multiplier][3] = \
                    self.velocity_matrix[row + index][3]

    def getVelocityVx(self, node, step):
        Vx = 0
        if node > 1:
            distance = math.sqrt(
                (self.simplified_nodal_structure[node - 1][0] - self.simplified_nodal_structure[node - 2][0]) ** 2 + (
                        self.simplified_nodal_structure[node - 1][1] - self.simplified_nodal_structure[node - 2][
                    1]) ** 2)
            goal_distance = self.distance_between_nodes[node - 2]
            difference = distance - goal_distance
            x = self.simplified_nodal_structure[node - 1][0]
            goal_x = math.sqrt(goal_distance ** 2 - (
                    self.simplified_nodal_structure[node - 1][1] - self.simplified_nodal_structure[node - 2][
                1]) ** 2) + self.simplified_nodal_structure[node - 2][0]
            inc_x = abs(x - goal_x)
            if difference > Constants.distance_tolerance:
                for index in range(node - 1, len(self.simplified_nodal_structure)):
                    new_x = self.simplified_nodal_structure[index][0] - inc_x
                    self.simplified_nodal_structure[index][0] = new_x
                Vx = -inc_x / Constants.dt
            elif difference < Constants.distance_tolerance:
                for index in range(node - 1, len(self.simplified_nodal_structure)):
                    new_x = self.simplified_nodal_structure[index][0] + inc_x
                    self.simplified_nodal_structure[index][0] = new_x
                    self.simplified_position_vs_time[step][index][0] = new_x
                Vx = inc_x / Constants.dt
        return Vx

    def createVelocityFile(self, path=None, printing=True):
        """
        First, a simplified structure is created taking profit of the symmetry of the problem and its extrapolation easily to 3D.
        Next, the velocity at each instance for each sequence of movements is found.
        Finally, it is generalized to the velocity matrix for each node, taking advantage another time of the symmetry.
        """
        # Initial formulation
        if printing:
            print("Creating the velocity file: ")
        n_nodes = Constants.n_nodes_per_layer * 3
        dt = Constants.dt
        tf = Constants.tf
        n_states = math.trunc(tf / dt)
        n_rows = int((n_nodes + 1) * n_states)
        n_columns = 4
        steps = int(1 / dt) + 1
        time_vector = np.linspace(0, 1, steps)
        for _ in time_vector:
            elements = []
            for element in range(int(2 + (n_nodes / 3 - 2) / 2)):
                elements.append([])
            self.simplified_position_vs_time.append(elements)
        self.velocity_matrix = np.zeros((n_rows, n_columns))
        t = 0
        row = 0
        n_errors = 0
        n_errors_x = 0
        # Creating the simplified nodal structure (for example, an structure of 18 nodes ends up as an structure of 5
        # nodes)
        self.getSimplifiedNodalStructure()
        step = 1
        while t < tf:
            x = []
            y = []
            # print(self.simplified_nodal_structure)
            # Writting first line of this instance:
            self.velocity_matrix[row][0] = dt
            self.velocity_matrix[row][1] = t + dt
            self.velocity_matrix[row][2] = n_nodes
            # Getting all the velocities for each node:
            Vx = []
            Vy = []
            Vz = np.zeros(len(self.simplified_nodal_structure))
            for node in range(1, len(self.simplified_nodal_structure) + 1):
                # Vx, Vy, Vz = GetHardcodedVelocities(t, coordinates)
                Vy.append(self.getVelocityVy(node, t, step))
            for node in range(1, len(self.simplified_nodal_structure) + 1):
                vx = self.getVelocityVx(node, step)
                Vx.append(vx)
            for node in range(1, len(self.simplified_nodal_structure) + 1):
                self.fillVelocity(node, row, Vx[node - 1], Vy[node - 1], Vz[node - 1])
            # Extrapolating this velocities to the rest of nodes. Going from simplified to complete structure.
            self.fillVelocityOtherLayers(row)
            for element in self.simplified_nodal_structure:
                x.append(element[0])
                y.append(element[1])
            step += 1
            t += dt
            row += n_nodes + 1
            if row > n_rows:
                break
            # In order to check that the distance is always the same:
            for index in range(1, len(x)):
                difference = round(math.sqrt((x[index] - x[index - 1]) ** 2 + (y[index] - y[index - 1]) ** 2) -
                                   self.distance_between_nodes[index - 1], 2)
                if difference > 0.01:
                    n_errors += 1
                if x[index] <= x[index - 1]:
                    n_errors_x += 1
        if Constants.print_structural_errors:
            print("Articulation errors: ", n_errors)
            print("Position errors: ", n_errors_x)
        # To check if everything is allright
        createTxt(self.velocity_matrix, 'fort.41', path=path)

    def plotPositionVsTime(self):
        if Constants.plot_structure_vs_time:
            dt = Constants.dt
            steps = int(1 / dt) + 1
            time_vector = np.linspace(0, 1, steps)
            for instant in range(len(self.simplified_position_vs_time)):
                x = []
                y = []
                for element in self.simplified_position_vs_time[instant]:
                    x.append(element[0] + Constants.x_translation)
                    y.append(element[1] + Constants.y_translation)
                # if instant%5==0:
                plotFigure(x, y, equal=True,
                           title='Position vs time at ' + str(round(time_vector[instant], 2)) + ' seconds.',
                           x_axis='x coordinates', y_axis='y coordinates', not_block=True)

    def runSimulation(self):
        if Constants.linux:
            print("Starting to run the simulation:")
            folder_name = 'single_simulation_folder'
            if Constants.linux:
                path = Constants.linux_path
            else:
                path = Constants.windows_path
            files = os.listdir(".")
            folders = []
            for name in files:
                if os.path.isdir(name):
                    folders.append(name)
            if folder_name in folders:
                shutil.rmtree(path + "/" + folder_name)
            os.makedirs(path + "/" + folder_name)
            for file_name in files:
                full_file_name = os.path.join(path, file_name)
                if os.path.isfile(full_file_name) and file_name in Constants.input_files_single_simulation:
                    shutil.copy(full_file_name, path + "/" + folder_name)
            current_location = str(subprocess.run('pwd', shell=True, stdout=subprocess.PIPE).stdout)
            # print(current_location)
            folder = current_location.split("/")[-1].split("\\")[0]
            if folder == "zebrafish":
                os.chdir(path + '/' + folder_name)
                # print(str(subprocess.run('pwd', shell=True, stdout=subprocess.PIPE).stdout))
            subprocess.run('killall /dev/null', shell=True)
            time.sleep(1)
            subprocess.run('nohup ./ibm3d* > out &', shell=True)

    def runSimpleIteration(self, variables=None, previous_results=None, step=0):
        goToMainRepository()
        results = []
        self.enterVariables(variables)
        if not variables:
            if self.activate_undulatory:
                variables = [self.c1, self.c2, self.wave_length]
                if self.activate_flapping:
                    variables = [self.c1, self.c2, self.wave_length, self.tail_amplitude, self.tail_movement_node,
                                 self.tail_movement_type]
            elif self.activate_flapping:
                variables = [self.tail_amplitude, self.tail_movement_node, self.tail_movement_type]
        self.createStructureFile()
        self.createVelocityFile()
        self.plotPositionVsTime()
        if guessOperatingSystem() == "Linux":
            self.runSimulation()
            if checkSimulation():
                print("Proceeding to calculate output values: ")
                a_x_mean, mean_efficiency, mean_CoT_inverse = postProcess()
                if self.activate_maximize_acceleration:
                    results.append(a_x_mean)
                if self.activate_maximize_efficiency:
                    results.append(mean_efficiency)
                if self.activate_maximize_CoT_inverse:
                    results.append(mean_CoT_inverse)
            else:
                print("There's been a problem with the simulation!")
        else:
            if self.activate_maximize_acceleration:
                results.append(np.random.random())
            if self.activate_maximize_efficiency:
                results.append(np.random.random())
            if self.activate_maximize_CoT_inverse:
                results.append(np.random.random())
        if previous_results:
            rewards = 0
            done = 0
            info = None
            for previous_result, result in zip(previous_results, results):
                if previous_result > result:
                    rewards += -1
                elif previous_result < result:
                    rewards += 1
                else:
                    pass
            if step == Constants.steps_per_episode:
                done = 1
            return results, rewards, done, info
        else:
            return variables, results

    def orderBestCombinations(self, vectors):
        final_vector = []
        ranking = len(vectors[0])
        points = np.zeros(self.activate_population_per_generation)
        for vector in vectors:
            for index, element in enumerate(vector):
                points[element] += ranking - index
        while len(final_vector) < ranking:
            best_index = -1
            best_punctuation = 0
            for index, point in enumerate(points):
                if point > best_punctuation:
                    best_index = index
                    best_punctuation = point
            final_vector.append(best_index)
            points[best_index] = 0
        return final_vector

    def runMultipleTimesOneFunctionsVaryingTheAttributes(self, name_function, list_of_attributes, thread=False):
        start = time.perf_counter()
        if thread:
            with concurrent.futures.ThreadPoolExecutor()as executor:
                results = executor.map(name_function, list_of_attributes)
        else:
            with concurrent.futures.ProcessPoolExecutor()as executor:
                results = executor.map(name_function, list_of_attributes)
        finish = time.perf_counter()
        print("Multiprocess finished in " + str(round(finish - start, 2)) + " second(s)")
        return results

    def runGeneticAlgorithm(self):
        generation = 0
        population = []
        for index in range(self.activate_population_per_generation):
            proposal = []
            if self.activate_undulatory:
                c1 = random.uniform(self.range_c1[0], self.range_c1[1])
                self.range_c2 = [0, 0.15 * Constants.L - c1]
                c2 = random.uniform(self.range_c2[0], self.range_c2[1])
                wave_length = random.uniform(self.range_wave_length[0], self.range_wave_length[1])
                proposal = [c1, c2, wave_length]
            if self.activate_flapping:
                tail_movement_node = random.randint(self.range_tail_movement_node[0], self.range_tail_movement_node[1])
                tail_movement_type = random.choice(Constants.range_flapping_movement)
                if self.activate_undulatory:
                    self.range_amplitude = [0, 0.15 * Constants.L - c1 - c2]
                    tail_amplitude = random.uniform(self.range_amplitude[0], self.range_amplitude[1])
                    proposal = [c1, c2, wave_length, tail_amplitude, tail_movement_node, tail_movement_type]
                else:
                    self.range_amplitude = [0, 0.15 * Constants.L]
                    tail_amplitude = random.uniform(self.range_amplitude[0], self.range_amplitude[1])
                    proposal = [tail_amplitude, tail_movement_node, tail_movement_type]
            population.append(proposal)
        while generation < self.activate_number_of_generations:
            print("Creating generation number ", generation + 1)
            maximize_1 = []
            maximize_2 = []
            maximize_3 = []
            all_results = self.runSimulationsGeneticAlgorithm(population, generation)
            # for index, individual in enumerate(population):
            #     _,results = self.runSimpleIteration(variables=individual)
            for results in all_results:
                for index, result in enumerate(results):
                    if index == 0:
                        maximize_1.append(result)
                    if index == 1:
                        maximize_2.append(result)
                    if index == 2:
                        maximize_3.append(result)
            iterations = self.activate_population_per_generation - self.activate_deaths_per_generation
            if maximize_1:
                best_combinations_index_1 = orderByRelevance(iterations, maximize_1)
            if maximize_2:
                best_combinations_index_2 = orderByRelevance(iterations, maximize_2)
            if maximize_3:
                best_combinations_index_3 = orderByRelevance(iterations, maximize_3)
            maximize = []
            if maximize_1:
                maximize.append(best_combinations_index_1)
            if maximize_2:
                maximize.append(best_combinations_index_2)
            if maximize_3:
                maximize.append(best_combinations_index_3)
            best_combinations_index = self.orderBestCombinations(maximize)
            generation += 1
            if self.activate_undulatory:
                values_c1 = []
                values_c2 = []
                values_wave_length = []
                for index in best_combinations_index:
                    values_c1.append(population[index][0])
                    values_c2.append(population[index][1])
                    values_wave_length.append(population[index][2])
                if self.activate_flapping:
                    values_amplitude = []
                    values_movement_node = []
                    values_movement_type = []
                    for index in best_combinations_index:
                        values_amplitude.append(population[index][3])
                        values_movement_node.append(population[index][4])
                        values_movement_type.append(population[index][5])
            else:
                values_amplitude = []
                values_movement_node = []
                values_movement_type = []
                for index in best_combinations_index:
                    values_amplitude.append(population[index][0])
                    values_movement_node.append(population[index][1])
                    values_movement_type.append(population[index][2])
            if generation < self.activate_number_of_generations:
                for index, _ in enumerate(population):
                    if index not in best_combinations_index:
                        if Constants.randomness_exploration_policy:
                            if Constants.random_genetic_parameter <= np.random.random():
                                if self.activate_undulatory:
                                    c1 = random.uniform(min(values_c1), max(values_c1))
                                    c2 = random.uniform(min(values_c2), max(values_c2))
                                    wave_length = random.uniform(min(values_wave_length), max(values_wave_length))
                                    proposal = [c1, c2, wave_length]
                                    if self.activate_flapping:
                                        amplitude = random.uniform(min(values_amplitude), max(values_amplitude))
                                        movement_node = random.randint(min(values_movement_node), max(values_movement_node))
                                        movement_type = random.choice(values_movement_type)
                                        proposal = [c1, c2, wave_length, amplitude, movement_node, movement_type]
                                else:
                                    amplitude = random.uniform(min(values_amplitude), max(values_amplitude))
                                    movement_node = random.randint(min(values_movement_node), max(values_movement_node))
                                    movement_type = random.choice(values_movement_type)
                                    proposal = [amplitude, movement_node, movement_type]
                            else:
                                if self.activate_undulatory:
                                    c1 = random.uniform(min(self.range_c1), max(self.range_c1))
                                    c2 = random.uniform(min(self.range_c2), max(self.range_c2))
                                    wave_length = random.uniform(min(self.range_wave_length), max(self.range_wave_length))
                                    proposal = [c1, c2, wave_length]
                                    if self.activate_flapping:
                                        amplitude = random.uniform(min(self.range_amplitude), max(self.range_amplitude))
                                        movement_node = random.randint(min(self.range_tail_movement_node), max(self.range_tail_movement_node))
                                        movement_type = random.choice(values_movement_type)
                                        proposal = [c1, c2, wave_length, amplitude, movement_node, movement_type]
                                else:
                                    amplitude = random.uniform(min(values_amplitude), max(values_amplitude))
                                    movement_node = random.randint(min(values_movement_node), max(values_movement_node))
                                    movement_type = random.choice(self.range_tail_movement_type)
                                    proposal = [amplitude, movement_node, movement_type]
                        else:
                            if self.activate_undulatory:
                                c1 = random.uniform(min(values_c1), max(values_c1))
                                c2 = random.uniform(min(values_c2), max(values_c2))
                                wave_length = random.uniform(min(values_wave_length), max(values_wave_length))
                                proposal = [c1, c2, wave_length]
                                if self.activate_flapping:
                                    amplitude = random.uniform(min(values_amplitude), max(values_amplitude))
                                    movement_node = random.randint(min(values_movement_node), max(values_movement_node))
                                    movement_type = random.choice(values_movement_type)
                                    proposal = [c1, c2, wave_length, amplitude, movement_node, movement_type]
                            else:
                                amplitude = random.uniform(min(values_amplitude), max(values_amplitude))
                                movement_node = random.randint(min(values_movement_node), max(values_movement_node))
                                movement_type = random.choice(values_movement_type)
                                proposal = [amplitude, movement_node, movement_type]
                        population[index] = proposal
            else:
                best_combination = population[best_combinations_index[0]]
                best_maximize = []
                if maximize_1:
                    best_maximize_1 = round(maximize_1[int(best_combinations_index[0])], 4)
                    best_maximize.append(best_maximize_1)
                if maximize_2:
                    best_maximize_2 = round(maximize_2[int(best_combinations_index[0])], 4)
                    best_maximize.append(best_maximize_2)
                if maximize_3:
                    best_maximize_3 = round(maximize_3[int(best_combinations_index[0])], 4)
                    best_maximize.append(best_maximize_3)
                if self.activate_undulatory:
                    best_c1 = best_combination[0]
                    best_c2 = best_combination[1]
                    best_wave_length = best_combination[2]
                    results = [best_c1, best_c2, best_wave_length]
                    if self.activate_flapping:
                        best_amplitude = best_combination[3]
                        best_movement_node = best_combination[4]
                        best_movement_type = best_combination[5]
                        results = [best_c1, best_c2, best_wave_length, best_amplitude, best_movement_node,
                                   best_movement_type]
                else:
                    best_amplitude = best_combination[0]
                    best_movement_node = best_combination[1]
                    best_movement_type = best_combination[2]
                    results = [best_amplitude, best_movement_node, best_movement_type]
                best_maximize = [round(element, 4) for element in best_maximize]
                auxiliar_vector = []
                for result in results:
                    try:
                        result = round(result, 4)
                        auxiliar_vector.append(result)
                    except:
                        auxiliar_vector.append(results)
                results = auxiliar_vector
                print("The best combination was found int the simulation ", str(best_combinations_index[0] + 1),
                      " which had as parameters:")
                changeNameFolders(generation, Constants.best_simulation_name, index=best_combinations_index[0]+1,
                                  number_of_generations=self.activate_number_of_generations)
                if Constants.delete:
                    deleteFolders()
                if self.activate_undulatory:
                    for index, element in enumerate(Constants.undulatory_variables):
                        print(element, ": ", str(results[index]))
                    if self.activate_flapping:
                        for index, element in enumerate(Constants.flapping_variables, start=3):
                            print(element, ": ", str(results[index]))
                else:
                    for index, element in enumerate(Constants.flapping_variables):
                        print(element, ": ", str(results[index]))
                print("The results obtained are:")
                if self.activate_maximize_acceleration:
                    print("Acceleration in the x direction: ", best_maximize_1)
                    if self.activate_maximize_efficiency:
                        print("Efficiency: ", best_maximize_2)
                        if self.activate_maximize_CoT_inverse:
                            print("Inverse of the CoT: ", best_maximize_3)
                    elif self.activate_maximize_CoT_inverse:
                        print("Inverse of the CoT: ", best_maximize_2)
                elif self.activate_maximize_efficiency:
                    print("Efficiency: ", best_maximize_1)
                    if self.activate_maximize_CoT_inverse:
                        print("Inverse of the CoT: ", best_maximize_2)
                else:
                    print("Inverse of the CoT: ", best_maximize_1)
                return results, best_maximize

    def runDQL(self):
        variables = []
        results = []
        input_shape = [self.items_to_maximize]
        n_outputs = 27
        model = keras.models.Sequential([
            keras.layers.Dense(25, activation="elu", input_shape=input_shape),
            keras.layers.Dense(25, activation="elu"),
            keras.layers.Dense(n_outputs)
        ])
        replay_buffer = deque(maxlen=200)
        batch_size = 32
        discount_factor = 0.95
        optimizer = keras.optimizers.Adam(lr=0.001)
        loss_fn = keras.losses.mean_squared_error

        def sample_experiences(batch_size):
            indices = np.random.randint(len(replay_buffer), size=batch_size)
            batch = [replay_buffer[index] for index in indices]
            states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch])
                                                            for field_index in range(5)]
            return states, actions, rewards, next_states, dones

        def play_one_step(state, epsilon, step, previous_parameters):
            action, parameters = epsilon_greedy_policy(state, previous_parameters, epsilon, step)
            next_state, reward, done, info = self.runSimpleIteration(previous_results=state, step=step,
                                                                     variables=parameters)
            replay_buffer.append((state, action, reward, next_state, done))
            return next_state, reward, done, info, parameters

        def modifyVariables(variables, modify_vector, step):
            new_variables = []
            for variable, modify in zip(variables, modify_vector):
                if not Constants.constant_variations_dql:
                    if modify == 0:
                        pass
                    elif modify == 1:

                        variable = variable * (1 + Constants.variation / step)
                    else:
                        variable = variable * (1 - Constants.variation / step)
                else:
                    if modify == 0:
                        pass
                    elif modify == 1:

                        variable = variable * (1 + Constants.variation)
                    else:
                        variable = variable * (1 - Constants.variation)
                new_variables.append(variable)
            return new_variables

        def epsilon_greedy_policy(state, previous_parameters, step, epsilon=0):
            if np.random.rand() < epsilon:
                random_options = [np.random.randint(0, 2), np.random.randint(0, 2), np.random.randint(0, 2)]
                for index, element in enumerate(Constants.actions_available):
                    if random_options == element:
                        action = index
                parameters = modifyVariables(previous_parameters, random_options, step)
                return action, parameters
            else:
                Q_values = model.predict(state, batch_size=32, verbose=0)
                action = np.argmax(Q_values)
                parameters = modifyVariables(previous_parameters, Constants.actions_available[action], step)
                return action, parameters

        def training_step(batch_size):
            experiences = sample_experiences(batch_size)
            states, actions, rewards, next_states, dones = experiences
            next_Q_values = model.predict(next_states)
            max_next_Q_values = np.max(next_Q_values, axis=1)
            target_Q_values = (rewards + (1 - dones) * discount_factor * max_next_Q_values)
            mask = to_categorical(actions, num_classes=n_outputs)
            with tf.GradientTape() as tape:
                all_Q_values = model(tf.convert_to_tensor(np.float32(states)))
                Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
                loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        for episode in range(Constants.episodes):
            previous_parameters = []
            if self.activate_undulatory:
                c1 = random.uniform(self.range_c1[0], self.range_c1[1])
                self.range_c2 = [0, 0.15 * Constants.L - c1]
                c2 = random.uniform(self.range_c2[0], self.range_c2[1])
                wave_length = random.uniform(self.range_wave_length[0], self.range_wave_length[1])
                previous_parameters = [c1, c2, wave_length]
            if self.activate_flapping:
                tail_movement_node = random.uniform(self.range_tail_movement_node[0], self.range_tail_movement_node[1])
                tail_movement_type = random.choice(Constants.range_flapping_movement)
                if self.activate_undulatory:
                    self.range_amplitude = [0, 0.15 * Constants.L - c1 - c2]
                    tail_amplitude = random.uniform(self.range_amplitude[0], self.range_amplitude[1])
                    previous_parameters = [c1, c2, wave_length, tail_amplitude, tail_movement_node, tail_movement_type]
                else:
                    self.range_amplitude = [0, 0.15 * Constants.L]
                    tail_amplitude = random.uniform(self.range_amplitude[0], self.range_amplitude[1])
                    previous_parameters = [tail_amplitude, tail_movement_node, tail_movement_type]
            _, obs = self.runSimpleIteration(variables=previous_parameters)
            variables = []
            results = []
            for step in range(Constants.steps_per_episode):
                epsilon = max(1 - episode / 50, 0.01)
                obs, reward, done, info, parameters = play_one_step(obs, epsilon, step, previous_parameters)
                variables.append(parameters)
                results.append(obs)
                if done:
                    break
                previous_parameters = parameters
            if episode > 10:
                training_step(batch_size)
        return variables, results

    def save_results(self, variables, results, time, individuals=None, deaths=None, generations=None):
        if variables and results:
            maximization_index = 0
            variables_index = 0
            experiment_text = ""
            if self.activate_genetic_algorithm:
                experiment_text += "Genetic Algorithm: "
            if self.activate_simple_run:
                experiment_text += "Single run: "
            if self.activate_DQL:
                experiment_text += "DQL Algorithm: "
            if self.activate_maximize_acceleration:
                experiment_text += str(results[maximization_index]) + " "
                maximization_index += 1
            else:
                experiment_text += "--- "
            if self.activate_maximize_efficiency:
                experiment_text += str(results[maximization_index]) + " "
                maximization_index += 1
            else:
                experiment_text += "--- "
            if self.activate_maximize_CoT_inverse:
                experiment_text += str(results[maximization_index]) + " "
            else:
                experiment_text += "--- "
            if self.activate_undulatory:
                variables_index += 3
                experiment_text += str(variables[0]) + " "
                experiment_text += str(variables[1]) + " "
                experiment_text += str(variables[2]) + " "
            else:
                for _ in range(3):
                    experiment_text += "--- "
            if self.activate_flapping:
                experiment_text += str(variables[variables_index]) + " "
                experiment_text += str(variables[variables_index + 1]) + " "
                experiment_text += str(variables[variables_index + 2]) + " "
            else:
                for _ in range(3):
                    experiment_text += "--- "
            experiment_text += str(time) + " "
            experiment_text += str(datetime.datetime.now())
            if individuals and generations and deaths:
                experiment_text += " "
                experiment_text += str(individuals)
                experiment_text += " "
                experiment_text += str(deaths)
                experiment_text += " "
                experiment_text += str(generations)
            addNewLine('results.txt', experiment_text)
        else:
            pass


if __name__ == '__main__':
    variables = []
    results = []
    createMesh()
    if Constants.plot_ideal_structure:
        plotGeometry()
    agent = Agent()
    if agent.activate_maximize_acceleration or agent.activate_maximize_efficiency or agent.activate_maximize_CoT_inverse:
        start = time.perf_counter()
        if agent.activate_simple_run:
            variables, results = agent.runSimpleIteration()
            print("The results are: ", results)
        elif agent.activate_genetic_algorithm:
            variables, results = agent.runGeneticAlgorithm()
        elif agent.activate_DQL:
            variables, results = agent.runDQL()
        finish = time.perf_counter()
        if agent.activate_genetic_algorithm:
            agent.save_results(variables, results, round(finish - start, 3),
                               individuals=agent.activate_population_per_generation,
                               deaths=agent.activate_deaths_per_generation,
                               generations=agent.activate_number_of_generations)
        else:
            agent.save_results(variables, results, round(finish - start, 3))
        print("Simulation finished in " + str(round(finish - start, 3)) + " seconds(s)")
    else:
        print("No parameters to maximize.")
