import matplotlib.pyplot as plt
import numpy as np
import pylab
from ConstantsPy import Constants
from matplotlib.animation import FuncAnimation
import os
import platform
import time
import concurrent.futures

def guessOperatingSystem():
    return platform.system()

def readFilesThatContainAString(path, content):
    files = []
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and content in i:
            files.append(i)
    return files

def isANumber(candidate):
    return any(map(str.isdigit, candidate))

def plotFigure(x, y, symmetry=False, equal=True, title=None, x_axis=None, y_axis=None, not_block=False):
    plt.style.use('seaborn')
    plt.clf()
    if title:
        plt.title(title, fontsize=20)
    if x_axis:
        plt.xlabel(x_axis)
    if y_axis:
        plt.ylabel(y_axis)
    if equal:
        plt.axis('equal')
    plt.plot(x, y, 'b')
    if symmetry:
        plt.plot(x, -y, 'b')
    if not_block:
        plt.show(block=False)
        plt.pause(0.0001)
        plt.clf()
    else:
        plt.show()

def readTxtReturnMatrix(file_name):
    with open(file_name, 'r') as reader:
        matrix = []
        row = []
        content = reader.read()
        content_vector = content.strip().split(" ")
        for index, content in enumerate(content_vector):
            content.strip()
            end_line = False
            if str(content) == "":
                pass
            else:
                if "\n" in content:
                    end_line = True
                    content = content.split("\n")[0]
                if isANumber(content):
                    row.append(float(content))
                    if end_line:
                        matrix.append(row)
                        row = []
    return matrix

def createTxt(matrix, name,line_elements=None, path=None):
    if not path:
        path=os.getcwd()
    matrix=np.matrix(matrix)
    if name=='fort.41':
        name = str(name)
    else:
        name = str(name) + '.dat'
    with open(os.path.join(path, name), 'wb') as f:
        for line in matrix:
            if line[0,0]==0:
                line=np.matrix([[]])
            elif int(-10) == int(line[0,0]):
                line=np.matrix([[-10, -10, -10]])
            if line!=np.matrix([[]]):
                np.savetxt(f, line,fmt='%.8f')
            else:
                np.savetxt(f, line)

    with open(os.path.join(path,name), "r") as f:
        lines = f.readlines()
    with open(os.path.join(path, name), "w") as f:
        for position, line in enumerate(lines):
            if line!="\n":
                elements=line.split(" ")
                if name=="fort.41" and float(elements[0])==Constants.dt:
                    number_of_decimals_dt=0
                    dt=Constants.dt
                    while dt<1:
                        dt=dt*10
                        number_of_decimals_dt+=1
                    piece1=round(float(elements[0]),number_of_decimals_dt)
                    piece2=round(float(elements[1]),2)
                    piece3=int(float(elements[2]))
                    line=str(piece1)+" "+str(piece2)+" "+str(piece3)+"\n"
                elif line_elements and name=='unstruc_surface_in.dat' and position>=line_elements-1:
                    for index, element in enumerate(elements):
                        if index==len(elements)-1:
                            element=element.strip("\n")
                        number_float=float(element)
                        number=int(number_float)
                        if index==0:
                            line=str(number)+" "
                        elif index!=len(elements)-1:
                            line+=str(number)+" "
                        else:
                            line+=str(number)+"\n"
                else:
                    number_float = float(elements[0])
                    number = int(number_float)
                    line = str(number) + " "
                    for index, element in enumerate(elements):
                        # if float(element).is_integer():
                        #     element=int(float(element))
                        if index!=0:
                            if name=='unstruc_surface_in.dat' and position==1 and index==1:
                                number=float(element)
                                line+=str(int(number))+"\n"
                                break
                            elif index!=len(elements)-1:
                                line+=str(element)
                                line+=" "
                            else:
                                line+=str(element)
            f.write(line)

def addNewLine( file_name,new_line, saving_path=None ):
    if saving_path:
        file_name = os.path.join(saving_path, file_name)
    with open(file_name, "a") as file:
        file.write(new_line + "\n")

def deleteFile(path, file_name):
    os.unlink(os.path.join(path, file_name))

def getFilesWithAPartOfTheName( path, content):
    files = []
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and content in i:
            files.append(i)
    return files


