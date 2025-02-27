from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import csv
import time
from scipy.spatial import distance
import numpy as np

def strToFloats(strings):
    out = tuple(map(float, strings.split(',')))
    return out

def SequentialFFT(points, K):
    # Implement the Farthest-First Traversal algorithm
    points_list = list(points)
    centers = []
    num_points = len(points_list)
    first_center = points_list[np.random.randint(0, num_points)]
    centers.append(first_center)
    
    distances_array = np.linalg.norm(np.array(centers[-1])- np.array(points_list), axis = 1)
    
    for _ in range(0, K-1):
        dist_with_last_c = np.linalg.norm(np.array(centers[-1]) - np.array(points_list), axis = 1)
        distances_array = np.minimum(distances_array, dist_with_last_c)
        centers.append(points_list[np.argmax(distances_array)])
    return centers


def MRFFT(sc,points_rdd, K):
    # Round 1: Run SequentialFFT on partitions to get coreset
    R1_start_time = time.time()
    coreset = points_rdd.mapPartitions(lambda iterator: SequentialFFT(iterator, K)).cache()
    coreset_local = coreset.collect()
    R1_end_time = time.time()
    
    # Round 2: Run SequentialFFT on coreset to get centers
    R2_start_time = time.time()
    centers = SequentialFFT(coreset_local, K)
    centers_RDD = sc.broadcast(centers)
    R2_end_time = time.time()
    
    # Round 3: Compute radius of clustering
    R3_start_time = time.time()
    
    def compute_radius(point):
        min_distance = distance.cdist([point], centers_RDD.value).min()
        return min_distance
        
    max_radius = points_rdd.map(compute_radius).max()
    R3_end_time = time.time()
    
    return float(max_radius), int((R1_end_time - R1_start_time)*1000), int((R2_end_time - R2_start_time)*1000), int((R3_end_time - R3_start_time)*1000)





    # Function to calculate cells for approximate algorithm
def cell_identifier(point, D):
    if isinstance(point, tuple) and len(point) == 2:
        i_0, j_0 = point
    else:
        raise ValueError("Invalid point format. Expected string or tuple of length 2.")

    Lambda = D / (2 * (2**0.5))
    i = int(float(i_0) // Lambda)
    j = int(float(j_0) // Lambda)
    return ((i, j), 1)


def gather_pairs(cells):
        cells_dict = {}
        for c in cells:
            cell, no_of_points = c[0], c[1]
            if cell not in cells_dict.keys():
                cells_dict[cell] = no_of_points
            else:
                cells_dict[cell] += no_of_points
        return [(key, cells_dict[key]) for key in cells_dict.keys()]


def MRApproxOutliers(points_rdd, D, M):
    # Transform RDD into RDD of non-empty cells with their sizes
    cells_list_with_sizes_RDD = (points_rdd.map(lambda point: cell_identifier(point, D)) # <-- MAP PHASE (R1)
                                 .mapPartitions(gather_pairs)                    # <-- REDUCE PHASE (R1)
                                 .reduceByKey(lambda x, y: x + y))         # <-- REDUCE PHASE (R2)


    # Collect the data from cells_list_with_sizes_RDD to dictionary
    cells_with_sizes = cells_list_with_sizes_RDD.collectAsMap()


    # Calculate N3 and N7 values to non-empty cells
    def compute_N3_N7(cell):
        i, j = cell[0]
        cell_size = cell[1]
        N3 = sum([cells_with_sizes[ii, jj] for ii in range(i-1, i+2) for jj in range(j-1, j+2) if (ii, jj) in cells_with_sizes])
        N7 = sum([cells_with_sizes[ii, jj] for ii in range(i-3, i+4) for jj in range(j-3, j+4) if (ii, jj) in cells_with_sizes])
        return (cell[0], (cell_size, N3, N7))

    cells_with_info_rdd = cells_list_with_sizes_RDD.map(compute_N3_N7)

    # Compute sure and uncertain outliers
    sure_outliers_rdd = cells_with_info_rdd.filter(lambda cell: cell[1][1] <= M and cell[1][2] <= M).collect()
    sure_outliers_count = sum([cell[1][0] for cell in sure_outliers_rdd])
    uncertain_points_rdd = cells_with_info_rdd.filter(lambda cell: cell[1][1] <= M and cell[1][2] > M).collect()
    uncertain_points_count = sum([cell[1][0] for cell in uncertain_points_rdd])


    print("Number of sure outliers = {}".format(sure_outliers_count))
    print("Number of uncertain points = {}".format(uncertain_points_count))




def main():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 5, "Usage: python G043HW2.py <file_name> <M> <K> <L>"

    # SPARK SETUP
    conf = SparkConf().setAppName('G043HW2')
    conf.set("spark.locality.wait", "0s")
    sc = SparkContext(conf=conf)

    # INPUT READING
    # 1. Read input file 
    data_path = sys.argv[1]
    #assert os.path.isfile(data_path), "File or folder not found"
    

    # 2. Read parameter M - no. of points
    M = sys.argv[2]
    assert M.isdigit(), "M must be an integer"
    M = int(M)

    # 3. Read parameter K - no. of clusters
    K = sys.argv[3]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # 4. Read parameter L - no. of partitions
    L = sys.argv[4]
    assert L.isdigit(), "L must be an integer"
    L = int(L)

    #transform input file
    rawData = sc.textFile(data_path)
    inputPoints = rawData.map(lambda x : strToFloats(x)).repartition(L).cache()
    
    # PRINT INPUT PARAMETERS
    print("{} M={} K={} L={}".format(data_path, M, K, L))


    # SETTING GLOBAL VARIABLES
    numpoints = inputPoints.count()
    print("Number of points =", numpoints)
    
    # MRFFT ALGORITHM
    D, runtime_R1, runtime_R2, runtime_R3 = MRFFT(sc,inputPoints, K)
    print("Running time of MRFFT R1 =", runtime_R1, 'ms')
    print("Running time of MRFFT R2 =", runtime_R2, 'ms')
    print("Running time of MRFFT R3 =", runtime_R3, 'ms')
    print("Radius =", D)
    
    # APPROXIMATE ALGORITHM
    start_time_approx = time.time()
    MRApproxOutliers(inputPoints, D, M)
    end_time_approx = time.time()
    print("Running time of MRApproxOutliers =", int((end_time_approx - start_time_approx)*1000), 'ms')

if __name__ == "__main__":
	main()


