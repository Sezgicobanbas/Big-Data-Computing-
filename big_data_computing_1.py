from pyspark import SparkContext, SparkConf
import sys
import os
import itertools
import random as rand
import csv
import time

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    if isinstance(p1, str):
        x1, y1 = map(float, p1.split(','))
    else:
        x1, y1 = p1

    if isinstance(p2, str):
        x2, y2 = map(float, p2.split(','))
    else:
        x2, y2 = p2
    
    return (((x1 - x2) ** 2) + ((y1 - y2) ** 2)) ** 0.5


### D = distance threshold
### M = maximum allowable size of neighbourhood, if point has more than M neighbors,
##it considered to be non-outlier
### k =#of outliers to be reported


def ExactOutliers(points, D, M, K):
    outliers = []
    n = len(points)
    counts = [1] * n
    
    for i in range(n-1):
        for j in range(i+1, n):
            dist = euclidean_distance(points[i], points[j])
            if dist <= D:
                counts[j] += 1
                counts[i] += 1
                
    for k in range(n):
        if counts[k] <= M:
            outliers.append((points[k], counts[k]))
            
            
    # Sort outliers by |B_S(p, D)|
    outliers.sort(key=lambda x: x[1])

    # Print results
    print("Number of Outliers = {}".format(len(outliers)))
    for i in range(min(K, len(outliers))):
        print("Point: ",outliers[i][0])
def cell_identifier(point, D):
    if isinstance(point, str):
        i_0, j_0 = point.split(',')
    elif isinstance(point, tuple) and len(point) == 2:
        i_0, j_0 = point
    else:
        raise ValueError("Invalid point format. Expected string or tuple of length 2.")

    Lambda = D / (2 * (2**0.5))
    i = int(float(i_0) // Lambda)
    j = int(float(j_0) // Lambda)
    return ((i, j), 1)

def gather_pairs(cells):
        cells_dict = {}
        for c in cells[1]:
            cell, no_of_points = c[0], c[1]
            if cell not in cells_dict.keys():
                cells_dict[cell] = no_of_points
            else:
                cells_dict[cell] += no_of_points
        return [(key, cells_dict[key]) for key in cells_dict.keys()]


def MRApproxOutliers(points_rdd, D, M, K, L):
    # Step 1: Transform RDD into RDD of non-empty cells with their sizes
    cells_list_with_sizes_RDD = (points_rdd.map(lambda point: cell_identifier(point, D)) # <-- MAP PHASE (R1)
                                 .groupBy(lambda x: (rand.randint(0,L-1))) # <-- SHUFFLE+GROUPING
                                 .flatMap(gather_pairs)                    # <-- REDUCE PHASE (R1)
                                 .reduceByKey(lambda x, y: x + y))         # <-- REDUCE PHASE (R2)


    # Step 2.a: Collect the data from cells_list_with_sizes_RDD to dictionary
    cells_with_sizes = cells_list_with_sizes_RDD.collectAsMap()


    # Step 3: Calculate N3 and N7 values to non-empty cells
    def compute_N3_N7(cell):
        i, j = cell[0]
        cell_size = cell[1]
        N3 = sum([cells_with_sizes[ii, jj] for ii in range(i-1, i+2) for jj in range(j-1, j+2) if (ii, jj) in cells_with_sizes])
        N7 = sum([cells_with_sizes[ii, jj] for ii in range(i-3, i+4) for jj in range(j-3, j+4) if (ii, jj) in cells_with_sizes])
        return (cell[0], (cell_size, N3, N7))

    cells_with_info_rdd = cells_list_with_sizes_RDD.map(compute_N3_N7)

    # Step 4: Compute sure and uncertain outliers
    sure_outliers_rdd = cells_with_info_rdd.filter(lambda cell: cell[1][1] <= M and cell[1][2] <= M).collect()
    sure_outliers_count = sum([cell[1][0] for cell in sure_outliers_rdd])
    uncertain_points_rdd = cells_with_info_rdd.filter(lambda cell: cell[1][1] <= M and cell[1][2] > M).collect()
    uncertain_points_count = sum([cell[1][0] for cell in uncertain_points_rdd])



    # Step 5: Print output
    print("Number of sure outliers: {}".format(sure_outliers_count))
    print("Number of uncertain points: {}".format(uncertain_points_count))

    sorted_cells = cells_with_info_rdd.sortByKey().sortBy(lambda cell: cell[1][0]).take(K)
    for cell in sorted_cells:
        print("Cell: {} Size = {}".format(cell[0], cell[1][0]))



def main():
    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 6, "Usage: python G043HW1_.py <D> <M> <K> <L> <file_name>"

    # SPARK SETUP
    conf = SparkConf().setAppName('G043HW1_')
    sc = SparkContext(conf=conf)

    # INPUT READING
    # 1. Read parameter D - distance
    D = sys.argv[1]
    assert D.replace(".", "", 1).isdigit(), "D must be an integer"
    assert float(D) > 0, "D must be greater than zero"
    D = float(D)

    # 2. Read parameter M - no. of points
    M = sys.argv[2]
    assert M.isdigit(), "M must be an integer"
    M = int(M)

    # 3. Read parameter K - no. of outliers to print
    K = sys.argv[3]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # 4. Read parameter L - no. of partitions
    L = sys.argv[4]
    assert L.isdigit(), "L must be an integer"
    L = int(L)

    # 5. Read input file and subdivide it into L random partitions
    data_path = sys.argv[5]
    assert os.path.isfile(data_path), "File or folder not found"
    #docs = sc.textFile(data_path).repartition(numPartitions=L).cache()
    docs = sc.textFile(data_path).cache()
    #points = sc.textFile(data_path).cache() #for exactOutliers
    with open(data_path, 'r') as file:
        reader = csv.reader(file)
        listOfPoints = [tuple(map(float, row)) for row in reader]


    # PRINT INPUT PARAMETERS
    print(data_path)
    print("D=", D)
    print("M=", M)
    print("K=", K)
    print("L=", L)

    # SETTING GLOBAL VARIABLES
    numpoints = docs.count();
    print("Number of points = ", numpoints)

    #Code to transform csv file into list for ExactAlg
    #with open(data_path, newline='') as f:
    #    reader = csv.reader(f)
    #    docs_str = [tuple(row) for row in reader]
    #    docs_list = [(float(docs_str[i][0]), float(docs_str[i][1])) for i in range(len(docs_str))]
   
   
    # EXACT ALGORITHM
    if numpoints <= 200000:
        start_time_exact = time.time()
        ExactOutliers(listOfPoints, D, M, K)
        end_time_exact = time.time()
        print("Running time of ExactOutliers = ", int((end_time_exact - start_time_exact)*1000), " ms")
    
    # APPROXIMATE ALGORITHM
    start_time_approx = time.time()
    MRApproxOutliers(docs, D, M, K, L)
    end_time_approx = time.time()
    print("Running time of MRApproxOutliers = ", int((end_time_approx - start_time_approx)*1000), "ms")

if __name__ == "__main__":
	main()