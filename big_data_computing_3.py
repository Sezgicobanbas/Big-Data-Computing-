from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import random
import math

# Function to compute true frequent items
def get_true_frequent_items(counts, phi, total_items):
    threshold = phi * total_items
    return sorted([item for item, count in counts.items() if count >= threshold])

# Reservoir Sampling function
def reservoir_sampling(item, reservoir, m, total_items):
    if len(reservoir) < m:
        reservoir.append(item)
    else:
        i = random.randint(0, m-1)
        p = random.uniform(0,1)
        if p <= m/total_items:
            reservoir[i] = item

# Sticky Sampling function
def sticky_sampling(item, sampling_dict, r,n):
    # Check if item is already in the sampling dictionary
    if item in sampling_dict:
        sampling_dict[item] += 1
    else:
        # Determine if the item should be sampled
        sampling_rate = r / n
        if random.random() < sampling_rate:
            sampling_dict[item] = 1
        

# Main function
def main():
    assert len(sys.argv) == 6, "Usage: python G043HW3.py <n> <phi> <epsilon> <delta> <portExp>"
    
    # set up Spark context
    conf = SparkConf().setMaster("local[*]").setAppName("G043HW3")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")
    stopping_condition = threading.Event()


    # INPUT READING
    n = int(sys.argv[1])
    phi = float(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    portExp = int(sys.argv[5])

    print("INPUT PROPERTIES")
    print("n = {} phi = {} epsilon = {} delta = {} port = {}".format(n, phi, epsilon, delta, portExp))

    m = math.ceil(1 / phi)
    r = math.log(1 / (phi * delta)) / epsilon

    # Create a DStream from the specified port
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    
    # Initialize variables
    item_counts = {}  # Dictionary to store the count of each item
    reservoir = []  # List to store items for reservoir sampling
    sampling_dict = {}  # Dictionary to store items for sticky sampling
    total_items = 0  # Total number of items processed
    streamLength = [0]

    # Process each RDD in the stream
    def process_rdd(rdd):
        nonlocal total_items
        batch_size = rdd.count()
        if streamLength[0]>=n:
            return
        streamLength[0] += batch_size
        items = rdd.collect()

        for item in items:
            total_items += 1
            item = int(item)

            # Update true counts
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1

            # Perform reservoir sampling
            reservoir_sampling(item, reservoir, m, total_items)

            # Perform sticky sampling
            sticky_sampling(item, sampling_dict, r, n)
        if streamLength[0] >= n:
            stopping_condition.set()

    stream.foreachRDD(lambda batch: process_rdd(batch))

    # start the computation
    print("Starting streaming engine")
    ssc.start()
    print("Waiting for shutdown condition")
    stopping_condition.wait()
    print("Stopping the streaming engine")
    ssc.stop(stopSparkContext=False, stopGraceFully=True)
    print("Streaming engine stopped")

    # Filter sticky sampling results
    filtered_sampling_dict = {k: v for k, v in sampling_dict.items() if v > (phi - epsilon) * n}

    # Print results
    true_frequent_items = get_true_frequent_items(item_counts, phi, total_items)
    print("EXACT ALGORITHM")
    print(f"Number of items in the data structure = {streamLength[0]}")
    print(f"Number of true frequent items = {len(true_frequent_items)}")
    print("True frequent items:")
    for item in true_frequent_items:
        print(item)
    print("RESERVOIR SAMPLING")
    print(f"Size m of the sample = {m}")
    print(f"Number of estimated frequent items = {len(set(reservoir))}")
    print("Estimated frequent items:")
    for item in sorted(set(reservoir)):
        if item in true_frequent_items:
            print(item, "+")
        else:
            print(item, "-")
    print("STICKY SAMPLING")
    print(f"Number of items in the Hash Table = {len(sampling_dict)}")
    print(f"Number of estimated frequent items = {len(filtered_sampling_dict)}")
    print("Estimated frequent items:")
    for item in sorted(filtered_sampling_dict.keys()):
        if item in true_frequent_items:
            print(item, "+")
        else:
            print(item, "-")
    


if __name__ == "__main__":
    main()
