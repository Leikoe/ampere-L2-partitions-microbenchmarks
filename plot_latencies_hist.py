import csv

import matplotlib.pyplot as plt

with open("l2_latency_results.csv") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=";")
    smid = int(next(spamreader)[0].split(":")[1])
    print(f"smid: {smid}")
    next(spamreader)  # skip header

    latencies = []
    for row in spamreader:
        i, latency = [int(x) for x in row]
        latencies.append(latency)

    n, bins, patches = plt.hist(
        latencies,
        bins=10,
        color="#5c9cd6",
        alpha=0.9,
        edgecolor="white",
    )

    plt.ylabel("Number of accesses")
    plt.xlabel("Cycles")
    plt.title(f"L2 hit latencies for SM {smid}")
    plt.tight_layout()
    plt.savefig(f"l2_latency_histogram_smid{smid}.png")
