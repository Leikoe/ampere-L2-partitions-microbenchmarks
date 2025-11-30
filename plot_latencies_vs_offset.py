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

    # n, bins, patches = plt.hist(
    #     latencies,
    #     bins=10,
    #     color="#5c9cd6",
    #     alpha=0.9,
    #     edgecolor="white",
    # )

    latencies = latencies[:4096]  # truncate
    offsets = [i * 4 for i in range(len(latencies))]
    plt.plot(offsets, latencies, linewidth=1)

    plt.vlines(
        [i * 4096 for i in range((len(latencies) // 1024) + 1)],
        0,
        500,
        linestyles="dotted",
        colors="red",
        label="4KB sections",  # IMPORTANT: This is KB and not KiB
    )
    plt.legend()
    plt.ylabel("Load latency (Cycles)")
    plt.xlabel("Offset (Bytes)")
    plt.title(f"L2 hit latencies for SM {smid}")
    plt.savefig(f"l2_latency_vs_offset_smid{smid}.png")
