import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
import math

"""
  1⃣ Genera un numero di arrivi seguendo una distribuzione data per ogni step temporale indicato:
    in pratica divide il tempo in intervalli e calcola quanti eventi devono arrivare in ogni step.
  2⃣ Distribuisce gli arrivi casualmente all'interno di ogni intervallo temporale.
  3️⃣ calcola i tempi inter-arrivo per ottenere una serie temporale realistica.
"""

np.random.seed(123)
DISTRIBUTION = "sinusoid"

PERIOD = 1800  # Periodo dell'onda sinusoidale in secondi (es. 1800 secondi = 30 minuti)
FREQ = 2 / PERIOD * 2 * np.pi  # Frequenza della sinusoide (ciclo completo ogni PERIOD secondi)

TRACE_DURATION = 3 * PERIOD  # Durata totale della simulazione (3 periodi della sinusoide)
STEP_LEN = 120  # Lunghezza di ogni step (es. 1800/30 = 60 secondi)
STEPS = int (TRACE_DURATION / STEP_LEN) # Numero di passi temporali (es. 5400 / 60 = 90)


def generate_sinusoidal(i, min_rate=5, max_rate=50):
    return np.round(min_rate + (max_rate - min_rate) / 2 + (max_rate - min_rate) / 2 * math.sin(FREQ * STEP_LEN * i))

def generate_square_wave(i, min_rate=5, max_rate=50, period=600):
    return max_rate if (i * STEP_LEN) % (2 * period) < period else min_rate

def generate_sawtooth_wave(i, min_rate=5, max_rate=50, period=600):
    phase = (i * STEP_LEN) % period / period
    return np.round(min_rate + (max_rate - min_rate) * phase)

def generate_logistic_map(i, r=3.8, x0=0.5, min_rate=5, max_rate=50):
    x = x0
    for _ in range(i):
        x = r * x * (1 - x)
    return np.round(min_rate + (max_rate - min_rate) * x)

def generate_gaussian_modulated(i, min_rate=5, max_rate=50, sigma=0.5):
    mod = np.exp(-0.5 * ((i - STEPS / 2) / (sigma * STEPS / 2))**2)
    return np.round(min_rate + (max_rate - min_rate) * mod)


def graph(interarrivals, rates, file_path):
    """ Funzione per salvare l'immagine del grafico. """

    # Creazione del grafico con due subplot affiancati
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Grafico degli interarrivi
    axs[0].plot(interarrivals)#, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axs[0].set_xlabel("Interarrival Time (minutes)")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Interarrival Times")
    axs[0].grid(True)

    # Grafico dei rates
    axs[1].plot(np.arange(STEPS) * STEP_LEN, rates, label=f"Arrival Rate (per {STEP_LEN})", marker="o", color="r")
    axs[1].set_xlabel("Time (minutes)")
    axs[1].set_ylabel(f"Arrival Rate")
    axs[1].set_title(f"*{DISTRIBUTION}* Arrival Rate Over Time")
    axs[1].legend()
    axs[1].grid(True)

    # Mostra i grafici
    plt.tight_layout()
    plt.savefig(file_path)


def main():
    """ Funzione principale che seleziona la distribuzione e simula gli eventi. """
    if len(sys.argv) < 2:
        print("Uso: python script.py <sinusoid|square-wave|sawtooth-wave|logistic-map>")
        return

    DISTRIBUTION = sys.argv[1].lower()
    nArrivals = np.zeros(STEPS)  # Inizializza un array per gli arrivi in ogni step

    for i in range(STEPS):
        # Genera la quantità di arrivi per ogni step a seconda della distribuzione
        if DISTRIBUTION == "sinusoid":
            nArrivals[i] = generate_sinusoidal(i)
        elif DISTRIBUTION == "square-wave":
            nArrivals[i] = generate_square_wave(i)
        elif DISTRIBUTION == "sawtooth-wave":
            nArrivals[i] = generate_sawtooth_wave(i)
        elif DISTRIBUTION == "logistic-map":
            nArrivals[i] = generate_logistic_map(i)
        elif DISTRIBUTION == "gaussian-modulated":
            nArrivals[i] = generate_gaussian_modulated(i)
        else:
            print("Distribuzione non supportata!")
            return

    # Otteniamo i rates a partire dall'array di arrivi (nel nostro caso STEP_LEN=120s)
    rates = nArrivals / STEP_LEN

    total_arrivals = int(sum(nArrivals))      # Calcola il numero totale di arrivi
    arrival_times = np.zeros(total_arrivals)  # Inizializza gli array dei tempi di arrivo
    count = 0
    rng = np.random.default_rng(123)
    for i in range(STEPS):
        t0 = STEP_LEN * i  # Inizio dell'intervallo temporale
        t1 = t0 + STEP_LEN  # Fine dell'intervallo temporale
        # Genera arrivi casuali dentro [t0, t1] e li ordina;
        # praticamente genera tempi di arrivo uniformemente distribuiti in ogni step.
        arrival_times[count:count + int(nArrivals[i])] = np.sort(rng.uniform(t0, t1, nArrivals[i].astype(int)))
        count += int(nArrivals[i])
    inter_arrival_times = np.diff(arrival_times)

    # Genera e salva grafico
    graph(inter_arrival_times, rates, f"traces/img/{DISTRIBUTION}_plot.png")

    # Salva interarrivi (simulation trace), seconda metà
    with open("traces/synthetic/synthetic_"+DISTRIBUTION+"_arrivals.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(inter_arrival_times))

    # Salva rates (training data), prima metà
    with open("models/training/synthetic_" + DISTRIBUTION + "_rates.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rate"])
        writer.writerows(zip(rates))


if __name__ == "__main__":
    main()