import numpy as np
import matplotlib.pyplot as plt
import sys
import csv

np.random.seed(123)

def generate_sinusoidal(ampiezza=2, frequenza=0.5, offset=1):
    """ Genera un tempo di interarrivo sinusoidale. """
    t = np.abs(ampiezza * np.sin(2 * np.pi * frequenza * np.random.random())) + offset
    return t, 1/t

def generate_square_wave(ampiezza=2, offset=1):
    """Generates an inter-arrival time with a square wave pattern."""
    t = ampiezza * (1 if np.random.random() > 0.5 else -1) + offset
    return abs(t), 1/abs(t)

def generate_sawtooth_wave(ampiezza=2, periodo=10, offset=1):
    """Generates an inter-arrival time with a sawtooth wave pattern."""
    t = (ampiezza * (np.random.random() % periodo) / periodo) + offset
    return t, 1/t

def generate_logistic_map(r=3.8, x0=None, offset=1):
    """Generates an inter-arrival time based on the logistic map for chaotic variations."""
    if x0 is None:
        x0 = np.random.random()  # Random initial condition
    x = r * x0 * (1 - x0)  # Logistic map iteration
    t = abs(x * 10) + offset
    return t, 1/t

def generate_gaussian_modulated(ampiezza=2, frequenza=0.5, offset=1, sigma=0.5):
    """Generates an inter-arrival time with a sinusoidal wave modulated by Gaussian noise."""
    t = np.abs(ampiezza * np.sin(2 * np.pi * frequenza * np.random.random()) + np.random.normal(0, sigma)) + offset
    return t, 1/t


def graph(x, z=[], title="Interarrivi", filename="plot.png"):
    """ Funzione per salvare l'immagine del grafico. """

    print("Graphing...")
    plt.figure(figsize=(21, 7))
    plt.plot(range(len(x)), x, label="Interarrivals", color='b')
    plt.plot(range(len(z)), z, label="Rates", color='g')
    plt.xlabel("Eventi")
    plt.ylabel("Interarrivi")
    plt.title(f"Distribuzione: {title}")
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()


def main():
    """ Funzione principale che seleziona la distribuzione e simula gli eventi. """
    if len(sys.argv) < 3:
        print("Uso: python script.py <sinusoid|square-wave|sawtooth-wave|logistic-map> <close-the-door-time>")
        return

    distribution = sys.argv[1].lower()
    close_the_door = float(sys.argv[2])  # Tempo massimo consentito

    # Parametri opzionali per le distribuzioni
    total_time = 0
    times = []
    rates = []

    while total_time <= close_the_door:
        # Genera il tempo di interarrivo a seconda della distribuzione
        if distribution == "sinusoid":
            interarrivo, rate = generate_sinusoidal(ampiezza=1.7, frequenza=0.7, offset=1.0)
        elif distribution == "square-wave":
            interarrivo, rate = generate_square_wave(ampiezza=1.2, offset=0.5)
        elif distribution == "sawtooth-wave":
            interarrivo, rate = generate_sawtooth_wave(ampiezza=0.2, periodo=0.35)
        elif distribution == "logistic-map":
            interarrivo, rate = generate_logistic_map()
        else:
            print("Distribuzione non supportata!")
            return

        if total_time <= close_the_door:  # Aggiungi solo se non supera il limite
            times.append(interarrivo)
            rates.append(rate)

        total_time += interarrivo

    # Genera e salva grafico
    graph(times[:300], rates[:300], distribution, f"traces/img/{distribution}_plot.png")

    # Salva interarrivi (simulation trace), seconda metà
    with open("traces/synthetic/synthetic_"+distribution+"_arrivals.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["interarrival"])
        writer.writerows(zip(times[(int(len(times)/2)):]))

    # Salva rates (training data), prima metà
    with open("models/training/synthetic_" + distribution + "_rates.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rate"])
        writer.writerows(zip(rates))


if __name__ == "__main__":
    main()