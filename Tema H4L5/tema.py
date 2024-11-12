import numpy as np

# stari posibile de dificultate
stari = ["dificil", "mediu", "usor"]
nr_stari = len(stari)

# observatii posibile de note
observatii = ["FB", "B", "S", "NS"]
nr_observatii = len(observatii)

# probabilitatea de start (probabilitati initiale egale)
probabilitate_start = [1/3, 1/3, 1/3]

# matricea de tranzitie intre stari (in functie de starea curenta)
probabilitate_tranzitie = np.array([
    [0, 0.5, 0.5],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]
])

# matricea de emisie (probabilitatea ca o anumita nota sa fie obtinuta la o anumita dificultate)
probabilitate_emisie = np.array([
    [0.1, 0.2, 0.4, 0.3],  # pentru test dificil
    [0.15, 0.25, 0.5, 0.1], # pentru test mediu
    [0.2, 0.3, 0.4, 0.1]    # pentru test usor
])

# secventa de observatii (notele primite de student)
secventa_observatii = ["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B", "S"]

def algoritm_viterbi(observatii, stari, probabilitate_start, probabilitate_tranzitie, probabilitate_emisie, secventa):
    nr_stari = len(stari)
    nr_observatii = len(secventa)
    
    # convertim observatiile in indecsi pentru a putea accesa probabilitatile de emisie
    secventa_indici = [observatii.index(obs) for obs in secventa]
    
    # initializam matricea Viterbi pentru stocarea probabilitatilor maxime
    viterbi = np.zeros((nr_observatii, nr_stari))
    
    # initializam matricea de traseu pentru a tine evidenta starilor anterioare
    traseu = np.zeros((nr_observatii, nr_stari), dtype=int)
    
    # etapa de initializare (pasul t = 0)
    for s in range(nr_stari):
        viterbi[0, s] = probabilitate_start[s] * probabilitate_emisie[s, secventa_indici[0]]
        traseu[0, s] = s
    
    # etapa de recursivitate pentru t > 0
    for t in range(1, nr_observatii):
        for s in range(nr_stari):
            max_probabilitate, stare_anterioara = max(
                (viterbi[t - 1, s_prev] * probabilitate_tranzitie[s_prev, s] * probabilitate_emisie[s, secventa_indici[t]], s_prev)
                for s_prev in range(nr_stari)
            )
            viterbi[t, s] = max_probabilitate
            traseu[t, s] = stare_anterioara

    # etapa de backtracking pentru a gasi calea cea mai probabila
    cale_optima = [0] * nr_observatii
    cale_optima[-1] = np.argmax(viterbi[-1, :])
    for t in range(nr_observatii - 2, -1, -1):
        cale_optima[t] = traseu[t + 1, cale_optima[t + 1]]
    
    # convertim indecsii in denumiri ale starilor
    secventa_cale_optima = [stari[i] for i in cale_optima]
    probabilitate_totala = viterbi[-1, cale_optima[-1]]
    
    return secventa_cale_optima, probabilitate_totala

# apelam algoritmul Viterbi pe secventa de observatii data
secventa, probabilitate = algoritm_viterbi(observatii, stari, probabilitate_start, probabilitate_tranzitie, probabilitate_emisie, secventa_observatii)
print("Cea mai probabila secventa de dificultati:", secventa)
print("Probabilitatea acestei secvente:", probabilitate)