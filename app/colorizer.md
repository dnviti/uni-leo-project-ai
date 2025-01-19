# Progetto: Risoluzione del Problema di Uniform Coloring su Griglia con Agente Mobile

## Introduzione

In questa lezione, affronteremo il problema di **Uniform Coloring** su una griglia utilizzando un agente mobile. L'obiettivo è sviluppare un algoritmo che permetta all'agente di colorare le celle della griglia secondo determinati vincoli, utilizzando tecniche di ricerca nello spazio degli stati e algoritmi di Intelligenza Artificiale.

## Obiettivi della Lezione

- Descrivere formalmente il dominio del problema e i vincoli associati.
- Implementare le classi necessarie per la ricerca nello spazio degli stati utilizzando il framework di AIMA-python.
- Utilizzare dataset di immagini (MNIST/eMNIST) per l'acquisizione e la classificazione dell'input.
- Risolvere il problema utilizzando algoritmi di ricerca informati e non informati.
- Simulare l'esecuzione del piano d'azione calcolato.

---

## 1. Descrizione Formale del Dominio e dei Vincoli

### Dominio del Problema

Abbiamo una griglia bidimensionale composta da celle. Un agente (testina) può muoversi all'interno della griglia e colorare le celle secondo determinati colori.

### Vincoli sulle Azioni dell'Agente

- **V1**: L'agente può compiere un solo passo alla volta.
- **V2**: L'agente può muoversi solo in orizzontale o verticale tra celle adiacenti (non in diagonale).
- **V3**: L'azione di colorazione ha un costo associato; diversi colori possono avere costi differenti.
- **V4**: L'agente non può uscire dai bordi della griglia.
- **V5**: Una cella già colorata con il colore corretto non necessita di essere ricolorata.

### Stati e Azioni

- **Stato**: Configurazione della griglia con le posizioni dell'agente e lo stato di colorazione di ogni cella.
- **Azioni Possibili**:
  - **Movimento**: Su, Giù, Sinistra, Destra.
  - **Colorazione**: Applicare un colore alla cella corrente.

---

## 2. Implementazione delle Classi per la Ricerca nello Spazio degli Stati

Utilizzeremo le classi del framework AIMA-python per implementare il problema.

### 2.1. Importazione delle Librerie Necessarie

```python
import sys
!{sys.executable} -m pip install aima-python opencv-python numpy matplotlib
```

```python
from aima_python.search import Problem, astar_search, breadth_first_tree_search, depth_first_tree_search
import numpy as np
import cv2
import matplotlib.pyplot as plt
```

### 2.2. Definizione della Classe `UniformColoring`

```python
class UniformColoring(Problem):
    def __init__(self, initial, goal):
        super().__init__(initial, goal)

    def actions(self, state):
        actions = []
        agent_pos = state['agent_pos']
        grid = state['grid']
        max_row, max_col = grid.shape

        # Movimento
        moves = {'Up': (-1, 0), 'Down': (1, 0), 'Left': (0, -1), 'Right': (0, 1)}
        for action, (dr, dc) in moves.items():
            new_row = agent_pos[0] + dr
            new_col = agent_pos[1] + dc
            if 0 <= new_row < max_row and 0 <= new_col < max_col:
                actions.append(action)

        # Colorazione (se il colore della cella corrente non è quello desiderato)
        current_color = state['grid'][agent_pos]
        goal_color = self.goal['grid'][agent_pos]
        if current_color != goal_color:
            actions.append('Paint')

        return actions

    def result(self, state, action):
        new_state = {
            'agent_pos': state['agent_pos'],
            'grid': state['grid'].copy()
        }

        if action == 'Up':
            new_state['agent_pos'] = (state['agent_pos'][0] - 1, state['agent_pos'][1])
        elif action == 'Down':
            new_state['agent_pos'] = (state['agent_pos'][0] + 1, state['agent_pos'][1])
        elif action == 'Left':
            new_state['agent_pos'] = (state['agent_pos'][0], state['agent_pos'][1] - 1)
        elif action == 'Right':
            new_state['agent_pos'] = (state['agent_pos'][0], state['agent_pos'][1] + 1)
        elif action == 'Paint':
            agent_pos = state['agent_pos']
            new_state['grid'][agent_pos] = self.goal['grid'][agent_pos]

        return new_state

    def goal_test(self, state):
        return np.array_equal(state['grid'], self.goal['grid'])

    def h(self, node):
        """Heuristica: numero di celle da colorare più distanza Manhattan dall'agente alla cella più lontana da colorare."""
        state = node.state
        mismatched_cells = np.sum(state['grid'] != self.goal['grid'])
        agent_pos = state['agent_pos']
        goal_positions = np.argwhere(state['grid'] != self.goal['grid'])
        if goal_positions.size == 0:
            return 0
        distances = [abs(agent_pos[0] - pos[0]) + abs(agent_pos[1] - pos[1]) for pos in goal_positions]
        return mismatched_cells + min(distances)
```

### 2.3. Analisi dell'Euristica

L'euristica definita è ammissibile poiché non sovrastima mai il costo reale per raggiungere lo stato goal. È consistente perché la differenza tra le stime dell'euristica per due stati successivi non supera mai il costo dell'azione che li collega.

---

## 3. Acquisizione e Classificazione degli Input

Utilizzeremo OpenCV per elaborare l'immagine di input e il dataset MNIST per riconoscere cifre e lettere.

### 3.1. Caricamento e Pre-elaborazione dell'Immagine

```python
# Caricamento dell'immagine
image = cv2.imread('griglia_input.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap='gray')
plt.title('Immagine di Input')
plt.show()
```

*Nota: Assicurarsi che l'immagine 'griglia_input.png' sia nella stessa directory del notebook.*

### 3.2. Segmentazione della Griglia

```python
# Thresholding per binarizzare l'immagine
_, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# Rilevamento dei contorni per individuare le celle
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Individuazione delle celle e creazione della griglia
cells = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 20 and h > 20:  # Filtra i contorni troppo piccoli
        cell = image[y:y+h, x:x+w]
        cells.append((cell, (x, y, w, h)))
```

### 3.3. Classificazione delle Celle

Utilizzeremo un modello pre-addestrato sul dataset MNIST per riconoscere cifre (colori) e lettere (posizione iniziale dell'agente).

```python
from tensorflow.keras.models import load_model

# Caricamento del modello pre-addestrato (da implementare)
model = load_model('mnist_model.h5')

grid_size = int(np.sqrt(len(cells)))
grid = np.zeros((grid_size, grid_size), dtype=int)
agent_pos = None

# Ordina le celle in base alla loro posizione per riempire la griglia correttamente
cells.sort(key=lambda x: (x[1][1], x[1][0]))  # Ordina per y, poi per x

index = 0
for i in range(grid_size):
    for j in range(grid_size):
        cell_image, (x, y, w, h) = cells[index]
        cell_image = cv2.resize(cell_image, (28, 28))
        cell_image = cell_image.reshape(1, 28, 28, 1) / 255.0
        prediction = model.predict(cell_image)
        predicted_class = np.argmax(prediction)

        # Se la classe predetta è una lettera (assumiamo codifica ASCII per le lettere maiuscole)
        if 65 <= predicted_class <= 90:
            agent_pos = (i, j)
            grid[i, j] = 0  # Colore iniziale per le celle con l'agente
        else:
            grid[i, j] = predicted_class  # Colore della cella

        index += 1
```

*Nota: Il codice sopra presuppone l'esistenza di un modello `mnist_model.h5` pre-addestrato capace di riconoscere cifre e lettere. Dovrai addestrare o importare un modello adeguato per questa parte.*

### 3.4. Definizione dello Stato Iniziale e Goal

```python
initial_state = {
    'agent_pos': agent_pos,
    'grid': grid
}

# Definizione dello stato goal (da definire in base al problema)
goal_grid = np.zeros_like(grid)  # Ad esempio, griglia tutta colorata con 0
goal_state = {
    'agent_pos': agent_pos,  # L'agente può essere in qualsiasi posizione finale
    'grid': goal_grid
}
```

---

## 4. Risoluzione del Problema

### 4.1. Ricerca Non Informata: Breadth-First Tree Search

```python
problem = UniformColoring(initial_state, goal_state)
solution_node = breadth_first_tree_search(problem)

# Estrazione del percorso di azioni
actions = solution_node.solution()
print("Azioni trovate con Breadth-First Tree Search:")
print(actions)
```

### 4.2. Ricerca Informata: A* Search

```python
solution_node = astar_search(problem)

# Estrazione del percorso di azioni
actions = solution_node.solution()
print("Azioni trovate con A* Search:")
print(actions)
```

### 4.3. Confronto delle Soluzioni

- **Breadth-First Search**: Garantisce la soluzione più breve in termini di numero di azioni, ma può essere computazionalmente costosa.
- **A\* Search**: Utilizzando l'euristica, può trovare soluzioni ottimali in termini di costo totale più efficacemente.

---

## 5. Simulazione dell'Esecuzione

Creeremo una sequenza di immagini che mostrano l'agente mentre esegue le azioni del piano trovato.

```python
def simulate_execution(initial_state, actions):
    state = initial_state.copy()
    frames = []

    for action in actions:
        state = problem.result(state, action)
        grid_display = state['grid'].copy()
        agent_pos = state['agent_pos']
        grid_display[agent_pos] = -1  # Codice per visualizzare l'agente

        frames.append(grid_display)

    return frames

frames = simulate_execution(initial_state, actions)

# Visualizzazione della simulazione
for i, frame in enumerate(frames):
    plt.imshow(frame, cmap='viridis')
    plt.title(f'Passo {i+1}')
    plt.show()
```

*Nota: La visualizzazione utilizza una mappa di colori per distinguere i diversi colori delle celle e la posizione dell'agente.*

---

## Conclusioni

In questa lezione, abbiamo:

- Definito formalmente il problema di Uniform Coloring su una griglia.
- Implementato le classi necessarie per eseguire la ricerca nello spazio degli stati.
- Utilizzato tecniche di visione artificiale e di apprendimento automatico per interpretare l'input da immagini.
- Applicato algoritmi di ricerca informati e non informati per risolvere il problema.
- Simulato l'esecuzione del piano d'azione trovato.

Questo approccio può essere esteso e migliorato, ad esempio ottimizzando l'euristica utilizzata, gestendo una più ampia varietà di input o implementando interfacce utente più avanzate per l'interazione con il sistema.

---

# Domande Frequenti

**1. Come posso migliorare l'accuratezza del riconoscimento delle immagini?**

- Puoi addestrare un modello più complesso o utilizzare reti neurali convoluzionali più profonde. Inoltre, assicurati di avere un dataset di training che rappresenti bene i dati che il modello vedrà in produzione.

**2. È possibile utilizzare altri algoritmi di ricerca?**

- Certamente! Puoi sperimentare con altri algoritmi come la ricerca Best-First, la ricerca a costo uniforme o algoritmi genetici, a seconda delle caratteristiche specifiche del problema.

**3. Come posso gestire griglie di dimensioni maggiori?**

- Potrebbe essere necessario ottimizzare il codice, utilizzare rappresentazioni degli stati più efficienti e implementare euristiche più forti per mantenere i tempi di calcolo ragionevoli.

---

# Prossimi Passi

- **Estensione del Problema**: Introduci nuovi vincoli o obiettivi per rendere il problema più complesso e interessante.
- **Interfaccia Utente**: Sviluppa un'interfaccia grafica per facilitare l'interazione con il sistema.
- **Apprendimento Rinforzato**: Esplora l'utilizzo di algoritmi di apprendimento automatico che permettono all'agente di imparare autonomamente la migliore strategia.

---

Grazie per l'attenzione!