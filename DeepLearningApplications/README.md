# Aplicaciones del Deep Learning con PyTorch

Cinco notebooks independientes que muestran cómo se aplica el deep learning fuera de la visión por computador.
Cada uno construye el modelo desde cero en PyTorch, lo compara con una línea base y termina con ejercicios y
pistas para pasar a datos o librerías reales.

| # | Notebook | Dominio | Modelo | Datos | Idea clave |
|---|---|---|---|---|---|
| 1 | [01_Audio_Clasificacion_Sonidos](01_Audio_Clasificacion_Sonidos.ipynb) | Audio | CNN 2D sobre espectrogramas Mel + SpecAugment | Sintéticos (5 tipos de sonido) | El audio se convierte en una "imagen" tiempo × frecuencia |
| 2 | [02_Finanzas_Volatilidad_LSTM](02_Finanzas_Volatilidad_LSTM.ipynb) | Finanzas | LSTM | Mercado simulado con GARCH(1,1) (opcional: `yfinance`) | La dirección del precio no es predecible; la volatilidad sí. Evaluar sin fugas de información |
| 3 | [03_Biologia_Moleculas_GNN](03_Biologia_Moleculas_GNN.ipynb) | Química / biología | Graph Neural Network (*message passing*) | ESOL: solubilidad de 1128 moléculas (se descarga solo) | Una molécula es un grafo; parser de SMILES y batching de grafos sin librerías externas |
| 4 | [04_Texto_Transformer_Sentimiento](04_Texto_Transformer_Sentimiento.ipynb) | Texto (NLP) | Transformer encoder escrito bloque a bloque | Corpus de reseñas en español generado con reglas | Por qué la auto-atención resuelve la negación y el contraste, y la bolsa de palabras no |
| 5 | [05_ReinforcementLearning_CartPole](05_ReinforcementLearning_CartPole.ipynb) | Reinforcement Learning | Actor-crítico, REINFORCE con *baseline* | Entorno CartPole implementado desde cero | Aprender sólo a partir de recompensas |

## Resultados de referencia

Valores obtenidos al ejecutar los notebooks (GPU RTX 3060; todo corre también en CPU en pocos minutos):

| Notebook | Línea base | Modelo |
|---|---|---|
| Audio | — | 97 % de exactitud en test (5 clases) |
| Finanzas (volatilidad, MSE en log-vol.) | Histórica 20 días: 0.257 | LSTM: 0.220 (oráculo GARCH: 0.176) |
| Finanzas (dirección) | Clase mayoritaria: 51.5 % | LSTM: 51.5 % (como se esperaba) |
| Moléculas (RMSE log S) | Ecuación ESOL: 0.93 · MLP con descriptores: 0.87 | GNN: ≈ 0.50 (varía un poco entre ejecuciones en GPU) |
| Texto | Bolsa de palabras: 63 % | Transformer: 100 % |
| RL | Agente aleatorio: 23 pasos | Agente entrenado: 500/500 pasos |

## Requisitos

```bash
pip install torch torchaudio numpy pandas matplotlib networkx
```

- `networkx` sólo se usa para dibujar moléculas (opcional).
- El notebook 3 descarga ESOL (~100 KB) en `data/` la primera vez.
- No se necesitan `gymnasium`, `rdkit`, `torch_geometric` ni `transformers`; cada notebook explica cómo usarlos
  como siguiente paso.

## Orden sugerido

Los notebooks son independientes. Si se usan en clase, un orden natural es **Texto → Audio → Finanzas →
Moléculas → RL**: va de arquitecturas conocidas (Transformer, CNN, RNN) hacia estructuras de datos menos
habituales (grafos) y, al final, a un paradigma de aprendizaje distinto (recompensas en lugar de etiquetas).
