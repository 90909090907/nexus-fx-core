# NEXUS FX CORE v0.1

Proyecto nuevo e independiente para investigación cuantitativa de Forex. **No importa ni reutiliza código de la aplicación anterior.**

## Qué implementa esta primera capa

1. **FX Matrix**: sincroniza una red de pares.
2. **Latent Currency Engine**: modela cada retorno como diferencia entre dos estados latentes de moneda.
3. **Filtro de Kalman multivariado**: actualiza fuerza relativa e incertidumbre con gauge de suma cero.
4. **Velocity / Acceleration**: derivadas suavizadas del estado latente.
5. **Residual / Divergence Engine**: compara retorno observado con retorno reconstruido por la red.
6. **Lead/Lag Network**: busca relaciones adelantadas descriptivas entre pares.
7. **Triangular Consistency**: mide residuos logarítmicos entre triángulos FX disponibles.
8. **Tests sintéticos**: valida reconstrucción, identificación y triángulos.

## Qué NO hace todavía

No genera BUY/SELL. No conecta dinero real. No contiene todavía macro, calendario, opciones, CFTC, microestructura broker-grade, Hawkes, causalidad condicional, entropía cuántica ni Meta-Observer.

## Matemática central

Para cada par `BASE/QUOTE`:

```text
r_pair,t = s_BASE,t - s_QUOTE,t + epsilon_t
```

En forma matricial:

```text
y_t = H x_t + epsilon_t
x_t = phi x_(t-1) + w_t
```

Como las fuerzas monetarias son relativas, imponemos el gauge:

```text
sum_i x_i = 0
```

El filtro acepta datos incompletos y usa pseudoinversa para tolerar universos parcialmente redundantes.

## Ejecutar

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest -q
```

## Sobre la fuente de datos incluida

`data_yahoo.py` existe solo para investigación/prototipado y permite arrancar sin claves API. El núcleo no depende de Yahoo: recibe una matriz sincronizada de precios, por lo que en la siguiente fase podremos conectar un feed broker-grade sin reescribir el modelo matemático.

## Próxima fase propuesta

**v0.2 — Regime + Causal Graph**

- detección probabilística de régimen;
- change points;
- correlación parcial / VAR regularizado;
- información mutua y tests de estabilidad;
- grafo de propagación temporal;
- reliability score por arista;
- walk-forward para evitar elegir relaciones espurias.
