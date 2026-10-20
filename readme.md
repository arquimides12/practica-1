# WeatherML: predicción de lluvia con Pipes and Filters

Aplicación web en Flask que entrena un modelo Random Forest a partir de un CSV
meteorológico y predice si lloverá. El procesamiento está organizado con el
estilo arquitectónico **Tuberías y Filtros (Pipes and Filters)**.

## Estructura

```
practica-1/
├── app.py                  # Solo rutas web (Flask). No contiene lógica de ML.
├── config.py               # Todos los valores configurables (variables de entorno)
├── filtros/                # Filtros independientes + la tubería que los une
│   ├── carga_dataset.py    # 1. Carga el CSV
│   ├── validar_dataset.py  # 2. Valida el dataset
│   ├── preparar_datos.py   # 3. Separa X / y y arma el preprocesamiento
│   ├── dividir_datos.py    # 4. Separa entrenamiento y prueba
│   ├── entrenar_modelo.py  # 5. Entrena el Random Forest
│   ├── evaluar_modelo.py   # 6. Calcula métricas y matriz de confusión
│   └── pipeline.py         # Tubería: lista ordenada de filtros
├── services/               # Lógica de la aplicación separada de la web
│   ├── training_service.py # Ejecuta la tubería y guarda el modelo
│   ├── prediction_service.py # Arma el formulario, valida y predice
│   ├── model_store.py      # Guarda / lee el modelo (.joblib)
│   └── errors.py           # UserError: errores con mensaje simple
├── templates/index.html    # Interfaz (carga, resultados y predicción)
├── public/css/styles.css   # Estilos
├── models/                 # Modelo entrenado
├── data/                   # CSV de ejemplo para pruebas
└── tests/                  # Pruebas unitarias (pytest)
```

## Cómo ejecutar

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
python app.py
```

Abrir <http://127.0.0.1:5001>, subir el CSV y pulsar **Entrenar y ver resultados**.
Debajo de las métricas aparece el formulario **Predecir lluvia**.

## Pruebas

```bash
python -m pytest -v
```

## Configuración (config.py)

Se cambia sin tocar el código, con variables de entorno:

| Variable | Valor por defecto | Para qué sirve |
|---|---|---|
| `WEATHERML_PORT` | 5001 | Puerto de la aplicación |
| `WEATHERML_TARGET` | rain | Columna objetivo |
| `WEATHERML_TEST_SIZE` | 0.20 | Porcentaje de datos de prueba |
| `WEATHERML_N_ESTIMATORS` | 100 | Árboles del Random Forest |
| `WEATHERML_RANDOM_STATE` | 42 | Semilla |
| `WEATHERML_MODEL_PATH` | models/weather_model.joblib | Dónde se guarda el modelo |
| `WEATHERML_SECRET_KEY` | (clave de desarrollo) | Clave de Flask |

Ejemplo en Windows: `set WEATHERML_TEST_SIZE=0.25` y luego `python app.py`.

---

# Reto de arquitectura (Paso 11)

## 1. ¿Cómo mejorar la interfaz y la experiencia de usuario?

### Mejoras de interfaz (UX)

- [x] Agregar el formulario de predicción (antes no existía).
- [x] Mostrar el resultado como "Sí lloverá" o "No lloverá", con su porcentaje de confianza.
- [x] Usar campos numéricos y listas desplegables en lugar de texto libre.
- [x] Mostrar mensajes de error simples, sin texto técnico.

### Mejoras de mantenibilidad

- [x] Pipeline: para agregar un filtro solo se añade a la lista y no se toca app.py.
- [x] config.py: los valores se cambian sin editar el código.
- [x] services/: la web queda separada de la lógica de Machine Learning.
- [x] tests/: cada filtro se prueba por separado.

## 2. Decisión arquitectónica (rol de arquitecto)

**Problema identificado.** `app.py` hacía todo: recibía la petición, llamaba a
cada filtro a mano, dividía los datos, guardaba el modelo y armaba la respuesta.
Agregar un filtro obligaba a editar `app.py`, los valores (semilla, % de prueba,
ruta del modelo) estaban fijos en el código, no había pruebas y los errores
mostraban el texto técnico de la excepción. Además no existía un formulario de
predicción: el usuario no podía usar el modelo entrenado.

**Atributos de calidad a mejorar.** Mantenibilidad (modificabilidad y
testabilidad) y usabilidad.

**Decisión.**
1. Una tubería explícita (`filtros/pipeline.py`): cada `Step` conecta un filtro
   con un contexto compartido indicando qué lee y qué escribe. La tubería es una
   lista; agregar un filtro es agregar un elemento.
2. Una capa de servicios (`services/`) entre la web y los filtros. `app.py`
   solo traduce HTTP ↔ servicios.
3. Configuración centralizada en `config.py`, leída de variables de entorno.
4. Errores de usuario tipados (`UserError`) con mensajes simples; los errores
   inesperados se registran en el log y el usuario ve un mensaje genérico.
5. Formulario de predicción generado a partir del modelo: los campos numéricos
   usan `type="number"` con límites y las variables categóricas se muestran como
   lista desplegable con textos en español.

**Alternativas consideradas.**
- *Dejar todo en `app.py` y solo añadir el formulario*: más rápido, pero mantiene
  el acoplamiento y no permite probar los filtros por separado.
- *Usar solo el `Pipeline` de scikit-learn*: no cubre carga, validación ni
  evaluación, que no son transformadores de scikit-learn.
- *Un framework de orquestación (Airflow, Prefect)*: excesivo para una práctica.

Se eligió la tubería propia + servicios porque respeta el estilo Pipes and
Filters de la práctica, no cambia el código de los filtros originales y
agrega muy poco código.

**Archivos modificados o creados.** `app.py`, `config.py` (nuevo),
`filtros/pipeline.py` (nuevo), `filtros/dividir_datos.py` (nuevo),
`filtros/entrenar_modelo.py` (parámetros configurables), `filtros/__init__.py`,
`services/` (nuevo), `templates/index.html`, `public/css/styles.css`,
`tests/` (nuevo), `data/` (nuevo).

**Impacto esperado y compromisos (trade-offs).**
- (+) Agregar o reordenar filtros sin tocar la web; cada filtro se prueba aislado.
- (+) El usuario obtiene una respuesta clara ("Sí lloverá", 91 %) y no puede
  enviar datos inválidos.
- (−) Más archivos y una capa extra de indirección que hay que entender.
- (−) El contexto compartido de la tubería usa nombres de texto; un error de
  nombre solo aparece al ejecutar (mitigado con las pruebas).
