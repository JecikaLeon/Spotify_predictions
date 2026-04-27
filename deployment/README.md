# Disponibilizacion del modelo Spotify

Esta carpeta contiene los archivos para exportar el modelo entrenado a un binario `.pkl` sin compresion y usarlo en dos despliegues:

- Flask en EC2: `deployment/flask/model.pkl`
- AWS Lambda: `deployment/lambda/model.pkl`

## 1. Exportar el modelo

Desde la raiz del proyecto:

```bash
python deployment/export_model.py
```

El script entrena los modelos finales con los datos de entrenamiento preprocesados y guarda el mismo artefacto en ambas carpetas de despliegue:

```text
deployment/flask/model.pkl
deployment/lambda/model.pkl
```

El archivo se guarda con `pickle.dump` directamente, sin gzip, zip ni joblib comprimido. Esto evita errores de descompresion al cargar el modelo en AWS Lambda.

## 2. Probar Flask localmente

Instalar dependencias:

```bash
pip install -r deployment/flask/requirements.txt
```

Ejecutar la API:

```bash
python deployment/flask/app.py
```

Endpoint de salud:

```bash
curl http://localhost:5000/health
```

Endpoint de prediccion:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"records\":[{\"track_id\":\"x\",\"artists\":\"artist\",\"album_name\":\"album\",\"track_name\":\"song\",\"track_genre\":\"pop\",\"duration_ms\":210000,\"explicit\":false,\"danceability\":0.7,\"energy\":0.8,\"key\":1,\"loudness\":-5.0,\"mode\":1,\"speechiness\":0.04,\"acousticness\":0.2,\"instrumentalness\":0.0,\"liveness\":0.1,\"valence\":0.6,\"tempo\":120.0,\"time_signature\":4}]}"
```

Respuesta esperada:

```json
{
  "count": 1,
  "predictions": [42.0]
}
```

El valor exacto cambia segun el modelo entrenado.

## 3. Despliegue Flask en EC2

Copiar a EC2 la carpeta `deployment/flask` despues de ejecutar el exportador. La carpeta debe contener:

```text
app.py
model.pkl
model_runtime.py
requirements.txt
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar con Gunicorn:

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

## 4. Despliegue en AWS Lambda

Despues de exportar el modelo, empaquetar el contenido de `deployment/lambda` junto con sus dependencias:

```text
lambda_function.py
model.pkl
model_runtime.py
requirements.txt
```

El handler configurado en Lambda debe ser:

```text
lambda_function.lambda_handler
```

La funcion acepta un JSON con un registro individual, una lista de registros o un objeto con la llave `records`.

