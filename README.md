Predicción de Precios de Acciones

Esta aplicación de escritorio predice el precio al cierre del dia actual de las acciones utilizando modelos de redes neuronales
LSTM (Long Short-Term Memory).
La aplicación obtiene datos históricos de acciones desde la API de Alpha Vantage, calcula varios indicadores técnicos y
utiliza estos datos para entrenar un modelo de predicción.
Esta aplicación fue desarrollada con fines educativos y no como una herramienta financiera definitiva.


Características:

Entrada de Ticker de acción y clave API:
Introduce el Ticker de la acción y la clave API para obtener datos históricos.
Cálculo de indicadores técnicos: Calcula EMA, VWMA, ADX y el oscilador estocástico.
Modelo LSTM: Entrena un modelo LSTM para predecir los precios de las acciones.
Visualización de predicciones: Muestra las predicciones de precios en una gráfica e informa el valor estimado al final
del dia y su variación porcentual.
Guardar predicciones: Guarda las predicciones en un archivo CSV que puede ser consultado desde el GUI.
Requisitos
Python 3.6 o superior
Bibliotecas de Python: requests, pandas, numpy, scikit-learn, tensorflow, tkinter, matplotlib
Instalación
Clona el repositorio o descarga los archivos del proyecto.

Instala las dependencias necesarias usando pip:

pip install requests pandas numpy scikit-learn tensorflow matplotlib

Uso
Obtener una clave API de Alpha Vantage:

Regístrate en Alpha Vantage y obtén una clave API gratuita.
https://www.alphavantage.co/support/#api-key

Ejecutar la aplicación:

Abre una terminal y navega al directorio del proyecto.
Ejecuta el archivo main.py


Introducir datos:

Introduce el Ticker de la acción y tu clave API en los campos correspondientes.
Haz clic en "Calcular" para iniciar la predicción.

Visualizar y guardar resultados:

La predicción del precio de la acción y el porcentaje de cambio se mostrarán en la interfaz.
La gráfica de las predicciones se mostrará en la parte inferior de la ventana.
La aplicación  guardar las predicciones en un archivo CSV llamado prediccion.csv.

Archivos del Proyecto
main.py: Interfaz gráfica de usuario (GUI) y lógica de la aplicación.
stock_predictor.py: Clase principal para obtener datos de acciones, preparar datos, entrenar el modelo y hacer predicciones.
utils.py: Funciones auxiliares para calcular indicadores técnicos.

Notas
Asegúrate de que la clave API de Alpha Vantage sea válida y que el símbolo de la acción esté correctamente ingresado.
La aplicación guarda los datos en formato Json para reducir el numero de llamadas. Vantage ofrece 25 requests al dia,
por lo que la aplicación solo realiza la llamada una vez por dia por ticker.
El modelo utiliza datos históricos para realizar predicciones y puede no ser preciso en todos los casos.

Contribuciones
¡Las contribuciones son bienvenidas! Si tienes alguna mejora o corrección, por favor abre un issue o envía un pull request.

Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

