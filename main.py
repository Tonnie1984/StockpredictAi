import tkinter as tk
from tkinter import ttk, messagebox
from stock_predictor import StockPredictor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd

class StockPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictor diario de acciones")
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.destroy)

    def create_widgets(self):
        # Introducir Ticker
        ttk.Label(self.root, text="Ticker").grid(row=0, column=0, padx=5, pady=5)
        self.symbol_entry = ttk.Entry(self.root)
        self.symbol_entry.grid(row=0, column=1, padx=5, pady=5)

        # Ingresar API
        ttk.Label(self.root, text="API Key").grid(row=1, column=0, padx=5, pady=5)
        self.api_key_entry = ttk.Entry(self.root)
        self.api_key_entry.grid(row=1, column=1, padx=5, pady=5)

        # Boton Calcular
        self.predict_button = ttk.Button(self.root, text="Calcular", command=self.run_prediction)
        self.predict_button.grid(row=2, columnspan=2, padx=5, pady=5)

        # Ver Predicciónes guardadas
        self.view_csv_button = ttk.Button(self.root, text="Predicciones guardadas", command=self.view_csv)
        self.view_csv_button.grid(row=3, columnspan=2, padx=5, pady=5)

        self.exit_button = ttk.Button(self.root, text="Salir", command=self.destroy)
        self.exit_button.grid(row=4, columnspan=2, padx=5, pady=5)

        self.prediction_result = tk.StringVar()
        ttk.Label(self.root, textvariable=self.prediction_result).grid(row=5, columnspan=2, padx=5, pady=5)


        self.canvas = None

    def destroy(self):
        if messagebox.askokcancel("Salir", "¿Está seguro que desea salir?"):
            self.root.quit()

    def run_prediction(self):
        symbol = self.symbol_entry.get()
        api_key = self.api_key_entry.get()
        predictor = StockPredictor(symbol=symbol, api_key=api_key)
        predictor.get_stock_data()
        predictor.prepare_data()
        X_train, X_test, y_train, y_test = predictor.split_data()

        model = predictor.build_model()
        predictor.train_model(model, X_train, y_train, X_test, y_test)


        scaler = predictor.get_scaler()
        X_scaled = scaler.fit_transform(predictor.data[['Open', 'High', 'Low', 'Price_Mean', 'EMA_50', 'EMA_200',
                                                        'VWMA_50', 'VWMA_200', 'ADX', 'Stochastic']].astype(float))
        next_day_prediction, percentage_change = predictor.predict_next_day(model, X_scaled)

        self.prediction_result.set(f"Predicción: {next_day_prediction:.2f}\nVariación Diaria: {percentage_change:.2f}%")


        self.plot_predictions(y_test, model.predict(X_test))

    def plot_predictions(self, y_test, predictions):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(y_test.values, color='blue', label='Precio Real de la acción')
        ax.plot(predictions.flatten(), color='red', label='Predicción del precio')
        ax.set_title('Predicción Diaria')
        ax.set_xlabel('Fechas')
        ax.set_ylabel('Precio')
        ax.legend()

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=5, columnspan=2, padx=5, pady=5)

    def view_csv(self):
        try:
            df = pd.read_csv('saved_predictions/prediccion.csv')
            self.show_csv(df)
        except FileNotFoundError:
            messagebox.showerror("Error", "CSV file not found")

    def show_csv(self, df):
        top = tk.Toplevel(self.root)
        top.title("Predicciones Guardadas")

        cols = list(df.columns)
        tree = ttk.Treeview(top, columns=cols, show='headings')

        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row))

        tree.pack(fill=tk.BOTH, expand=True)


root = tk.Tk()
app = StockPredictorApp(root)
root.mainloop()
