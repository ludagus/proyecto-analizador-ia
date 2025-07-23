# Requisitos:
# pip install kivy pyttsx3 matplotlib reportlab numpy scikit-learn

import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.core.image import Image as CoreImage
from kivy.utils import get_color_from_hex
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp
from kivy.uix.scrollview import ScrollView

import sqlite3
import threading
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import time

import pyttsx3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

Window.size = (900, 750)


class CodeAnalysisApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=dp(15), spacing=dp(15), **kwargs)

        with self.canvas.before:
            Color(*get_color_from_hex("#F9FAFB"))  # fondo muy claro
            self.rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self.update_rect, size=self.update_rect)

        self.engine = pyttsx3.init()

        self.conn = sqlite3.connect("code_analysis_history.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT,
            lines INT,
            complexity INT,
            time_taken REAL,
            errors INT,
            prediction TEXT,
            feedback TEXT
        )''')
        self.conn.commit()

        # Entrenamiento modelos
        X_train = np.array([
            [3, 1, 2.0, 0],
            [6, 2, 5.5, 1],
            [12, 5, 15.0, 5],
            [4, 1, 3.0, 0],
            [8, 3, 7.0, 2],
            [14, 6, 20.0, 6]
        ])
        y_train = ['Bajo', 'Medio', 'Alto', 'Bajo', 'Medio', 'Alto']
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(X_train, y_train)
        self.dt = DecisionTreeClassifier()
        self.dt.fit(X_train, y_train)

        # Entrada código dentro de scroll para mejor usabilidad
        scroll = ScrollView(size_hint=(1, 0.6))
        self.code_input = TextInput(
            hint_text="Ingrese código aquí...",
            font_size=16,
            multiline=True,
            background_color=get_color_from_hex("#FFFFFF"),
            foreground_color=get_color_from_hex("#333333"),
            padding=[dp(10), dp(10), dp(10), dp(10)],
            size_hint_y=None,
            height=dp(300),
            cursor_color=get_color_from_hex("#3498db")
        )
        scroll.add_widget(self.code_input)
        self.add_widget(scroll)

        # Botón analizar con estilo personalizado
        self.analyze_btn = Button(
            text="Analizar Código",
            size_hint=(1, None),
            height=dp(50),
            background_color=get_color_from_hex("#3498db"),
            color=(1,1,1,1),
            font_size=18,
            bold=True
        )
        self.analyze_btn.bind(on_press=self.analyze_code)
        self.add_widget(self.analyze_btn)

        # Resultados en labels con borde y fondo suave
        self.result_label = Label(
            text="Predicción: ",
            font_size=20,
            size_hint=(1, None),
            height=dp(30),
            color=get_color_from_hex("#2c3e50"),
            halign='left'
        )
        self.add_widget(self.result_label)

        self.feedback_label = Label(
            text="Feedback: ",
            font_size=16,
            size_hint=(1, None),
            height=dp(60),
            color=get_color_from_hex("#34495e"),
            halign='left'
        )
        self.add_widget(self.feedback_label)

        # Imagen gráfica con borde redondeado y tamaño fijo
        self.graph_img = Image(
            size_hint=(1, None),
            height=dp(240),
            allow_stretch=True,
            keep_ratio=True,
        )
        with self.graph_img.canvas.before:
            Color(0.9, 0.9, 0.9, 1)
            self.graph_border = Rectangle(pos=self.graph_img.pos, size=self.graph_img.size)
        self.graph_img.bind(pos=self.update_graph_border, size=self.update_graph_border)
        self.add_widget(self.graph_img)

        # Botones de exportación organizados con mejor espacio
        buttons_layout = BoxLayout(size_hint=(1, None), height=dp(45), spacing=dp(15))

        self.export_pdf_btn = Button(
            text="Exportar a PDF",
            background_color=get_color_from_hex("#27ae60"),
            color=(1,1,1,1),
            font_size=16,
            bold=True
        )
        self.export_pdf_btn.bind(on_press=self.export_pdf)

        self.export_txt_btn = Button(
            text="Exportar a TXT",
            background_color=get_color_from_hex("#2980b9"),
            color=(1,1,1,1),
            font_size=16,
            bold=True
        )
        self.export_txt_btn.bind(on_press=self.export_txt)

        self.graph_history_btn = Button(
            text="Mostrar gráfico histórico",
            background_color=get_color_from_hex("#8e44ad"),
            color=(1,1,1,1),
            font_size=16,
            bold=True
        )
        self.graph_history_btn.bind(on_press=self.show_historical_graph)

        buttons_layout.add_widget(self.export_pdf_btn)
        buttons_layout.add_widget(self.export_txt_btn)
        buttons_layout.add_widget(self.graph_history_btn)
        self.add_widget(buttons_layout)

        self.last_analysis = None

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def update_graph_border(self, *args):
        self.graph_border.pos = self.graph_img.pos
        self.graph_border.size = self.graph_img.size

    def calculate_complexity(self, code):
        lines = code.strip().split('\n')
        complexity = 0
        keywords = ['if ', 'for ', 'while ', 'elif ', 'else:']
        for line in lines:
            if any(k in line for k in keywords):
                complexity += 1
        return len(lines), complexity

    def analyze_code(self, instance):
        code = self.code_input.text.strip()
        if not code:
            self.result_label.text = "[ERROR] Ingrese código válido"
            return

        start_time = time.time()
        errors = 0
        try:
            compile(code, '<string>', 'exec')
            simulated_time = len(code.splitlines()) * 0.05
            time.sleep(simulated_time)
        except Exception:
            errors = 1
        time_taken = round(time.time() - start_time, 2)

        lines, complexity = self.calculate_complexity(code)
        features = np.array([[lines, complexity, time_taken, errors]])

        pred_knn = self.knn.predict(features)[0]
        pred_dt = self.dt.predict(features)[0]
        prediction = pred_knn if pred_knn == pred_dt else pred_dt
        feedback = self.generate_feedback(prediction)

        self.result_label.text = f"Predicción: {prediction}"
        self.feedback_label.text = f"Feedback: {feedback}"

        threading.Thread(target=self.speak_feedback, args=(feedback,), daemon=True).start()

        self.cursor.execute('''
            INSERT INTO analysis (code, lines, complexity, time_taken, errors, prediction, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (code, lines, complexity, time_taken, errors, prediction, feedback))
        self.conn.commit()

        self.last_analysis = {
            "code": code,
            "lines": lines,
            "complexity": complexity,
            "time_taken": time_taken,
            "errors": errors,
            "prediction": prediction,
            "feedback": feedback
        }

        self.show_graph(lines, complexity, time_taken, errors)

    def generate_feedback(self, prediction):
        if prediction == 'Bajo':
            return "Buen dominio. Seguimiento estándar recomendado."
        elif prediction == 'Medio':
            return "Se detectan algunas dificultades. Reforzar conceptos."
        else:
            return "Dificultades significativas. Intervención personalizada necesaria."

    def speak_feedback(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def show_graph(self, lines, complexity, time_taken, errors):
        plt.clf()
        categories = ['Líneas', 'Complejidad', 'Tiempo (s)', 'Errores']
        values = [lines, complexity, time_taken, errors]
        colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']

        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#f9fafb')  # fondo del gráfico suave
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.7)
        ax.set_title("Métricas del Análisis", fontsize=14, color='#2c3e50')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom', fontsize=10, color='#2c3e50')

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)

        img_data = buf.read()
        buf.close()

        b64_data = base64.b64encode(img_data)
        data = BytesIO(base64.b64decode(b64_data))
        im = CoreImage(data, ext='png')
        self.graph_img.texture = im.texture

    def export_pdf(self, instance):
        if not self.last_analysis:
            self.result_label.text = "[ERROR] No hay análisis para exportar."
            return

        filename = "reporte_analisis.pdf"
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 50, "Reporte de Análisis de Código")
        c.line(50, height - 55, width - 50, height - 55)

        lines = [
            "Código analizado:",
            self.last_analysis['code'],
            "",
            f"Líneas: {self.last_analysis['lines']}",
            f"Complejidad: {self.last_analysis['complexity']}",
            f"Tiempo (s): {self.last_analysis['time_taken']}",
            f"Errores: {self.last_analysis['errors']}",
            f"Predicción: {self.last_analysis['prediction']}",
            f"Feedback: {self.last_analysis['feedback']}"
        ]

        y = height - 80
        for line in lines:
            if len(line) > 90:
                chunks = [line[i:i+90] for i in range(0, len(line), 90)]
                for chunk in chunks:
                    c.drawString(50, y, chunk)
                    y -= 15
            else:
                c.drawString(50, y, line)
                y -= 20

        c.save()
        self.result_label.text = f"[INFO] Reporte PDF guardado como {filename}"

    def export_txt(self, instance):
        if not self.last_analysis:
            self.result_label.text = "[ERROR] No hay análisis para exportar."
            return

        filename = "reporte_analisis.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Reporte de Análisis de Código\n")
            f.write("=" * 40 + "\n\n")
            f.write("Código analizado:\n")
            f.write(self.last_analysis['code'] + "\n\n")
            f.write(f"Líneas: {self.last_analysis['lines']}\n")
            f.write(f"Complejidad: {self.last_analysis['complexity']}\n")
            f.write(f"Tiempo (s): {self.last_analysis['time_taken']}\n")
            f.write(f"Errores: {self.last_analysis['errors']}\n")
            f.write(f"Predicción: {self.last_analysis['prediction']}\n")
            f.write(f"Feedback: {self.last_analysis['feedback']}\n")

        self.result_label.text = f"[INFO] Reporte TXT guardado como {filename}"

    def show_historical_graph(self, instance):
        self.cursor.execute('''
            SELECT lines, complexity, time_taken, errors, prediction
            FROM analysis
            ORDER BY id DESC LIMIT 5
        ''')
        data = self.cursor.fetchall()

        if not data:
            popup = Popup(title="Gráfico histórico", content=Label(text="No hay datos para mostrar."), size_hint=(0.5, 0.3))
            popup.open()
            return

        labels = [f"Análisis {i+1}" for i in range(len(data))]
        lines = [row[0] for row in data]
        complexity = [row[1] for row in data]
        time_taken = [row[2] for row in data]
        errors = [row[3] for row in data]
        predictions = [row[4] for row in data]

        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#f9fafb')

        width = 0.2
        x = np.arange(len(labels))

        ax.bar(x - width, complexity, width, label='Complejidad', color='#2ecc71', edgecolor='black', linewidth=0.7)
        ax.bar(x, time_taken, width, label='Tiempo (s)', color='#e67e22', edgecolor='black', linewidth=0.7)
        ax.bar(x + width, errors, width, label='Errores', color='#e74c3c', edgecolor='black', linewidth=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Histórico Últimos 5 Análisis", fontsize=14, color='#2c3e50')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.legend()

        for i, pred in enumerate(predictions):
            max_val = max(complexity[i], time_taken[i], errors[i])
            ax.text(x[i], max_val + 0.5, pred, ha='center', fontsize=9, fontweight='bold', color='#2c3e50')

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)

        img_data = buf.read()
        buf.close()

        b64_data = base64.b64encode(img_data)
        data = BytesIO(base64.b64decode(b64_data))
        im = CoreImage(data, ext='png')

        img_widget = Image(size_hint=(1,1))
        img_widget.texture = im.texture
        popup = Popup(title="Gráfico Histórico", content=img_widget, size_hint=(0.9, 0.7))
        popup.open()


class AIAppMain(App):
    def build(self):
        return CodeAnalysisApp()


if __name__ == "__main__":
    AIAppMain().run()
