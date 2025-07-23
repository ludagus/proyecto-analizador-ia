# Requisitos:
# pip install kivy pyttsx3 matplotlib numpy scikit-learn

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from io import BytesIO
import base64
import time
import pyttsx3

class CodeAnalysisApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        self.engine = pyttsx3.init()

        self.code_input = TextInput(hint_text="Ingrese código aquí...", multiline=True, size_hint=(1, 0.5))
        self.add_widget(self.code_input)

        self.analyze_btn = Button(text="Analizar Código", size_hint=(1, 0.1))
        self.analyze_btn.bind(on_press=self.analyze_code)
        self.add_widget(self.analyze_btn)

        self.result_label = Label(text="Resultado", size_hint=(1, 0.1))
        self.add_widget(self.result_label)

        self.image = Image(size_hint=(1, 0.3))
        self.add_widget(self.image)

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

    def analyze_code(self, instance):
        code = self.code_input.text
        if not code.strip():
            self.result_label.text = "Ingrese código válido"
            return

        lines = code.strip().split('\n')
        complexity = sum(1 for line in lines if any(k in line for k in ['if', 'for', 'while', 'elif', 'else']))
        errors = 0
        try:
            compile(code, '<string>', 'exec')
        except Exception:
            errors = 1

        time_taken = round(len(lines) * 0.05, 2)
        features = np.array([[len(lines), complexity, time_taken, errors]])
        prediction = self.knn.predict(features)[0]

        self.result_label.text = f"Predicción: {prediction}"
        self.speak_feedback(prediction)
        self.show_graph(len(lines), complexity, time_taken, errors)

    def speak_feedback(self, pred):
        if pred == 'Bajo':
            msg = "Buen dominio. Seguimiento estándar recomendado."
        elif pred == 'Medio':
            msg = "Se detectan algunas dificultades."
        else:
            msg = "Dificultades significativas."
        self.engine.say(msg)
        self.engine.runAndWait()

    def show_graph(self, lines, complexity, time_taken, errors):
        plt.clf()
        categories = ['Líneas', 'Complejidad', 'Tiempo', 'Errores']
        values = [lines, complexity, time_taken, errors]
        plt.bar(categories, values, color=['#3498db', '#2ecc71', '#e67e22', '#e74c3c'])
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_data = base64.b64encode(buf.read())
        buf.close()
        data = BytesIO(base64.b64decode(image_data))
        self.image.texture = CoreImage(data, ext='png').texture

class AIAppMain(App):
    def build(self):
        return CodeAnalysisApp()

if __name__ == '__main__':
    AIAppMain().run()
