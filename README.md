# Analizador de Código con Inteligencia Artificial

Este proyecto es una aplicación desarrollada con **Kivy** y **Python**, que permite analizar fragmentos de código fuente escritos por estudiantes y brindar una predicción sobre su nivel de dominio, junto con feedback personalizado. Utiliza algoritmos de inteligencia artificial y técnicas de análisis estático para brindar apoyo al aprendizaje de la programación.

## 🧠 Características

- Análisis automático del código ingresado
- Predicción del nivel de dominio (Bajo, Medio, Alto) con KNN y Decision Tree
- Visualización de métricas (líneas, complejidad, tiempo de ejecución, errores)
- Generación de reportes en formato **PDF** y **TXT**
- Historial de análisis almacenado en base de datos SQLite
- Exportación gráfica e informes con Matplotlib y ReportLab
- Feedback por voz con pyttsx3

## 🚀 Tecnologías utilizadas

- Python
- Kivy
- SQLite
- Matplotlib
- Scikit-learn
- pyttsx3
- ReportLab

## 📦 Requisitos de instalación

Antes de ejecutar la aplicación, instalá las siguientes dependencias:

```bash
pip install kivy pyttsx3 matplotlib numpy scikit-learn reportlab
