import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, periodogram, welch
from scipy.fft import fft, fftfreq
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFormLayout, QMessageBox, QSpinBox, QComboBox, QGroupBox, QCheckBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from laba1 import SignalApp as Lab1App
from laba2 import SignalApp as Lab2App
from laba3 import SignalApp as Lab3App

class UnifiedApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторные")
        self.setGeometry(100, 100, 1000, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tabs.addTab(Lab1App(), "Лабораторная 1")
        self.tabs.addTab(Lab2App(), "Лабораторная 2")
        self.tabs.addTab(Lab3App(), "Лабораторная 3")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UnifiedApp()
    window.show()
    sys.exit(app.exec_())