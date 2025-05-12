import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QFormLayout, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class SignalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Моделирование сигналов")
        self.setGeometry(100, 100, 800, 600)

        # Основной виджет и layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Поля для ввода параметров сигнала
        self.form_layout = QFormLayout()
        self.duration_input = QLineEdit("1.0")
        self.sampling_rate_input = QLineEdit("1000")
        self.noise_amplitude_input = QLineEdit("0.1")
        self.form_layout.addRow("Длительность сигнала (сек):", self.duration_input)
        self.form_layout.addRow("Частота дискретизации (Гц):", self.sampling_rate_input)
        self.form_layout.addRow("Амплитуда шума:", self.noise_amplitude_input)

        # Поля для ввода гармоник
        self.harmonics_layout = QVBoxLayout()
        self.add_harmonic_button = QPushButton("Добавить гармонику")
        self.add_harmonic_button.clicked.connect(self.add_harmonic)
        self.harmonics_layout.addWidget(self.add_harmonic_button)
        self.harmonics = []

        # Кнопки для генерации сигнала и вычисления характеристик
        self.generate_button = QPushButton("Сгенерировать сигнал")
        self.generate_button.clicked.connect(self.generate_signal)
        self.calculate_button = QPushButton("Вычислить характеристики")
        self.calculate_button.clicked.connect(self.calculate_characteristics)

        # График
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addLayout(self.form_layout)
        self.layout.addLayout(self.harmonics_layout)
        self.layout.addWidget(self.generate_button)
        self.layout.addWidget(self.calculate_button)
        self.layout.addWidget(self.canvas)

    def add_harmonic(self):
        harmonic_layout = QHBoxLayout()
        amplitude_input = QLineEdit("1.0")
        frequency_input = QLineEdit("5.0")
        phase_input = QLineEdit("0.0")
        harmonic_layout.addWidget(QLabel("Амплитуда:"))
        harmonic_layout.addWidget(amplitude_input)
        harmonic_layout.addWidget(QLabel("Частота (Гц):"))
        harmonic_layout.addWidget(frequency_input)
        harmonic_layout.addWidget(QLabel("Фаза (рад):"))
        harmonic_layout.addWidget(phase_input)
        self.harmonics.append((amplitude_input, frequency_input, phase_input))
        self.harmonics_layout.addLayout(harmonic_layout)

    def generate_signal(self):
        try:
            duration = float(self.duration_input.text())
            sampling_rate = float(self.sampling_rate_input.text())
            t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
            signal = np.zeros_like(t)

            for amplitude_input, frequency_input, phase_input in self.harmonics:
                A = float(amplitude_input.text())
                f = float(frequency_input.text())
                phi = float(phase_input.text())
                signal += A * np.sin(2 * np.pi * f * t + phi)

            self.signal = signal
            self.t = t

            # Добавление шума
            noise_amplitude = float(self.noise_amplitude_input.text())
            self.noise = noise_amplitude * np.random.normal(size=len(t))
            self.noisy_signal = self.signal + self.noise

            # Очистка и построение графиков
            self.figure.clear()
            ax1 = self.figure.add_subplot(211)
            ax1.plot(t, self.signal)
            ax1.set_title("Полигармонический сигнал")
            ax1.set_xlabel("Время (сек)")
            ax1.set_ylabel("Амплитуда")
            ax1.grid(True)

            ax2 = self.figure.add_subplot(212)
            ax2.plot(t, self.noisy_signal)
            ax2.set_title("Полигармонический сигнал с шумом")
            ax2.set_xlabel("Время (сек)")
            ax2.set_ylabel("Амплитуда")
            ax2.grid(True)

            self.figure.subplots_adjust(hspace=0.6)

            self.canvas.draw()

        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", f"Некорректный ввод: {e}")

    def calculate_characteristics(self):
        if not hasattr(self, 'signal'):
            QMessageBox.warning(self, "Ошибка", "Сначала сгенерируйте сигнал.")
            return

        # Характеристики детерминированного сигнала
        energy = np.sum(self.signal**2)
        average_power = np.mean(self.signal**2)
        max_amplitude = np.max(np.abs(self.signal))

        # Характеристики случайного сигнала
        mean_value = np.mean(self.noisy_signal)
        variance = np.var(self.noisy_signal)
        std_deviation = np.std(self.noisy_signal)

        # Вывод характеристик
        QMessageBox.information(self, "Характеристики сигналов",
            f"Детерминированный сигнал:\n"
            f"Энергия: {energy:.2f}\n"
            f"Средняя мощность: {average_power:.2f}\n"
            f"Максимальная амплитуда: {max_amplitude:.2f}\n\n"
            f"Случайный сигнал:\n"
            f"Математическое ожидание: {mean_value:.2f}\n"
            f"Дисперсия: {variance:.2f}\n"
            f"Среднее квадратическое отклонение: {std_deviation:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalApp()
    window.show()
    sys.exit(app.exec_())
