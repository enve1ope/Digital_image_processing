import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch
from scipy.fft import fft, fftfreq
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout, QFormLayout, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class SignalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Спектральный и корреляционный анализ сигналов")
        self.setGeometry(100, 100, 1000, 800)

        # Основной виджет и layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Поля для ввода параметров сигнала
        self.form_layout = QFormLayout()
        self.duration_input = QLineEdit("1.0")
        self.sampling_rate_input = QLineEdit("512")
        self.noise_amplitude_input = QLineEdit("20")
        self.form_layout.addRow("Длительность сигнала (сек):", self.duration_input)
        self.form_layout.addRow("Частота дискретизации (Гц):", self.sampling_rate_input)
        self.form_layout.addRow("Амплитуда шума:", self.noise_amplitude_input)

        # Поля для ввода гармоник
        self.harmonics_layout = QVBoxLayout()
        self.add_harmonic_button = QPushButton("Добавить гармонику")
        self.add_harmonic_button.clicked.connect(self.add_harmonic)
        self.harmonics_layout.addWidget(self.add_harmonic_button)
        self.harmonics = []

        # Кнопки для генерации сигнала и анализа
        self.generate_button = QPushButton("Сгенерировать сигнал")
        self.generate_button.clicked.connect(self.generate_signal)
        self.analyze_button = QPushButton("Анализировать сигнал")
        self.analyze_button.clicked.connect(self.analyze_signal)

        # График
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addLayout(self.form_layout)
        self.layout.addLayout(self.harmonics_layout)
        self.layout.addWidget(self.generate_button)
        self.layout.addWidget(self.analyze_button)
        self.layout.addWidget(self.canvas)

    def add_harmonic(self):
        harmonic_layout = QHBoxLayout()
        amplitude_input = QLineEdit("15")
        frequency_input = QLineEdit("10")
        phase_input = QLineEdit("0")
        harmonic_layout.addWidget(QLabel("Амплитуда:"))
        harmonic_layout.addWidget(amplitude_input)
        harmonic_layout.addWidget(QLabel("Частота (Гц):"))
        harmonic_layout.addWidget(frequency_input)
        harmonic_layout.addWidget(QLabel("Фаза (град):"))
        harmonic_layout.addWidget(phase_input)
        self.harmonics.append((amplitude_input, frequency_input, phase_input))
        self.harmonics_layout.addLayout(harmonic_layout)

    def generate_signal(self):
        try:
            duration = float(self.duration_input.text())
            sampling_rate = float(self.sampling_rate_input.text())
            t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
            signal = np.zeros_like(t)
            print(1)
            for amplitude_input, frequency_input, phase_input in self.harmonics:
                A = float(amplitude_input.text())
                f = float(frequency_input.text())
                try:
                    phase = float(phase_input.text())
                    phi = phase * np.pi / 180
                except ValueError:
                    print("Ошибка: введите числовое значение для фазы")
                signal += A * np.sin(2 * np.pi * f * t + phi)

            self.signal = np.where(np.abs(signal) < 1e-10, 0, signal)
            self.t = t

            # Добавление шума
            noise_amplitude = float(self.noise_amplitude_input.text())
            self.noise = noise_amplitude * np.random.normal(size=len(t))
            self.noisy_signal = self.signal + self.noise

            # Очистка и построение графиков
            self.figure.clear()
            ax1 = self.figure.add_subplot(211)
            ax1.plot(t, self.signal)
            ax1.set_title("Детерминированный сигнал")
            ax1.set_xlabel("Время (сек)")
            ax1.set_ylabel("Амплитуда")
            ax1.grid(True)

            ax2 = self.figure.add_subplot(212)
            ax2.plot(t, self.noisy_signal)
            ax2.set_title("Случайный сигнал (детерминированный + шум)")
            ax2.set_xlabel("Время (сек)")
            ax2.set_ylabel("Амплитуда")
            ax2.grid(True)

            self.figure.subplots_adjust(hspace=0.5)

            self.canvas.draw()

        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", f"Некорректный ввод: {e}")

    def analyze_signal(self):
        if not hasattr(self, 'signal'):
            QMessageBox.warning(self, "Ошибка", "Сначала сгенерируйте сигнал.")
            return

        # Спектральный анализ детерминированного сигнала
        N = len(self.t)
        frequencies = fftfreq(N, 1 / float(self.sampling_rate_input.text()))
        fft_values = fft(self.signal)
        fft_values_cleaned = np.where(np.abs(fft_values) < 1e-10, 0, fft_values)
        amplitude_spectrum = np.abs(fft_values_cleaned) / N#АЧХ
        phase_spectrum = np.angle(fft_values_cleaned)#ФЧХ
        
        # Визуализация АЧХ и ФЧХ
        self.figure.clear()
        ax1 = self.figure.add_subplot(311)
        ax1.plot(frequencies[:N // 2], amplitude_spectrum[:N // 2])
        ax1.set_title("Амплитудно-частотная характеристика (АЧХ)")
        ax1.set_xlabel("Частота (Гц)")
        ax1.set_ylabel("Амплитуда")
        ax1.grid(True)

        ax2 = self.figure.add_subplot(312)
        ax2.plot(frequencies[:N // 2], phase_spectrum[:N // 2])
        ax2.set_title("Фазо-частотная характеристика (ФЧХ)")
        ax2.set_xlabel("Частота (Гц)")
        ax2.set_ylabel("Фаза (рад)")
        ax2.grid(True)

        # Оценка СПМ для случайного сигнала
        f, Pxx = periodogram(self.noisy_signal, fs=float(self.sampling_rate_input.text()), scaling='spectrum')
        f_welch, Pxx_welch = welch(self.noisy_signal, fs=float(self.sampling_rate_input.text()), scaling='spectrum')

        ax3 = self.figure.add_subplot(313)
        ax3.plot(f, Pxx, label="Периодограмма")
        ax3.plot(f_welch, Pxx_welch, label="Метод Уэлча")
        ax3.set_title("Спектральная плотность мощности (СПМ)")
        ax3.set_xlabel("Частота (Гц)")
        ax3.set_ylabel("СПМ")
        ax3.legend()
        ax3.grid(True)

        self.figure.subplots_adjust(hspace=0.75)

        self.canvas.draw()

        # Корреляционный анализ
        def autocorrelation(x):
            result = np.correlate(x, x, mode='full')
            return result[result.size // 2:]

        acf_signal = autocorrelation(self.signal)
        acf_noisy_signal = autocorrelation(self.noisy_signal)

        # Визуализация АКФ
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(acf_signal)
        ax1.set_title("Автокорреляционная функция (АКФ) детерминированного сигнала")
        ax1.set_xlabel("Задержка")
        ax1.set_ylabel("АКФ")
        ax1.grid(True)

        ax2.plot(acf_noisy_signal)
        ax2.set_title("Автокорреляционная функция (АКФ) случайного сигнала")
        ax2.set_xlabel("Задержка")
        ax2.set_ylabel("АКФ")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalApp()
    window.show()
    sys.exit(app.exec_())
