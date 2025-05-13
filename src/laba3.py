import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, 
                            QLineEdit, QPushButton, QHBoxLayout, QFormLayout, 
                            QMessageBox, QDialog, QSpinBox, QComboBox, QGroupBox,
                            QCheckBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class PlotWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Графики сигналов")
        self.setGeometry(200, 200, 750, 600)
        
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

# class SignalApp(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.add_template_frequency_input = 10
#         self.initUI()

#     def initUI(self):
class SignalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.add_template_frequency_input = 10

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.setWindowTitle("Сегментация зашумленных сигналов")
        self.setGeometry(100, 100, 600, 600)

        # Основной виджет и layout
        # self.central_widget = QWidget()
        # self.setCentralWidget(self.central_widget)
        # self.layout = QVBoxLayout(self.central_widget)

        # Поля для ввода параметров
        self.form_layout = QFormLayout()
        
        # Добавляем выбор типа шума
        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItems(["Гауссовский", "Равномерный", "Импульсный", "Экспоненциальный"])
        self.form_layout.addRow("Тип шума:", self.noise_type_combo)
        
        self.duration_input = QLineEdit("1.0")
        self.sampling_rate_input = QLineEdit("1000")
        self.noise_amplitude_input = QLineEdit("1.0")
        self.segments_per_template_input = QSpinBox()
        self.segments_per_template_input.setMinimum(1)
        self.segments_per_template_input.setValue(1)
        
        # Чекбокс для случайного заполнения
        self.random_fill_checkbox = QCheckBox("Случайное заполнение сегментами(абсолютное)")
        self.random_fill_checkbox.setChecked(False)
        
        self.form_layout.addRow("Длительность сигнала (сек):", self.duration_input)
        self.form_layout.addRow("Частота дискретизации (Гц):", self.sampling_rate_input)
        self.form_layout.addRow("Амплитуда шума:", self.noise_amplitude_input)
        self.form_layout.addRow("Сегментов на шаблон:", self.segments_per_template_input)
        self.form_layout.addRow(self.random_fill_checkbox)

        # Поля для ввода шаблонов
        self.templates_layout = QVBoxLayout()
        self.templates_layout.setContentsMargins(0, 0, 0, 0)  
        self.templates_layout.setSpacing(2)  
        self.add_template_button = QPushButton("Добавить шаблон")
        self.add_template_button.clicked.connect(self.add_template)
        self.templates_layout.addWidget(self.add_template_button)
        self.templates = []  # Хранит данные о шаблонах

        # Кнопки управления
        self.generate_button = QPushButton("Сгенерировать сигнал")
        self.generate_button.clicked.connect(self.generate_signal)
        self.segment_button = QPushButton("Выполнить сегментацию")
        self.segment_button.clicked.connect(self.segment_signal)

        # График
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        
        self.layout.addLayout(self.form_layout)
        self.layout.addLayout(self.templates_layout)
        self.layout.addWidget(self.generate_button)
        self.layout.addWidget(self.segment_button)
        self.layout.addWidget(self.canvas)

    def add_template(self):
        # Создаем группу для шаблона
        template_group = QGroupBox(f"Шаблон {len(self.templates)+1}")
        template_layout = QVBoxLayout(template_group)
        
        # Кнопка для добавления компоненты
        add_component_button = QPushButton("Добавить компоненту")
        add_component_button.clicked.connect(lambda: self.add_component(template_group))
        
        # Кнопка удаления шаблона
        delete_button = QPushButton("Удалить шаблон")
        delete_button.setStyleSheet("QPushButton { color: red; }")
        delete_button.clicked.connect(lambda: self.remove_template(template_group))
        
        # Layout для кнопок управления шаблоном
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(add_component_button)
        buttons_layout.addWidget(delete_button)
        
        template_layout.addLayout(buttons_layout)
        
        # Список компонент шаблона
        components_layout = QVBoxLayout()
        components_layout.setContentsMargins(10, 5, 5, 5)
        template_layout.addLayout(components_layout)
        
        # Сохраняем данные шаблона
        template_data = {
            'group': template_group,
            'components_layout': components_layout,
            'components': []
        }
        self.templates.append(template_data)
        self.templates_layout.addWidget(template_group)
        
        # Добавляем первую компоненту по умолчанию
        self.add_component(template_group)

    def add_component(self, template_group):
        # Находим данные шаблона
        template_data = next(t for t in self.templates if t['group'] == template_group)
        components_layout = template_data['components_layout']
        
        # Создаем виджет компоненты
        component_widget = QWidget()
        component_layout = QHBoxLayout(component_widget)
        component_layout.setContentsMargins(0, 0, 0, 0)
        
        # Поля ввода параметров компоненты
        amplitude_input = QLineEdit("1.0")
        amplitude_input.setFixedWidth(60)
        frequency_input = QLineEdit(str(self.add_template_frequency_input))
        frequency_input.setFixedWidth(60)
        self.add_template_frequency_input += 10
        phase_input = QLineEdit("0")
        phase_input.setFixedWidth(60)
        
        # Кнопка удаления компоненты
        delete_button = QPushButton("×")
        delete_button.setFixedSize(20, 20)
        delete_button.setStyleSheet("QPushButton { font-weight: bold; color: red; }")
        
        # Добавляем элементы
        component_layout.addWidget(QLabel("Амплитуда:"))
        component_layout.addWidget(amplitude_input)
        component_layout.addWidget(QLabel("Частота:"))
        component_layout.addWidget(frequency_input)
        component_layout.addWidget(QLabel("Фаза(град):"))
        component_layout.addWidget(phase_input)
        component_layout.addWidget(delete_button)
        
        # Сохраняем данные компоненты
        component_data = {
            'widget': component_widget,
            'inputs': (amplitude_input, frequency_input, phase_input)
        }
        template_data['components'].append(component_data)
        components_layout.addWidget(component_widget)
        
        # Подключаем кнопку удаления
        delete_button.clicked.connect(lambda: self.remove_component(template_data, component_data))

    def remove_component(self, template_data, component_data):
        template_data['components'].remove(component_data)
        template_data['components_layout'].removeWidget(component_data['widget'])
        component_data['widget'].deleteLater()
        
        # Если это последняя компонента, удаляем весь шаблон
        if len(template_data['components']) == 0:
            self.remove_template(template_data['group'])

    def remove_template(self, template_group):
        template_data = next(t for t in self.templates if t['group'] == template_group)
        self.templates_layout.removeWidget(template_group)
        template_group.deleteLater()
        self.templates.remove(template_data)

    def generate_signal(self):
        try:
            if len(self.templates) == 0:
                QMessageBox.warning(self, "Ошибка", "Добавьте хотя бы один шаблон!")
                return

            duration = float(self.duration_input.text())
            sampling_rate = float(self.sampling_rate_input.text())
            segments_per_template = self.segments_per_template_input.value()
            random_fill = self.random_fill_checkbox.isChecked()
            
            t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
            self.t = t
            
            # Генерация шаблонов (каждый шаблон может состоять из нескольких компонент)
            self.template_signals = []
            for template in self.templates:
                template_signal = np.zeros_like(t)
                for component in template['components']:
                    amplitude_input, frequency_input, phase_input = component['inputs']
                    
                    try:
                        A = float(amplitude_input.text())
                        f = float(frequency_input.text())
                        phi = float(phase_input.text()) * np.pi / 180
                        
                        # Добавляем компоненту к шаблону
                        template_signal += A * np.sin(2 * np.pi * f * self.t + phi)
                        
                    except ValueError:
                        QMessageBox.warning(self, "Ошибка", 
                                          f"Проверьте параметры компоненты (A={amplitude_input.text()}, "
                                          f"f={frequency_input.text()}, φ={phase_input.text()})")
                        return
                
                self.template_signals.append(template_signal)

            # Формирование исходного сигнала
            self.signal = np.zeros_like(t)
            total_segments = len(self.template_signals) * segments_per_template
            segment_length = len(t) // total_segments
            
            if random_fill:
                # Случайное заполнение сегментами
                segments_sequence = []
                for _ in range(total_segments):
                    segments_sequence.append(np.random.randint(0, len(self.template_signals)))
            else:
                # Регулярное заполнение (как было раньше)
                segments_sequence = [i % len(self.template_signals) for i in range(total_segments)]
            
            # Распределение шаблонов согласно выбранной последовательности
            for seg_num in range(total_segments):
                template_idx = segments_sequence[seg_num]
                start = seg_num * segment_length
                end = (seg_num + 1) * segment_length if seg_num != total_segments - 1 else len(t)
                self.signal[start:end] = self.template_signals[template_idx][start:end]

            # Генерация шума в зависимости от выбранного типа
            noise_amplitude = float(self.noise_amplitude_input.text())
            noise_type = self.noise_type_combo.currentText()
            
            if noise_type == "Гауссовский":
                self.noise = noise_amplitude * np.random.normal(size=len(t))
            elif noise_type == "Равномерный":
                self.noise = noise_amplitude * (np.random.rand(len(t)) - 0.5) * 2
            elif noise_type == "Импульсный":
                self.noise = noise_amplitude * np.random.choice([-1, 0, 1], size=len(t), p=[0.05, 0.9, 0.05])
            elif noise_type == "Экспоненциальный":
                self.noise = noise_amplitude * np.random.exponential(scale=0.5, size=len(t))
                self.noise *= np.random.choice([-1, 1], size=len(t))  # Добавляем отрицательные значения
            
            self.noisy_signal = self.signal + self.noise

            # Отображение в новом окне
            self.plot_window = PlotWindow(self)
            self.plot_window.figure.clear()
            
            ax1 = self.plot_window.figure.add_subplot(211)
            ax1.plot(t, self.signal)
            ax1.set_title("Исходный сигнал (без шума)")
            ax1.set_xlabel("Время (сек)")
            ax1.set_ylabel("Амплитуда")
            ax1.grid(True)

            ax2 = self.plot_window.figure.add_subplot(212)
            ax2.plot(t, self.noisy_signal)
            ax2.set_title(f"Зашумленный сигнал ({noise_type} шум)")
            ax2.set_xlabel("Время (сек)")
            ax2.set_ylabel("Амплитуда")
            ax2.grid(True)

            self.plot_window.figure.tight_layout()
            self.plot_window.canvas.draw()
            self.plot_window.show()

            # Сохраняем последовательность сегментов для использования при сегментации
            self.segments_sequence = segments_sequence

        except ValueError as e:
            QMessageBox.warning(self, "Ошибка", f"Некорректный ввод: {e}")

    def segment_signal(self):
        if not hasattr(self, 'noisy_signal'):
            QMessageBox.warning(self, "Ошибка", "Сначала сгенерируйте сигнал!")
            return

        # Вычисление корреляции для каждого шаблона
        self.correlations = []
        self.normalized_correlations = []
        for template in self.template_signals:
            corr = correlate(self.noisy_signal, template, mode='same')
            self.correlations.append(corr)
            # Нормализованная корреляция для отображения
            norm_corr = corr / (np.linalg.norm(self.noisy_signal) * np.linalg.norm(template))
            self.normalized_correlations.append(norm_corr)

        # Определение границ сегментов
        segments_per_template = self.segments_per_template_input.value()
        total_segments = len(self.template_signals) * segments_per_template
        segment_length = len(self.t) // total_segments
        
        self.segment_boundaries = []
        self.segment_types = []
        
        # Используем сохраненную последовательность сегментов
        for seg_num in range(total_segments):
            start = seg_num * segment_length
            end = (seg_num + 1) * segment_length if seg_num != total_segments - 1 else len(self.t)
            
            # Определяем тип шаблона для текущего сегмента
            template_idx = self.segments_sequence[seg_num]
            self.segment_boundaries.append((start, end))
            self.segment_types.append(template_idx)

        # Визуализация результатов
        self.segmentation_window = PlotWindow(self)
        self.segmentation_window.figure.clear()
        
        # Создаем 2 подграфика - один над другим
        ax1 = self.segmentation_window.figure.add_subplot(211)  # График сегментации
        ax2 = self.segmentation_window.figure.add_subplot(212)  # График ВКФ
        
        colors = plt.colormaps.get_cmap('tab10').resampled(len(self.template_signals))
        used_labels = set()  # Для отслеживания уже использованных меток
        
        # 1. График сегментации
        for (start, end), seg_type in zip(self.segment_boundaries, self.segment_types):
            # Определяем, нужно ли добавлять метку в легенду
            label = f'Шаблон {seg_type+1}' if f'Шаблон {seg_type+1}' not in used_labels else ""
            if label:
                used_labels.add(f'Шаблон {seg_type+1}')
            
            ax1.plot(self.t[start:end], self.noisy_signal[start:end], 
                   color=colors(seg_type), 
                   label=label)
            
            # Подпись в середине сегмента
            mid = (start + end) // 2
            y_pos = np.max(self.noisy_signal[start:end]) * 1.05
            ax1.text(self.t[mid], y_pos, 
                   f'Шаблон {seg_type+1}', 
                   ha='center', va='bottom', 
                   color=colors(seg_type),
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            # Вертикальные линии границ (кроме первой)
            if start > 0:
                ax1.axvline(self.t[start], color='r', linestyle='--', alpha=0.7)

        ax1.set_title("Результаты сегментации сигнала")
        ax1.set_xlabel("Время (сек)")
        ax1.set_ylabel("Амплитуда")
        ax1.grid(True)
        
        # 2. График ВКФ
        for i, norm_corr in enumerate(self.normalized_correlations):
            ax2.plot(self.t, norm_corr, color=colors(i), label=f'ВКФ шаблона {i+1}')
        
        # Отметка границ сегментов на графике ВКФ
        for boundary in self.segment_boundaries[1:-1]:
            ax2.axvline(self.t[boundary[0]], color='r', linestyle='--', alpha=0.3)
        
        ax2.set_title("Нормализованная взаимная корреляционная функция (ВКФ)")
        ax2.set_xlabel("Время (сек)")
        ax2.set_ylabel("Корреляция")
        ax2.grid(True)
        ax2.legend()
        
        # Настройка легенды для графика сегментации
        handles1, labels1 = ax1.get_legend_handles_labels()
        if handles1:
            ax1.legend(handles1, labels1, loc='upper right')
        
        # Автоматическая настройка масштаба
        ax1.set_ylim(np.min(self.noisy_signal)*1.1, np.max(self.noisy_signal)*1.2)
        self.segmentation_window.figure.tight_layout()
        self.segmentation_window.canvas.draw()
        self.segmentation_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalApp()
    window.show()
    sys.exit(app.exec_())
