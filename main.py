from numpy import exp, array, random, dot
import data_module as dm
 
 
class NeuralNetwork():
    def __init__(self):
        # Заполнение генератора случайными числами, чтобы он генерировал те же числа
        # каждый раз при запуске программы.
        random.seed(1)
    
        # Модуляция одного нейрона с 7 входными подключениями и 1 выходным подключением.
        # Назначение случайного веса матрице 7 x 1 со значениями в диапазоне от -1 до 1
        self.synaptic_weights = 2 * random.random((7, 1)) - 1
    
    # Сигмоидальная функция, описывающая S-образную кривую.
    # Передача взвешенной суммы входных данных через эту функцию в
    # нормализованных от 0 до 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    # Производная сигмовидной функции.
    # Это градиент сигмовидной кривой.
    # Это показывает, уверенность в существующем весе.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # Обучение нейронной сети методом проб и ошибок.
    # Каждый раз корректировать синаптические веса.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Пропустить обучающий набор через нейронную сеть (отдельный нейрон).
            output = self.think(training_set_inputs)
    
            # Вычислить ошибку (Разница между желаемым результатом
            # и прогнозируемый результат).
            error = training_set_outputs - output
    
            # Умножьте ошибку на ввод и снова на градиент сигмовидной кривой.
            # Это означает, что менее достоверные веса корректируются больше.
            # Это означает, что входы, которые равны нулю, не вызывают изменения весов.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
    
            # Отрегулирование веса
            self.synaptic_weights += adjustment
    
    # Нейронная сеть думает..
    def think(self, inputs):
        # Передача входных данных через нейронную сеть (единственный нейрон)..
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
 
 
if __name__ == "__main__":
 
    # Изоляция нейронной сети с одним нейроном.
    neural_network = NeuralNetwork()
    
    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)
    
    # Обучающий набор. У нас есть 8 примера, каждый из 7 входных значений
    # и 1 выходное значение.
    training_set_inputs = array(dm.observation_vector)
    training_set_outputs = array(dm.required_response).T
    
    # Обучить нейронную сеть с помощью обучающего набора.
    # Сделать 10 000 раз и каждый раз вносить небольшие поправки.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)
    
    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)
    
    # Тестирование нейронной сети в новой ситуации.
    print(f"Considering new situation {dm.new_situation} -> ?: ")

    ann = neural_network.think(array(dm.new_situation))
    
    print(f"Probability of presence of water is - {round(*ann*100, 4)}%")

    
