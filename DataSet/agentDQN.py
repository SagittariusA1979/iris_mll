from collections import deque
from torch import optim
import random
import numpy as np

# Hiperparametry treningu sieci DQN
LEARNING_RATE = 0.001
BATCH_SIZE = 16

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size        # ilość informacji dot. stanu środowiska
        self.action_size = action_size      # ilość akcji, które agent może wykonać
        self.discount_factor = 0.99         # współczynnik spadku wartości nagrody
        self.epsilon_greedy = 1.0           # początkowy współczynnik losowości (1 = 100% losowości)
        self.epsilon_greedy_min = 0.1       # minimalny współczynnik losowości
        self.epsilon_greedy_decay = 0.995   # zmniejszanie stopnia losowości co iterację o 5%
        self.memory = deque(maxlen=1000)    # kolekcja przechowująca 1000 ostatnich zdarzeń
        self.train_start = 500              # liczba zdarzeń, od której zaczynamy trenować model

        self.model = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    # Zapisuje podjętą akcję w danym stanie i jej skutki 
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    # Wybiera akcje dla danego stanu. Jeśli aktualnie model
    # nie eksploruje (wykonuje losową akcje) to wybierana jest
    # akcja o najlepszym potencjale (najwyższa wartość nagrody)
    def act(self, state):
        if np.random.rand() <= self.epsilon_greedy:
            return random.randrange(self.action_size)
        # unsqueeze zapewnia odpowiedni wymiar [batch_size, state_size]
        # PyTorch narzuca format danych treningowych w postaci tensora, który
        # w pierwszym wymiarze zawiera informację i ilości paczek a następnie same
        # dane treningowe, dlatego 'unsqueeze' rozszerza wymiar danych mimo tego, że
        # mamy tylko jedną paczkę w tej funkcji
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values_predicted = self.model(state)
        return torch.argmax(q_values_predicted).item()
    

    def replay(self):
        # Nie zaczynamy trenować modelu dopóki nie zbierzemy
        # minimalnej ilości danych w buforze memory
        if len(self.memory) < self.train_start:
            return
        
        data_batch = random.sample(self.memory, BATCH_SIZE) # Losujemy paczkę danych do treningu
        
        total_mse_loss = 0
        for state, action, reward, next_state, done in data_batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.FloatTensor([reward])
            discounted_reward = reward
            if not done:
                discounted_reward += self.discount_factor * torch.max(self.model(next_state))
            
            dqn_prediction = self.model(state)
            true_reward = dqn_prediction.clone()     # Tworzymy klon aby nadpisać wynik dla akcji niżej
            true_reward[action] = discounted_reward  # Nadpisujemy wartość nagrody dla wykonanej akcji
            
            loss = self.criterion(dqn_prediction, true_reward)
            
            self.optimizer.zero_grad()  # Zerujemy gradient
            loss.backward()             # Liczymy gradient
            self.optimizer.step()       # Aktualizujemy wagi sieci

            total_mse_loss += loss.item()
        
        # Jeśli nie doszliśmy do minimalnej wartości współczynnika
        # eksploracji to nadal go zmniejszamy z każdą iteracją
        if self.epsilon_greedy > self.epsilon_greedy_min:
            self.epsilon_greedy *= self.epsilon_greedy_decay
        
        return total_mse_loss / BATCH_SIZE # zwracamy średni błąd MSE