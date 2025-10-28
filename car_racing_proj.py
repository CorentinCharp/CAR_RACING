import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2
import time
import random
import csv
import os

# Image preprocessing parameters
IMAGE_HEIGHT = 96 # Height the environment provides
IMAGE_WIDTH = 96 # Width the environment provides
NUM_STACKED_FRAMES = 4 # Stacking 4 images to give a sense of motion to the model 


class PreprocessWrapper(gym.ObservationWrapper):
    """ Wraps the environment and changes what the agent will see """
    def __init__(self, env):
        super(PreprocessWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(NUM_STACKED_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH), 
            dtype=np.uint8
        )
        # The double ended queue to stack up the 4 frames,
        # when the newest is appended, the oldest is dropped from the deque
        self.frames = deque(maxlen=NUM_STACKED_FRAMES)  
        

    # We take the raw frame and convert it from RGB to greyscale to make 
    # training faster from (96, 96, 3) to (96, 96, 1)
    def _to_grayscale(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return gray

    def _stack_frames(self):
        return np.stack(self.frames, axis=0)

    def observation(self, obs):
        gray_frame = self._to_grayscale(obs)
        self.frames.append(gray_frame)
        return self._stack_frames()

    # Called once every begining of an episode
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs) # Get the first raw frame from the environment
        gray_frame = self._to_grayscale(obs) # Converts this first frame to grayscale
        self.frames.clear() # Empties the double ended queue

        # Appends the first frame 4 times to stack it up to have the right shape
        for _ in range(NUM_STACKED_FRAMES):
            self.frames.append(gray_frame)
        return self._stack_frames(), info


# CNN based DQNetwork
# Defines the neural network architecture that learns to estimate Q-values from image inputs.
class DQNetwork_CNN(nn.Module):
    def __init__(self, action_size, hidden_size=512):
        super(DQNetwork_CNN, self).__init__()

        # Convolutional part of the network to process images
        # Processes the input image stack to extract spatial features, patterns, ...
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=NUM_STACKED_FRAMES, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Based on the infos extracted with the CNN, we can predict the Q-values
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
    # Defines the forward pass of the network (how data flows through the layers).
    def forward(self, x):
        x = x / 255.0 # Normalize
        x = self.conv_layers(x) # Passes the normalized input through the convolutional layers.
        x = self.fc_layers(x) # Passes the flattened features through the fully connected layers.
        return x


# Replay buffer acting as memory, a crucial component for stabilizing DQN training
# It stores past experiences (transitions) and allows the agent to sample random batches from them.
class ReplayBuffer:
    """ Simple replay buffer for storing experiences """
    def __init__(self, capacity):
        # deque to hold the experiences with a max length, when the buffer is full and a new experience is added,
        # the oldest experience is automatically removed from the other end.
        self.buffer = deque(maxlen=capacity)
    
    def store(self, state, action, reward, next_state, done):
        """ Add an experience to the buffer """
        state = state.astype(np.uint8) # Convert inputs to uint8 to save memory
        next_state = next_state.astype(np.uint8) 
        self.buffer.append((state, action, reward, next_state, done)) 
    
    def sample(self, batch_size):
        """ Sample a random batch of experiences from the buffer """
        batch = random.sample(self.buffer, batch_size) # This random batch can break temporal correlation
        
        states, actions, rewards, next_states, dones = zip(*batch) # Dezip the batch into separate lists every variable at any given time 
        
        # Convert to array to process it more efficiently
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        """ Get the current size of the buffer """
        return len(self.buffer) # To check if the buffer is full or not before a training
    
# This class implements the Deep Q-Network algorithm, incorporating key improvements
# like Experience Replay (using ReplayBuffer) and a Target Network for stability.
class DQNAgent:
    def __init__(self, action_size, learning_rate=1e-4, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01,
                 buffer_capacity=50000, batch_size=64, target_update_freq=10000):
        
        self.action_size = action_size
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_step_count = 0 # Counter for target network updates

        # Device setup 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device: {self.device}")

        # Create an instance of the ReplayBuffer class to store experiences
        self.memory = ReplayBuffer(buffer_capacity)
        
        # Create the primary policy network or Q-network.
        # It takes the state and outputs Q-values for each action.
        self.q_network_online = DQNetwork_CNN(action_size).to(self.device) 

        # Create the target Q-network.
        # This is a separate network used to provide stable targets during training.
        self.q_network_target = DQNetwork_CNN(action_size).to(self.device)
        
        # Initialize target network weights to match online network
        self.q_network_target.load_state_dict(self.q_network_online.state_dict())
        self.q_network_target.eval() # Put target network in evaluation mode
        
        # Create the Adam optimizer. It will adjust the weights of the 'q_network_online'.
        self.optimizer = optim.Adam(self.q_network_online.parameters(), lr=learning_rate) 

        # Define the loss function: Mean Squared Error (MSE).
        # This measures the difference between the Q-values predicted by the online network
        # and the target Q-values calculated using the target network.
        self.criterion = nn.MSELoss()

    # Selects an action based on the current state using the epsilon-greedy strategy.
    def choose_action(self, state):
        """ Selects action using epsilon-greedy policy: explore randomly or exploit best known action. """
        
        # Generate a random float between 0.0 and 1.0.
        # If the random number is less than epsilon, choose a random action.
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # If the random number is greater than or equal to epsilon, choose the best action predicted by the online network.
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        
        with torch.no_grad():
            # Pass the state tensor through the online Q-network to get Q-values for all actions.
            q_values = self.q_network_online(state_tensor)
        return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Stocke une transition dans la mémoire."""
        self.memory.store(state, action, reward, next_state, done)
    
    def train_agent(self):
        """ Trains the agent by sampling a batch from the replay buffer """

        # 1. If memorty has less than batch_size, do nothing
        if len(self.memory) < self.batch_size:
            return None # Not enough samples to train

        # 2. Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 3. Convert data into torch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 4. Calculer les Q-values actuelles (Q(s, a))
        # (utilise le réseau ONLINE)
        current_q_values = self.q_network_online(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # 5. Calculer les Q-values cibles (target)
        # (utilise le réseau TARGET)
        with torch.no_grad():
            # (Ici c'est DQN. Pour Double DQN comme code3, on utiliserait les deux réseaux)
            next_q_values_target = self.q_network_target(next_states)
            max_next_q = next_q_values_target.max(1)[0]
            
            # Formule de Bellman
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # 6. Calculer la perte (Loss)
        loss = self.criterion(current_q, target_q)
        
        # 7. Optimiser le réseau ONLINE
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 8. Mettre à jour le réseau TARGET (si c'est le moment)
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.update_target_network()
            
        return loss.item() # Retourne la valeur de la perte

    def update_target_network(self):
        """Copie les poids du réseau online vers le réseau target."""
        print("--- Mise à jour du Target Network ---")
        self.q_network_target.load_state_dict(self.q_network_online.state_dict())
    
    def decay_epsilon(self):
        """Diminue epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Sauvegarde l'état complet de l'agent (modèle, optimizer, epsilon)."""
        torch.save({
            'model_state_dict': self.q_network_online.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step_count': self.train_step_count
        }, filepath)
        print(f"Modèle sauvegardé sous : {filepath}")

    def load(self, filepath):
        """Charge l'état complet de l'agent."""
        
        # S'assurer de charger sur le bon appareil (CPU ou GPU)
        if not torch.cuda.is_available():
            checkpoint = torch.load(filepath, map_location='cpu')
        else:
            checkpoint = torch.load(filepath)
            
        self.q_network_online.load_state_dict(checkpoint['model_state_dict'])
        self.q_network_target.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step_count = checkpoint['train_step_count']
        
        self.q_network_online.eval() # Met le réseau en mode évaluation
        self.q_network_target.eval() # Met le réseau en mode évaluation
        print(f"Modèle chargé depuis : {filepath}. Epsilon restauré à {self.epsilon:.3f}")
        
def train_dqn(episodes=1000, render=False, learning_rate=1e-4, gamma=0.99,
              checkpoint_path=None, plot_save_path=None, resume=False):
    """ Train a DQN agent in the CarRacing-v3 environment """
    env = gym.make('CarRacing-v3', continuous=False, 
                   render_mode='human' if render else None)
    
    env = PreprocessWrapper(env)
    
    action_size = env.action_space.n
    
    # Create the new DQN agent
    agent = DQNAgent(action_size, 
                     learning_rate=learning_rate, 
                     discount_factor=gamma)
    
    start_episode = 0
    rewards_per_episode = []
    losses_per_episode = []

    rewards_path = f"{checkpoint_path}_rewards.npy" if checkpoint_path else None
    losses_path = f"{checkpoint_path}_losses.npy" if checkpoint_path else None

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            print(f"Reprise depuis le checkpoint : {checkpoint_path}")
            agent.load(checkpoint_path)
            
            # Charger les métriques sauvegardées
            rewards_per_episode = list(np.load(rewards_path))
            losses_per_episode = list(np.load(losses_path))
            start_episode = len(rewards_per_episode) # Repartir après le dernier épisode sauvegardé
            
            print(f"Reprise à l'épisode {start_episode}. Epsilon actuel: {agent.epsilon:.3f}")
        except Exception as e:
            print(f"Erreur lors du chargement du checkpoint, l'entraînement repart de zéro. Erreur : {e}")
            start_episode = 0 # repartir de zéro
            rewards_per_episode = []
            losses_per_episode = []
    
    if start_episode == 0: # Si on ne reprend pas, ou si la reprise a échoué
        print("Début d'un nouvel entraînement. Préchauffage de la mémoire...")
        WARMUP_STEPS = 100
        steps = 0
        while steps < WARMUP_STEPS:
            state, _ = env.reset()
            done = False
            while not done:
                action = env.action_space.sample() # Action aléatoire
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                steps += 1
                if steps >= WARMUP_STEPS:
                    break
    print(f"Mémoire préchauffée avec {len(agent.memory)} expériences.")

    CHECKPOINT_FREQ = 50  # Sauvegarde tous les 100 épisodes
    CHECKPOINT_PATH = "car_racing_checkpoint.pth"

    # Boucle d'entraînement principale
    for episode in range(start_episode, episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_losses = []
        done = False
        
        while not done:
            # 1. Choisir une action
            action = agent.choose_action(state)
            
            # 2. Agir dans l'environnement
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 3. Stocker l'expérience
            agent.store_transition(state, action, reward, next_state, done)
            
            # 4. Entraîner l'agent (en échantillonnant un batch)
            loss = agent.train_agent()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
        
        # 5. Décroissance d'epsilon à la fin de l'épisode
        agent.decay_epsilon()
        
        rewards_per_episode.append(total_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses_per_episode.append(avg_loss)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_per_episode[-10:])
            print(f"Épisode {episode + 1}/{episodes} | "
                  f"Récompense Moy (10 derniers): {avg_reward:.2f} | "
                  f"Perte Moy: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
            
        if (episode + 1) % CHECKPOINT_FREQ == 0 and checkpoint_path:
            agent.save(CHECKPOINT_PATH) # Sauvegarde le modèle, l'optimizer et l'epsilon
            
            # Sauvegarde les listes de métriques
            np.save(rewards_path, np.array(rewards_per_episode))
            np.save(losses_path, np.array(losses_per_episode))
            print(f"*** CHECKPOINT (Modèle + Métriques) SAUVEGARDÉ à l'épisode {episode + 1} ***")
            
            # Enregistre la vidéo
            record_test_video(agent, episode_number=(episode + 1))
            
            # Génère un graphique de progression
            if plot_save_path:
                plot_filename = plot_save_path.replace(".png", f"_ep{episode + 1}.png")
                plot_training_results(rewards_per_episode, losses_per_episode, save_path=plot_filename)
    
    env.close()
    print("=" * 60)
    print(f"Entraînement terminé pour LR={learning_rate}, Gamma={gamma}")
    
    return agent, rewards_per_episode, losses_per_episode

def plot_training_results(rewards, losses, save_path=None):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10)) # Keep 'fig' variable

    # Plot 1: Raw rewards
    axes[0, 0].plot(rewards, alpha=0.3, color='blue')
    axes[0, 0].set_title('Rewards per Episode', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Moving average rewards
    window = 50
    if len(rewards) >= window:
        # Calculate valid indices for x-axis
        x_axis = range(window - 1, len(rewards))
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(x_axis, moving_avg, color='green', linewidth=2) # Use x_axis
    else: # If not enough data, plot raw rewards
         axes[0, 1].plot(rewards, color='green', alpha=0.5)

    axes[0, 1].set_title(f'Moving Average Rewards (window={window})', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=900, color='r', linestyle='--', label='Solved Threshold')
    axes[0, 1].legend()

    # Plot 3: Training loss
    axes[1, 0].plot(losses, alpha=0.5, color='red')
    axes[1, 0].set_title('Training Loss per Episode', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Moving average loss
    if len(losses) >= window:
        # Calculate valid indices for x-axis
        x_axis_loss = range(window - 1, len(losses))
        moving_avg_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(x_axis_loss, moving_avg_loss, color='orange', linewidth=2) # Use x_axis_loss
    else: # If not enough data, plot raw losses
        axes[1, 1].plot(losses, color='orange', alpha=0.5)

    axes[1, 1].set_title(f'Moving Average Loss (window={window})', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Average Loss')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # --- CORRECTIONS HERE ---
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graphiques sauvegardés sous : {save_path}")
        plt.close(fig) # <-- ADD THIS to close the figure after saving
    else:
        # If no save_path is given (only at the end), show the plot
        plt.savefig('ddqn_per_training_results.png', dpi=150, bbox_inches='tight')
        print("Training plots saved to 'ddqn_per_training_results.png'")
        plt.show() # <-- KEEP plt.show() here for the final plot
        plt.close(fig) # Good practice to close even after showing
    # --- END OF CORRECTIONS ---

def record_test_video(agent, episode_number, video_folder_base='videos'):
    """
    Crée un environnement de test temporaire, enregistre 1 vidéo de l'agent
    en mode évaluation, puis ferme l'environnement.
    """
    print(f"\n--- Enregistrement vidéo de l'épisode {episode_number} ---")
    
    # Crée un dossier unique pour cette sauvegarde vidéo (ex: 'videos/ep_100')
    video_folder = os.path.join(video_folder_base, f"ep_{episode_number}")

    # 1. Créer l'environnement de test
    try:
        # 'rgb_array' est requis pour que RecordVideo fonctionne
        test_env = gym.make('CarRacing-v3', continuous=False, render_mode='rgb_array')
        
        # 2. Wrapper pour la vidéo.
        # Il crée le dossier 'video_folder' et enregistre
        # uniquement le premier épisode (trigger=...e == 0)
        test_env = gym.wrappers.RecordVideo(test_env, 
                                           video_folder, 
                                           episode_trigger=lambda e: e == 0, 
                                           name_prefix=f"dqn-agent-ep{episode_number}") 
        
        # 3. Wrapper pour le preprocessing (très important !)
        test_env = PreprocessWrapper(test_env)
    except Exception as e:
        print(f"Erreur lors de la création de l'env de test vidéo : {e}")
        return

    # Sauvegarde de l'epsilon actuel pour le restaurer après
    epsilon_backup = agent.epsilon
    agent.epsilon = 0.0 # Mode Évaluation (pas d'actions aléatoires)
    
    state, _ = test_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    
    print(f"Récompense du test vidéo : {total_reward:.2f}")
    
    # 4. Fermer l'environnement (ceci finalise l'enregistrement de la vidéo)
    test_env.close()
    
    # Restaurer l'epsilon d'entraînement
    agent.epsilon = epsilon_backup
    print(f"Vidéo de progression sauvegardée dans le dossier : {video_folder}")

def test_agent_visually(model_path, episodes=5):
    """
    Charge un modèle sauvegardé et le lance en mode 'human' pour le regarder.
    """
    print(f"\n" + "="*70)
    print(f"TEST VISUEL du modèle : {model_path}")
    print(f"Lancement de {episodes} épisodes...")
    
    # 1. Créer l'environnement en mode 'human'
    env = gym.make('CarRacing-v3', continuous=False, render_mode='human')
    env = PreprocessWrapper(env)
    
    action_size = env.action_space.n
    
    # 2. Créer un nouvel agent
    agent = DQNAgent(action_size)
    
    # 3. Charger les poids sauvegardés
    try:
        agent.load(model_path)
    except FileNotFoundError:
        print(f"Erreur : Fichier modèle non trouvé à {model_path}. Test visuel annulé.")
        env.close()
        return

    # 4. Mettre l'agent en mode ÉVALUATION (plus d'exploration aléatoire)
    agent.epsilon = 0.0
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Choisir l'action (meilleure action, car epsilon=0)
            action = agent.choose_action(state)
            
            # Agir
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
        
        print(f"Épisode de test {episode + 1} | Récompense totale : {total_reward:.2f}")
    
    env.close()
    print("Test visuel terminé.")
    print("="*70)


if __name__ == "__main__":
    
    # Defining final training parameters
    
    # Parameters based on hyperparameter search results
    FINAL_LR = 0.001  # (1e-4)
    FINAL_GAMMA = 0.99
    
    # Définissez le nombre total d'épisodes pour l'entraînement
    # 2000 est un bon point de départ pour un "vrai" run
    TOTAL_EPISODES = 1000

    # File to save the final model
    FINAL_MODEL_PATH = "model_final.pth"
    CHECKPOINT_PATH = "car_racing_checkpoint.pth" # Le fichier pour la sauvegarde/reprise
    PLOT_SAVE_PATH = "final_training_plot.png"

    # Mettez-le à True si vous voulez reprendre un entraînement arrêté
    RESUME_TRAINING = True
    # (Mettez True si le fichier car_racing_checkpoint.pth existe)

    print("=" * 70)
    print("DÉMARRAGE DE L'ENTRAÎNEMENT FINAL")
    print(f"Épisodes: {TOTAL_EPISODES}")
    print(f"Learning Rate: {FINAL_LR}")
    print(f"Gamma: {FINAL_GAMMA}")
    print(f"Modèle sauvegardé sous: {FINAL_MODEL_PATH}")
    print("=" * 70)

    # Mettre les seeds pour la reproductibilité
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # Start training
    agent, rewards, losses = train_dqn(
        episodes=TOTAL_EPISODES,
        render=False,      
        learning_rate=FINAL_LR,
        gamma=FINAL_GAMMA,
        checkpoint_path=CHECKPOINT_PATH,  # Passe le chemin
        plot_save_path=PLOT_SAVE_PATH,  # Passe le chemin
        resume=RESUME_TRAINING          # Passe le booléen de reprise
    )

    # --- 3. Sauvegarder et analyser les résultats ---
    print("\n" + "=" * 70)
    print("Entraînement final terminé !")
    
    # Save the final model
    agent.save(FINAL_MODEL_PATH)
    
    # Print final statistics
    print(f"\n{'='*70}")
    print("Final Statistics:")
    print(f"{'='*70}")
    
    # Verify if there are at least 100 episodes for average calculation
    if len(rewards) >= 100:
        print(f"  Average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    else:
        print(f"  Average reward (all episodes): {np.mean(rewards):.2f}")
        
    print(f"  Best episode reward: {max(rewards):.2f}")
    print(f"  Final epsilon: {agent.epsilon:.3f}")
    print(f"  Total steps trained: {agent.train_step_count}")
    print(f"  Buffer size: {len(agent.memory)}")

    # Plot results
    print("\nGenerating training plots...")
    plot_training_results(rewards, losses, save_path=PLOT_SAVE_PATH)

    # Visual terst of the final model
    print("\nStarting final visual test...")
    test_agent_visually(FINAL_MODEL_PATH, episodes=3)

    
    