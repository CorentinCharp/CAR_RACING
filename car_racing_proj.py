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

        # Temporarily deactivate gradient calculations for efficiency
        with torch.no_grad():
            # Pass the state tensor through the online Q-network to get Q-values for all actions.
            q_values = self.q_network_online(state_tensor)
        return q_values.argmax().item()
    
    # Stores a transition (experience) in the replay buffer.
    def store_transition(self, state, action, reward, next_state, done):
        """ Stores a (s, a, r, s', done) transition in the replay buffer """
        self.memory.store(state, action, reward, next_state, done)
    
    # Performs a single training step using a batch sampled from the replay buffer.
    def train_agent(self):
        """ Trains the agent by sampling a batch from the replay buffer """

        # If memorty has less than batch_size, do nothing
        if len(self.memory) < self.batch_size:
            return None # Not enough samples to train

        # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert data into torch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device) # We use LongTensor because the actions are discrete
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q-Value calculation
        # Calculate the Q-values for the sampled states using the ONLINE network
        current_q_values = self.q_network_online(states)
        # Select ONLY the Q-values corresponding to the 'actions' actually taken in those states
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Calculate the target Q-values using the TARGET network
        with torch.no_grad():
            # Get the Q-values for the 'next_states' from the TARGET network
            next_q_values_target = self.q_network_target(next_states)
            # Find the maximum Q-value among all possible actions in each 'next_state'
            max_next_q = next_q_values_target.max(1)[0]
            
            # Calculate the target Q-value using the Bellman equation
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Calculate the Mean Squared Error (MSE) loss between the predicted Q-values (current_q)
        # and the calculated target Q-values (target_q).
        loss = self.criterion(current_q, target_q)
        
        # Optimize the online network
        # Reset gradients from the previous step
        self.optimizer.zero_grad()
        # Compute gradients of the loss with respect to the online network's parameters (Backpropagation).
        loss.backward()
        # Update the online network's parameters using the computed gradients (Adam optimization step).
        self.optimizer.step()
        
        # Target network update
        self.train_step_count += 1 # Increment the training step counter
        if self.train_step_count % self.target_update_freq == 0: # Check if it's time to update the target network.
            self.update_target_network() # If yes, copy the weights from the online network to the target network.
        return loss.item() # Return the computed loss value

    # Copies the current weights from the online network to the target network.
    # This is done periodically to keep the target Q-values stable.
    def update_target_network(self):
        """ Copies weights from the online network to the target network """
        print("--- Mise à jour du Target Network ---")
        # Overwrite the target network's parameters with the online network's current parameters
        self.q_network_target.load_state_dict(self.q_network_online.state_dict())
    
    def decay_epsilon(self):
        """ Decreases epsilon according to the decay rate, but not below epsilon_min """

        # Multiply epsilon by the decay factor, ensuring it doesn't go below the minimum threshold
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """ Saves the agent's state (online network weights, optimizer state, epsilon, step count) to a file """
        torch.save({
            'model_state_dict': self.q_network_online.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step_count': self.train_step_count
        }, filepath)
        print(f"Modèle sauvegardé sous : {filepath}")

    def load(self, filepath):
        """ Loads the agent's state from a file """
        
        if not torch.cuda.is_available():
            checkpoint = torch.load(filepath, map_location='cpu')
        else:
            checkpoint = torch.load(filepath)
            
        self.q_network_online.load_state_dict(checkpoint['model_state_dict'])
        self.q_network_target.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step_count = checkpoint['train_step_count']
        
        self.q_network_online.eval() # Put online network into evaluation mode
        self.q_network_target.eval() # Put target network into evaluation mode
        print(f"Modèle chargé depuis : {filepath}. Epsilon restauré à {self.epsilon:.3f}")
        
def train_dqn(episodes=1000, render=False, learning_rate=1e-4, gamma=0.99,
              checkpoint_path=None, plot_save_path=None, resume=False):
    """ Train a DQN agent in the CarRacing-v3 environment with checkpointing and video recording """

    # Creates the CarRacing environment instance using Gymnasium with discrete actions
    env = gym.make('CarRacing-v3', continuous=False, render_mode='human' if render else None)
    
    # Wraps the environment with our custom PreprocessWrapper
    env = PreprocessWrapper(env)
    
    # Get the number of possible discrete actions from the environment's action space
    action_size = env.action_space.n
    
    # Create an instance of our DQNAgent class, passing the action size and hyperparameters
    agent = DQNAgent(action_size, 
                     learning_rate=learning_rate, 
                     discount_factor=gamma)
    
    start_episode = 0
    rewards_per_episode = [] # List to store the total reward collected in each episode
    losses_per_episode = [] # List to store the average loss calculated in each episode

    # Define paths for saving/loading reward and loss history
    rewards_path = f"{checkpoint_path}_rewards.npy" if checkpoint_path else None
    losses_path = f"{checkpoint_path}_losses.npy" if checkpoint_path else None

    # Check if resuming is requested, a checkpoint path is provided, and the checkpoint file actually exists
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            print(f"Reprise depuis le checkpoint : {checkpoint_path}")
            agent.load(checkpoint_path)
            
            # Load saved metrics and number of past episodes done
            rewards_per_episode = list(np.load(rewards_path))
            losses_per_episode = list(np.load(losses_path))
            start_episode = len(rewards_per_episode)
            
            print(f"Reprise à l'épisode {start_episode}. Epsilon actuel: {agent.epsilon:.3f}")
        except Exception as e:
            # If we can't load the checkpoint, we start form the begining
            print(f"Erreur lors du chargement du checkpoint, l'entraînement repart de zéro. Erreur : {e}")
            start_episode = 0 
            rewards_per_episode = []
            losses_per_episode = []
    
    if start_episode == 0: 
        print("Début d'un nouvel entraînement. Préchauffage de la mémoire...")

        # Define the number of initial random steps to take to fill the replay buffer
        WARMUP_STEPS = 1000
        steps = 0

        # Loop until the desired number of warmup steps is reached
        while steps < WARMUP_STEPS:
            state, _ = env.reset() # Reset the environment for a new trajectory
            done = False # Flag indicating if the episode has ended
            while not done:
                action = env.action_space.sample() # choose a random action
                next_state, reward, terminated, truncated, _ = env.step(action) # Take the random action in the environment
                done = terminated or truncated # Check if the episode ended
                agent.store_transition(state, action, reward, next_state, done) # Store the resulting transition in the agent's replay buffer
                state = next_state
                steps += 1
                if steps >= WARMUP_STEPS:
                    break
    print(f"Mémoire préchauffée avec {len(agent.memory)} expériences.")

    # Checkpoint configuration
    CHECKPOINT_FREQ = 50  # Save every 50 episodes
    CHECKPOINT_PATH = "car_racing_checkpoint.pth" # Path to ssave the model

    # Loop through episodes, starting from 'start_episode' up to (but not including) 'episodes'
    for episode in range(start_episode, episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_losses = []
        done = False
        
        while not done:
            # Choose an action
            action = agent.choose_action(state)
            
            # Interact with Environment: Take the chosen action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store Experience: Store the transition (s, a, r, s', done) in the agent's replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train Agent: Perform one training step by sampling a batch from the replay buffer.
            loss = agent.train_agent()
            if loss is not None: # If training occurred (loss is not None), record the loss value
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
        
        # Decay Epsilon: Decrease the exploration rate after the episode finishes
        agent.decay_epsilon()
        
        rewards_per_episode.append(total_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses_per_episode.append(avg_loss)
        
        # Every 10 episodesn we calculate the mean and do an update to the user with metrics
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_per_episode[-10:])
            print(f"Épisode {episode + 1}/{episodes} | "
                  f"Récompense Moy (10 derniers): {avg_reward:.2f} | "
                  f"Perte Moy: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
            
        # If checkpoint exists and episode is a multiple of checkpoint
        if (episode + 1) % CHECKPOINT_FREQ == 0 and checkpoint_path:
            agent.save(CHECKPOINT_PATH) # Save the complete agent state (model, optimizer, epsilon, step count)
            
            # Save the metrics
            np.save(rewards_path, np.array(rewards_per_episode))
            np.save(losses_path, np.array(losses_per_episode))
            print(f"*** CHECKPOINT (Modèle + Métriques) SAUVEGARDÉ à l'épisode {episode + 1} ***")
            
            # Record a video
            record_test_video(agent, episode_number=(episode + 1))
            
            # Generate metrics and graphs and save them
            if plot_save_path:
                plot_filename = plot_save_path.replace(".png", f"_ep{episode + 1}.png")
                plot_training_results(rewards_per_episode, losses_per_episode, save_path=plot_filename)
    
    env.close() # Close the Gymnasium environment to release resources.
    print("=" * 60)
    print(f"Entraînement terminé pour LR={learning_rate}, Gamma={gamma}")
    
    # Return the trained agent and the full history of rewards and losses.
    return agent, rewards_per_episode, losses_per_episode

def plot_training_results(rewards, losses, save_path=None):
    """ Plot training metrics """
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

    # Saving or plotting the figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graphiques sauvegardés sous : {save_path}")
        plt.close(fig) # Close the figure to free memory but also to not block the training
    else:
        # If no save_path is given (only at the end), show the plot
        plt.savefig('ddqn_per_training_results.png', dpi=150, bbox_inches='tight')
        print("Training plots saved to 'ddqn_per_training_results.png'")
        plt.show()
        plt.close(fig)

def record_test_video(agent, episode_number, video_folder_base='videos'):
    """
    Create a temporary test environment, record 1 video of the agent in evaluation mode, 
    then close the environment.
    """
    print(f"\n--- Enregistrement vidéo de l'épisode {episode_number} ---")
    
    video_folder = os.path.join(video_folder_base, f"ep_{episode_number}")

    try:
        test_env = gym.make('CarRacing-v3', continuous=False, render_mode='rgb_array')
        test_env = gym.wrappers.RecordVideo(test_env, 
                                           video_folder, 
                                           episode_trigger=lambda e: e == 0, 
                                           name_prefix=f"dqn-agent-ep{episode_number}") 

        test_env = PreprocessWrapper(test_env)
    except Exception as e:
        print(f"Erreur lors de la création de l'env de test vidéo : {e}")
        return

    epsilon_backup = agent.epsilon
    agent.epsilon = 0.0 
    
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
    
    test_env.close()
    
    agent.epsilon = epsilon_backup
    print(f"Vidéo de progression sauvegardée dans le dossier : {video_folder}")

def test_agent_visually(model_path, episodes=5):
    """ Load the model and run a few episodes in human render mode for visual inspection"""
    print(f"\n" + "="*70)
    print(f"TEST VISUEL du modèle : {model_path}")
    print(f"Lancement de {episodes} épisodes...")
    
    env = gym.make('CarRacing-v3', continuous=False, render_mode='human')
    env = PreprocessWrapper(env)
    
    action_size = env.action_space.n
    
    agent = DQNAgent(action_size)
    
    try:
        agent.load(model_path)
    except FileNotFoundError:
        print(f"Erreur : Fichier modèle non trouvé à {model_path}. Test visuel annulé.")
        env.close()
        return

    agent.epsilon = 0.0
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            
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
    FINAL_LR = 0.001 
    FINAL_GAMMA = 0.99
    
    # Number of desired episodes for training
    TOTAL_EPISODES = 1500

    # File to save the final model
    FINAL_MODEL_PATH = "model_final.pth"
    CHECKPOINT_PATH = "car_racing_checkpoint.pth"
    PLOT_SAVE_PATH = "final_training_plot.png"

    # Choose if you want to start from the beginning or resume a training (True = resume)
    RESUME_TRAINING = True

    print("=" * 70)
    print("DÉMARRAGE DE L'ENTRAÎNEMENT FINAL")
    print(f"Épisodes: {TOTAL_EPISODES}")
    print(f"Learning Rate: {FINAL_LR}")
    print(f"Gamma: {FINAL_GAMMA}")
    print(f"Modèle sauvegardé sous: {FINAL_MODEL_PATH}")
    print("=" * 70)

    # Defining seeds for reproductability
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # Start training
    agent, rewards, losses = train_dqn(
        episodes=TOTAL_EPISODES,
        render=False,      
        learning_rate=FINAL_LR,
        gamma=FINAL_GAMMA,
        checkpoint_path=CHECKPOINT_PATH,  
        plot_save_path=PLOT_SAVE_PATH,  
        resume=RESUME_TRAINING          
    )

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

    # # Plot results
    print("\nGenerating training plots...")
    plot_training_results(rewards, losses, save_path=PLOT_SAVE_PATH)

    # Visual terst of the final model
    print("\nStarting final visual test...")
    test_agent_visually(CHECKPOINT_PATH, episodes=3)

    
    