HYPERPARAMETERS = {
    'CartPole-v1': {
        'learning_rate': 2.3e-3,
        'batch_size': 64,
        'buffer_size': 100000,
        'learning_starts': 1000,
        'gamma': 0.99,
        'target_update_interval': 10,
        'train_freq': 256,
        'gradient_steps': 128,
        'exploration_fraction': 0.16,
        'exploration_final_eps': 0.04
    },

    'MountainCar-v0': {
        'learning_rate': 4e-3,
        'batch_size': 128,
        'buffer_size': 10000,
        'learning_starts': 1000,
        'gamma': 0.98,
        'target_update_interval': 600,
        'train_freq': 16,
        'gradient_steps': 8,
        'exploration_fraction': 0.2,
        'exploration_final_eps': 0.07
    },

    'LunarLander-v2': {
        'learning_rate': 6.3e-4,
        'batch_size': 128,
        'buffer_size': 50000,
        'learning_starts': 0,
        'gamma': 0.99,
        'target_update_interval': 250,
        'train_freq': 4,
        'gradient_steps': -1,
        'exploration_fraction': 0.12,
        'exploration_final_eps': 0.1
    },

    'Acrobot-v1': {
        'learning_rate': 6.3e-4,
        'batch_size': 128,
        'buffer_size': 50000,
        'learning_starts': 0,
        'gamma': 0.99,
        'target_update_interval': 250,
        'train_freq': 4,
        'gradient_steps': -1,
        'exploration_fraction': 0.12,
        'exploration_final_eps': 0.1
    }
}