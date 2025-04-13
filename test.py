import optuna
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3 import DQN
from rl_zoo3 import train
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

def objective(trial:optuna.Trial):
    """
    Create a tuner for the given params
    SpaceInvadersNoFrameskip-v4:
        env_wrapper:
            - stable_baselines3.common.atari_wrappers.AtariWrapper
        frame_stack: 4
        policy: 'CnnPolicy'
        n_timesteps: !!float 1e6
        buffer_size: 100000
        learning_rate: !!float 1e-4
        batch_size: 32
        learning_starts: 100000
        target_update_interval: 1000
        train_freq: 4
        gradient_steps: 1
        exploration_fraction: 0.1
        exploration_final_eps: 0.01
        # If True, you need to deactivate handle_timeout_termination
        # in the replay_buffer_kwargs
        optimize_memory_usage: False
    """
    frame_stack = trial.suggest_int("frame_stack", 3, 7)
    buffer_size = trial.suggest_categorical("buffer_size", [100000, 110000, 120000, 130000, 140000, 150000, 160000])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    learning_starts = trial.suggest_categorical("learning_starts", [100000, 110000, 120000, 130000, 140000, 150000, 160000])
    target_update_interval = trial.suggest_categorical("target_update_interval", [1000, 2000, 3000, 4000, 5000])
    train_freq = trial.suggest_int("train_freq", 1, 10)
    gradient_steps = trial.suggest_int("gradient_steps", 1, 10)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.7)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    params = {
        "env_wrapper": ["stable_baselines3.common.atari_wrappers.AtariWrapper"],
        "frame_stack": frame_stack,
        "policy": "CnnPolicy",  # Fixed for Atari
        "n_timesteps": 200000,
        "buffer_size": buffer_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        "target_update_interval": target_update_interval,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "optimize_memory_usage": False,
    }
    env_id = "SpaceInvadersNoFrameskip-v4"
    env = gym.make(env_id)
    env = AtariWrapper(env, frame_skip=4)  # Preprocess like RL-Zoo

    # Create DQN model with hyperparameters
    model = DQN(
        policy="CnnPolicy",  # For Atari games
        env=env,
        buffer_size=params["buffer_size"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        learning_starts=params["learning_starts"],
        target_update_interval=params["target_update_interval"],
        train_freq=params["train_freq"],
        gradient_steps=params["gradient_steps"],
        exploration_fraction=params["exploration_fraction"],
        exploration_initial_eps=1.0,  # Default from RL-Zoo
        exploration_final_eps=params["exploration_final_eps"],
        optimize_memory_usage=params["optimize_memory_usage"],
        verbose=1,  # Quiet output
        seed=42  # Reproducibility
    )

    model.learn(total_timesteps=100000)  # 100k for faster trials; adjust to 1M later

    # Evaluate the trained model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    env.close()

    return mean_reward  # Optuna will maximize this

def main():
    # Create an Optuna study to maximize reward
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # Run 20 trials (adjust as needed)

    # Print the best result
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    # Save the result
    with open("best_trial.txt", "w") as f:
        f.write(f"Best trial:\n  Value: {trial.value}\n  Params:\n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")
    # Visualize the params importance
    import optuna.visualization as vis
    fig = vis.plot_param_importances(study)
    # Save the figure
    fig.write_image("param_importances.png")

if __name__ == "__main__":
    main()