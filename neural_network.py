import tensorflow as tf
import numpy as np


RESIDUAL_BLOCKS = 5


class GameBoardModel(tf.keras.Model):
    def fit_game_state(self, game_states, ps, z):
        state = np.stack([gs.to_numpy() for gs in game_states])
        mask = np.vstack([gs.action_space_mask() for gs in game_states])

        X = {
            "state": state,
            "action_mask": mask,
        }
        y = {
            "policy_head": np.vstack(ps),
            "value_head": np.array(z),
        }
        self.fit(X, y)


def new_model(game_state: type) -> GameBoardModel:
    game_input = tf.keras.Input(game_state.shape(), name='state')
    action_mask = tf.keras.Input(game_state.action_space_size(), name='action_mask')

    x = conv_block(game_input)
    for _ in range(RESIDUAL_BLOCKS):
        x = residual_block(x)

    policy_head = tf.keras.layers.Conv2D(2, (1, 1), padding='same')(x)
    policy_head = tf.keras.layers.BatchNormalization()(policy_head)
    policy_head = tf.keras.layers.ReLU()(policy_head)
    policy_head = tf.keras.layers.Flatten()(policy_head)
    policy_head = tf.keras.layers.Dense(game_state.action_space_size())(policy_head)
    policy_head = tf.keras.layers.Multiply(name='policy_head')([policy_head, action_mask])

    value_head = tf.keras.layers.Conv2D(2, (1, 1))(x)
    value_head = tf.keras.layers.BatchNormalization()(value_head)
    value_head = tf.keras.layers.ReLU()(value_head)
    value_head = tf.keras.layers.Flatten()(value_head)
    value_head = tf.keras.layers.Dense(256, activation='relu')(value_head)
    value_head = tf.keras.layers.Dense(1, activation='tanh', name='value_head')(value_head)

    model = GameBoardModel([game_input, action_mask], [policy_head, value_head])

    losses = {'policy_head': 'categorical_crossentropy', 'value_head': 'mse'}
    model.compile(optimizer='adam', loss=losses)
    return model


def conv_block(x):
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def residual_block(x):
    x_init = x
    x = conv_block(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, x_init])
    x = tf.keras.layers.ReLU()(x)
    return x


if __name__ == "__main__":
    from games.connect4 import Connect4Board

    model = new_model(Connect4Board)
