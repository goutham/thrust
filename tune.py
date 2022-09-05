from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.scipy.optimize as jso
import numpy as np
import optax
import os
import pickle
import shutil
import time


@dataclass
class _Config:
    max_steps: int
    batch_size: int | str
    dump_interval: int
    logging_interval: int
    tune_split: float


_FULL_BATCH_CONFIG = _Config(
    max_steps=1000,
    batch_size='full',
    dump_interval=20,
    logging_interval=5,
    tune_split=0.8)
_MINI_BATCH_CONFIG = _Config(
    max_steps=100000,
    batch_size=100,
    dump_interval=100,
    logging_interval=20,
    tune_split=0.8)

CONFIG = _FULL_BATCH_CONFIG


def init_params():
    params = []
    with open('init.params') as f:
        for line in f:
            params.append(float(line.strip()))
    assert (len(params) == 782)
    return params


def load(filename='processed.pickle'):
    print(f'Loading {filename}...')
    with open(filename, 'rb') as f:
        boards = pickle.load(f)
        outcomes = pickle.load(f)
    print('Done')
    return np.asarray(boards), np.asarray(outcomes)


def reshape_psts(params):
    return jnp.reshape(params[14:], (12, 64))


def sigmoid(scale_factor, score):
    return 1. / (1. + pow(10, -scale_factor * score / 400.))


def point_error(res, val):
    return jnp.power(res - val, 2)


def evaluate(params, pos):
    psts = reshape_psts(params)
    mgame_psts = psts[0:6]
    egame_psts = psts[6:12]
    w_pieces = pos[0:6]
    b_pieces = pos[6:12]
    w_mgame_pv = jnp.einsum('i,ij->ij', params[1:7], w_pieces)
    w_egame_pv = jnp.einsum('i,ij->ij', params[8:14], w_pieces)
    b_mgame_pv = jnp.einsum('i,ij->ij', params[1:7], b_pieces)
    b_egame_pv = jnp.einsum('i,ij->ij', params[8:14], b_pieces)
    w_mgame_psts = jnp.einsum('ij,ij->ij', mgame_psts, w_pieces)
    w_egame_psts = jnp.einsum('ij,ij->ij', egame_psts, w_pieces)
    b_mgame_psts = jnp.einsum('ij,ij->ij', mgame_psts, b_pieces)
    b_egame_psts = jnp.einsum('ij,ij->ij', egame_psts, b_pieces)

    w_mgame_score = jnp.sum(jnp.add(w_mgame_pv, w_mgame_psts))
    b_mgame_score = jnp.sum(jnp.add(b_mgame_pv, b_mgame_psts))
    w_egame_score = jnp.sum(jnp.add(w_egame_pv, w_egame_psts))
    b_egame_score = jnp.sum(jnp.add(b_egame_pv, b_egame_psts))
    mgame_score = w_mgame_score - b_mgame_score
    egame_score = w_egame_score - b_egame_score

    all_pieces = jnp.add(w_pieces, b_pieces)
    game_phases = jnp.array([0., 4., 2., 1., 1., 0.])
    mgame_phase = jnp.min(jnp.array([24.0, jnp.sum(
        jnp.einsum('i,ij->ij', game_phases, all_pieces))]))
    egame_phase = 24.0 - mgame_phase
    # white pov score
    return (mgame_phase * mgame_score + egame_phase * egame_score) / 24.


def loss_fn(params, scale_factor, batch):
    scores = jax.vmap(evaluate, (None, 0))(params, batch[0])
    values = jax.vmap(sigmoid, (None, 0))(scale_factor, scores)
    errors = jax.vmap(point_error, (0, 0))(batch[1], values)
    return jnp.average(errors)


def setup():
    if os.path.exists('params'):
        shutil.rmtree('params')
    os.makedirs('params')


def dump(step, params, loss, test_set_loss):
    with open(f'params/{step}.params', 'w') as f:
        for param in params:
            f.write(f'{float(param)}\n')
    with open(f'params/losses', 'a') as f:
        f.write(f'{step}, {loss}, {test_set_loss}\n')


def tune_params(params, scale_factor, tune_set, test_set):
    optimizer = optax.adam(learning_rate=1.0)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        grad = jax.grad(loss_fn)(params, scale_factor, batch)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def log(step_num, loss, test_set_loss):
        print(f'step {step_num}. loss: {loss}, test_set_loss: {test_set_loss}')

    log(0, loss_fn(params, scale_factor, tune_set),
        loss_fn(params, scale_factor, test_set))
    for step_num in range(1, CONFIG.max_steps + 1):
        if CONFIG.batch_size == 'full':
            indices = jnp.arange(len(tune_set[0]))
        else:
            indices = jnp.arange(
                (step_num - 1)*CONFIG.batch_size, step_num*CONFIG.batch_size)
        batch = (jnp.take(tune_set[0], indices, axis=0, mode='wrap'),
                 jnp.take(tune_set[1], indices, axis=0, mode='wrap'))
        params, opt_state = step(params, opt_state, batch)
        if step_num % CONFIG.logging_interval == 0:
            loss = loss_fn(params, scale_factor, tune_set)
            test_set_loss = loss_fn(params, scale_factor, test_set)
            log(step_num, loss, test_set_loss)
        if step_num % CONFIG.dump_interval == 0:
            dump(step_num, params, loss_fn(
                params, scale_factor, tune_set), loss_fn(params, scale_factor, test_set))
    print('Reached max steps, terminating.')
    return params, loss_fn(params, scale_factor, tune_set), loss_fn(params, scale_factor, test_set)


def compute_scale_factor(params, tune_set):

    def _obj_fn(x0):
        return loss_fn(params, x0, tune_set)

    print('Computing scale_factor...')
    res = jso.minimize(_obj_fn, jnp.array([1.0]), method='BFGS')
    print('success:', res.success)
    print('status:', res.status)
    print('scale_factor:', res.x)
    assert (res.success)
    return res.x


def tune():
    setup()
    params = jnp.asarray(init_params())
    boards, outcomes = load()

    # Shuffle
    num_boards = len(boards)
    p = np.random.default_rng(int(time.time())).permutation(num_boards)
    boards = boards[p]
    outcomes = outcomes[p]

    tune_set_size = int(num_boards * CONFIG.tune_split)
    print(f'Tune set size: {tune_set_size}')
    print(f'Test set size: {num_boards - tune_set_size}')

    tune_set = (jnp.asarray(boards[:tune_set_size]),
                jnp.asarray(outcomes[:tune_set_size]))
    test_set = (jnp.asarray(boards[tune_set_size:]),
                jnp.asarray(outcomes[tune_set_size:]))

    scale_factor = compute_scale_factor(params, tune_set)

    final_params, loss, test_set_loss = tune_params(
        params, scale_factor, tune_set, test_set)
    print('Final params:', final_params)
    dump('final', final_params, loss, test_set_loss)


if __name__ == '__main__':
    tune()
