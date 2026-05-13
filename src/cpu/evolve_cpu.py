from argparse import ArgumentParser
from time import time

from joblib import Parallel, delayed
from tqdm import tqdm
import torch

from src.dna import GenePool, MAX_GAME_LEN
from src.cpu.env import EnvCPU as Env
from src.generators import generate_random_columns

WIN_BONUS = 1000

def generation(gene_pool: GenePool, e, win_ratio, game_size, num_obstacles, blues, reds):
    st = time()

    n_games = blues.size(0)
    n_winners = gene_pool.population // win_ratio

    def game(g):
        b_genes, b_sexes = gene_pool.create_swarm(game_size, blues[g:g+1])
        r_genes, r_sexes = gene_pool.create_swarm(game_size, reds[g:g+1])

        env = Env(
            b_genes.squeeze(0), b_sexes.squeeze(0),
            r_genes.squeeze(0), r_sexes.squeeze(0),
            generate_random_columns(num_obstacles, device='cpu')
        )

        game_over = torch.zeros(2)

        for steps in range(MAX_GAME_LEN):
            game_over = env.update()
            if game_over.any():
                break

        b_kills = env.b_kills.sum() / game_size
        r_kills = env.r_kills.sum() / game_size
        game_len = max(env.b_alive_time.max(), env.r_alive_time.max())

        b_wiped = game_over[0].item()
        r_wiped = game_over[1].item()

        b_won = r_wiped and not b_wiped
        r_won = b_wiped and not r_wiped

        b_score, r_score = GenePool.fitness(b_kills, r_kills, b_won, r_won, game_len)
        return b_score, r_score, steps

    scores = Parallel(min(n_games, 64), prefer='processes')(
        delayed(game)(g) for g in range(0, n_games)
    )
    en = time()
    elapsed = en-st

    b_scores, r_scores, steps = zip(*scores)
    game_scores = torch.tensor(b_scores + r_scores, dtype=torch.float32, device=gene_pool.device)
    game_idx = torch.cat([blues, reds])
    game_idx = torch.where(game_idx == -1, gene_pool.population, game_idx) # scatter gets angry about negative idxs

    # Scatter scores back to their queen's index
    scores = torch.zeros(gene_pool.population+1, device=gene_pool.device)
    scores = torch.scatter_reduce(scores, -1, game_idx, game_scores, 'mean', include_self=False)
    scores = scores[:-1] # Ignore games against queen -1 (baseline)
    scores = scores[:-1].nan_to_num_(0.0) # Shoudn't happen, but guard against NaN if queen never played

    avg_len = sum(steps) / len(steps)
    avg_fitness = scores.mean().item()
    avg_fitness_std = scores.std().item()

    winners = scores.topk(n_winners)
    top_fitness = winners.values.mean().item()
    top_fitness_std = winners.values.std().item()
    print(
        f"[{e}] Steps: {int(avg_len)},",
        f"Avg fitness: {avg_fitness:0.4f} (+/-) {avg_fitness_std:0.2f},",
        f"Top fitness: {top_fitness:0.4f} (+/-) {top_fitness_std:0.2f}",
        f"({elapsed:0.2f}s)"
    )

    return winners.indices, (
        avg_fitness, top_fitness,
        avg_fitness_std, top_fitness_std,
        avg_len, elapsed
    )


def evaluate(gene_pool: GenePool, winners, game_size, num_obstacles):
    st = time()
    DEVICE = gene_pool.device

    default = GenePool(1, device=DEVICE, use_baseline=True)
    n_games = winners.size(0)

    print("\tEvaluating... ", end='', flush=True)
    def game(g):
        blue_queen = winners[g:g+1]
        b_genes, b_sexes = gene_pool.create_swarm(game_size, blue_queen)

        r_genes, r_sexes = default.create_swarm(game_size, torch.tensor([0]))

        env = Env(
            b_genes.squeeze(0), b_sexes.squeeze(0),
            r_genes.squeeze(0), r_sexes.squeeze(0),
            generate_random_columns(num_obstacles, 'cpu')
        )

        game_over = torch.zeros(2)
        for step in range(MAX_GAME_LEN):
            game_over = env.update()
            if game_over.any():
                break

        b_kills = env.b_kills.sum() / game_size
        r_kills = env.r_kills.sum() / game_size
        game_len = max(env.b_alive_time.max(), env.r_alive_time.max())

        b_wiped = game_over[0].item()
        r_wiped = game_over[1].item()

        b_won = r_wiped and not b_wiped
        r_won = b_wiped and not r_wiped

        b_score, r_score = GenePool.fitness(b_kills, r_kills, b_won, r_won, game_len)
        return b_score

    fitness = Parallel(n_games, prefer='processes')(
        delayed(game)(g) for g in range(n_games)
    )
    en = time()
    elapsed = en-st
    print(f'({elapsed:0.2f}s)')

    return fitness, elapsed

