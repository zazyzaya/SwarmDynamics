from argparse import ArgumentParser
from time import time

import torch

from src.dna import GenePool, MAX_GAME_LEN
from src.cuda.env import Env

POPULATION = 1000
game_size = 100
DEVICE = 0

def generation(gene_pool: GenePool, e, win_ratio, game_size):
    st = time()
    DEVICE = gene_pool.device

    blues, reds = torch.randperm(gene_pool.population, device=DEVICE).chunk(2)

    BATCH_SIZE = blues.size(0)
    n_winners = gene_pool.population // win_ratio

    b_genes, b_sexes = gene_pool.create_swarm(game_size, blues)
    r_genes, r_sexes = gene_pool.create_swarm(game_size, reds)
    env = Env(b_genes, b_sexes, r_genes, r_sexes)

    final_game_over = torch.zeros(BATCH_SIZE, 2, dtype=torch.bool, device=DEVICE)
    finished = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=DEVICE)

    # Simulate
    game_lengths = torch.full((BATCH_SIZE,), MAX_GAME_LEN, dtype=torch.float, device=DEVICE)
    for step in range(MAX_GAME_LEN):
        step_game_over = env.update() # Returns (B, 2)

        # Identify games that finished on this exact frame
        just_finished = step_game_over.any(dim=1) & ~finished
        game_lengths[just_finished] = step

        # Lock in their final win/loss state
        final_game_over[just_finished] = step_game_over[just_finished]
        finished = finished | just_finished

        # Break only when ALL games are done
        if finished.all():
            break

    scores = torch.zeros(gene_pool.population, device=DEVICE)

    # Rank Queens based on the sum of their swarm's performance
    for b in range(BATCH_SIZE):
        blue_queen = blues[b]
        red_queen = reds[b]
        game_len = game_lengths[b]

        # Award kills regardless of who wins so in situations where there's a tie
        # especially ones where there's a tie because one team asymmetrically killed
        # most of the others drones, but a few escaped and hid until time ran out,
        # the more dangerous queen gets points
        b_kills = env.b_kills[b].sum() / game_size
        r_kills = env.r_kills[b].sum() / game_size

        b_score, r_score = GenePool.fitness(b_kills, r_kills, game_len / MAX_GAME_LEN)
        scores[blue_queen] = b_score
        scores[red_queen] = r_score

    en = time()

    avg_len = game_lengths.mean().item()
    avg_fitness = scores.mean().item()
    avg_fitness_std = scores.std().item()

    winners = scores.topk(n_winners)
    top_fitness = winners.values.mean().item()
    top_fitness_std = winners.values.std().item()
    print(
        f"[{e}] Steps: {int(avg_len)},",
        f"Avg fitness: {avg_fitness:0.4f}+/-{avg_fitness_std:0.2f},",
        f"Top fitness: {top_fitness:0.4f}+/-{top_fitness_std:0.2f}",
        f"({en-st:0.2f}s)"
    )

    gene_pool.reproduce(winners.indices)
    return avg_fitness, top_fitness, avg_fitness_std, top_fitness_std, avg_len


def evaluate(gene_pool: GenePool, game_size):
    st = time()
    DEVICE = gene_pool.device

    # We only need 1 Baseline Queen to test against
    default = GenePool(1, device=DEVICE, use_baseline=True)

    BATCH_SIZE = gene_pool.population

    print("\tEvaluating... ", end='', flush=True)

    # 1. Generate all 100 Evolved Swarms simultaneously
    blue_queens = torch.arange(BATCH_SIZE, device=DEVICE)
    b_genes, b_sexes = gene_pool.create_swarm(game_size, blue_queens)

    # 2. Generate the 1 Default Swarm
    red_queen = torch.tensor([0], device=DEVICE)
    r_genes_single, r_sexes_single = default.create_swarm(game_size, red_queen)

    # 3. Broadcast the Default Swarm 100 times so it can fight every evolved team!
    r_genes = r_genes_single.expand(BATCH_SIZE, game_size, -1)
    r_sexes = r_sexes_single.expand(BATCH_SIZE, game_size)

    env = Env(b_genes, b_sexes, r_genes, r_sexes)

    # Trackers
    steps_to_win = torch.full((BATCH_SIZE,), MAX_GAME_LEN, device=DEVICE)
    finished = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=DEVICE)

    for step in range(MAX_GAME_LEN):
        step_game_over = env.update()

        just_finished = step_game_over.any(dim=1) & ~finished

        # If red lost [1] and blue didn't [0], it's a valid blue win
        blue_won = step_game_over[:, 1] & ~step_game_over[:, 0]

        # Record the exact step for the valid wins
        valid_wins = just_finished & blue_won
        steps_to_win[valid_wins] = step

        finished = finished | just_finished
        if finished.all():
            break

    en = time()
    print(f'({en-st:0.2f}s)')

    # Get fitness score
    scores = torch.zeros(BATCH_SIZE, device=DEVICE)
    for b in range(BATCH_SIZE):
        b_kills = env.b_kills[b].sum() / game_size
        r_kills = env.r_kills[b].sum() / game_size
        game_len = steps_to_win[b]

        b_score, r_score = GenePool.fitness(b_kills, r_kills, game_len / MAX_GAME_LEN)
        scores[b] = b_score

    return scores.cpu().tolist()