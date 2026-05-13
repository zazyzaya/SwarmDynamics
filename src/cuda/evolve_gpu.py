from argparse import ArgumentParser
from time import time

import torch

from src.dna import GenePool, MAX_GAME_LEN
from src.cuda.env import EnvCUDA as Env
from src.generators import generate_random_columns

POPULATION = 1000
game_size = 100
DEVICE = 0

def generation(gene_pool: GenePool, e, game_size, num_obstacles, blues, reds):
    st = time()
    DEVICE = gene_pool.device
    BATCH_SIZE = blues.size(0)

    b_genes, b_sexes = gene_pool.create_swarm(game_size, blues)
    r_genes, r_sexes = gene_pool.create_swarm(game_size, reds)
    env = Env(
        b_genes, b_sexes,
        r_genes, r_sexes,
        generate_random_columns(num_obstacles, DEVICE, BATCH_SIZE)
    )

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

    scores_b = torch.zeros(blues.size(0), device=DEVICE)
    scores_r = torch.zeros(reds.size(0), device=DEVICE)

    # Rank Queens based on the sum of their swarm's performance
    for b in range(BATCH_SIZE):
        game_len = game_lengths[b]

        # Award kills regardless of who wins so in situations where there's a tie
        # especially ones where there's a tie because one team asymmetrically killed
        # most of the others drones, but a few escaped and hid until time ran out,
        # the more dangerous queen gets points
        b_kills = env.b_kills[b].sum() / game_size
        r_kills = env.r_kills[b].sum() / game_size

        # final_game_over[b] holds [blue_dead, red_dead]
        b_wiped = final_game_over[b, 0].item()
        r_wiped = final_game_over[b, 1].item()

        # A win is wiping the enemy while surviving yourself
        b_won = r_wiped and not b_wiped
        r_won = b_wiped and not r_wiped

        b_score, r_score = GenePool.fitness(b_kills, r_kills, b_won, r_won, game_len / MAX_GAME_LEN)
        scores_b[b] = b_score
        scores_r[b] = r_score

    game_scores = torch.cat([scores_b, scores_r])
    game_idx = torch.cat([blues, reds])
    game_idx = torch.where(game_idx == -1, gene_pool.population, game_idx) # No neg indexes for scatter

    # Scattter scores to the queen that generated them's idx
    scores = torch.zeros(gene_pool.population + 1, device=DEVICE)
    scores = torch.scatter_reduce(scores, -1, game_idx, game_scores, 'mean', include_self=False)
    scores = scores[:-1] # Ignore games against queen -1 (baseline)
    scores = scores.nan_to_num_(0.0) # Shouldn't happen, but if queen never plays, its score is NaN

    en = time()
    elapsed = en-st

    avg_len = game_lengths.mean().item()
    avg_fitness = scores.mean().item()
    avg_fitness_std = scores.std().item()

    winners = scores.topk(int(gene_pool.population * 0.05))
    top_fitness = winners.values.mean().item()
    top_fitness_std = winners.values.std().item()
    print(
        f"[{e}] Steps: {int(avg_len)},",
        f"Avg fitness: {avg_fitness:0.4f} (+/-) {avg_fitness_std:0.2f},",
        f"Top fitness: {top_fitness:0.4f} (+/-) {top_fitness_std:0.2f}",
        f"({elapsed:0.2f}s)"
    )

    return scores, winners.indices, (
        avg_fitness, top_fitness,
        avg_fitness_std, top_fitness_std,
        avg_len, elapsed
    )


def evaluate(gene_pool: GenePool, winners, game_size, num_obstacles):
    st = time()
    DEVICE = gene_pool.device

    # We only need 1 Baseline Queen to test against
    default = GenePool(1, device=DEVICE, use_baseline=True)

    BATCH_SIZE = winners.size(0)

    print("\tEvaluating... ", end='', flush=True)

    # 1. Generate all winning Swarms simultaneously
    b_genes, b_sexes = gene_pool.create_swarm(game_size, winners)

    # 2. Generate the 1 Default Swarm
    red_queen = torch.tensor([0], device=DEVICE)
    r_genes_single, r_sexes_single = default.create_swarm(game_size, red_queen)

    # 3. Broadcast the Default Swarm 100 times so it can fight every evolved team!
    r_genes = r_genes_single.expand(BATCH_SIZE, game_size, -1)
    r_sexes = r_sexes_single.expand(BATCH_SIZE, game_size)

    env = Env(
        b_genes, b_sexes,
        r_genes, r_sexes,
        generate_random_columns(num_obstacles, DEVICE, BATCH_SIZE)
    )

    # Trackers
    # Setup identical tracking to the training loop
    final_game_over = torch.zeros(BATCH_SIZE, 2, dtype=torch.bool, device=DEVICE)
    finished = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=DEVICE)
    game_lengths = torch.full((BATCH_SIZE,), MAX_GAME_LEN, dtype=torch.float, device=DEVICE)

    for step in range(MAX_GAME_LEN):
        step_game_over = env.update()

        # Identify games that finished on this exact frame
        just_finished = step_game_over.any(dim=1) & ~finished

        # Lock in length and win/loss state
        game_lengths[just_finished] = step
        final_game_over[just_finished] = step_game_over[just_finished]

        finished = finished | just_finished
        if finished.all():
            break

    en = time()
    elapsed = en-st
    print(f'({elapsed:0.2f}s)')

    # Get fitness score
    scores = torch.zeros(BATCH_SIZE, device=DEVICE)
    for b in range(BATCH_SIZE):
        b_kills = env.b_kills[b].sum() / game_size
        r_kills = env.r_kills[b].sum() / game_size
        game_len = game_lengths[b]

        b_wiped = final_game_over[b, 0].item()
        r_wiped = final_game_over[b, 1].item()

        b_won = r_wiped and not b_wiped
        r_won = b_wiped and not r_wiped

        # We only care about saving the blue score in eval, but we still pass everything!
        b_score, r_score = GenePool.fitness(
            b_kills, r_kills, b_won, r_won, game_len / MAX_GAME_LEN
        )
        scores[b] = b_score

    return scores.cpu().tolist(), elapsed