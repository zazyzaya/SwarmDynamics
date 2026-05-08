from argparse import ArgumentParser
from time import time

import torch

from src.dna import GenePool
from src.cuda.env import Env

MAX_GAME_LEN = 5000
POPULATION = 100
GAME_SIZE = 100
WIN_BONUS = 1000
DEVICE = 0

def generation(gene_pool: GenePool):
    st = time()

    blues, reds = torch.randperm(gene_pool.population, device=DEVICE).chunk(2)

    BATCH_SIZE = blues.size(0)
    n_winners = gene_pool.population // 2

    print("\tSimulating... ", end='', flush=True)

    b_genes, b_sexes = gene_pool.create_swarm(GAME_SIZE, blues)
    r_genes, r_sexes = gene_pool.create_swarm(GAME_SIZE, reds)
    env = Env(b_genes, b_sexes, r_genes, r_sexes)

    final_game_over = torch.zeros(BATCH_SIZE, 2, dtype=torch.bool, device=DEVICE)
    finished = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=DEVICE)

    # Simulate
    for _ in range(MAX_GAME_LEN):
        step_game_over = env.update() # Returns (B, 2)

        # Identify games that finished on this exact frame
        just_finished = step_game_over.any(dim=1) & ~finished

        # Lock in their final win/loss state
        final_game_over[just_finished] = step_game_over[just_finished]
        finished = finished | just_finished

        # Break only when ALL games are done
        if finished.all():
            break

    scores = torch.zeros(gene_pool.population, device=DEVICE)

    # Rank Queens based on the sum of their swarm's performance
    for b in range(BATCH_SIZE):
        game_over = final_game_over[b]
        draw = game_over.sum() == 0 or game_over.sum() == 2

        blue_queen = blues[b]
        red_queen = reds[b]

        if not draw and game_over[1]: # Blue wins
            game_len = env.b_alive_time[b].max()
            speed_bonus = (MAX_GAME_LEN - game_len) / MAX_GAME_LEN
            speed_bonus *= (WIN_BONUS / 2)

            # Sum up the performance of the 100 drones and assign it to the Queen
            scores[blue_queen] += WIN_BONUS + speed_bonus
            scores[blue_queen] += env.b_kills[b].sum() * (WIN_BONUS / 20)
            scores[blue_queen] += (env.b_alive_time[b] < game_len).float().sum() * (-WIN_BONUS/100)

        elif not draw and game_over[0]: # Red wins
            game_len = env.r_alive_time[b].max()
            speed_bonus = (MAX_GAME_LEN - game_len) / MAX_GAME_LEN
            speed_bonus *= (WIN_BONUS / 2)

            # Sum up the performance of the 100 drones and assign it to the Queen
            scores[red_queen] += WIN_BONUS + speed_bonus
            scores[red_queen] += env.r_kills[b].sum() * (WIN_BONUS / 20)
            scores[red_queen] += (env.r_alive_time[b] < game_len).float().sum() * (-WIN_BONUS/100)

    en = time()
    print(f" ({en-st:0.2f}s)")

    winners = scores.sort(descending=True).indices[:n_winners]
    gene_pool.reproduce(winners)


def evaluate(gene_pool: GenePool):
    st = time()
    # We only need 1 Baseline Queen to test against
    default = GenePool(1, device=DEVICE, use_baseline=True)

    BATCH_SIZE = gene_pool.population

    print("\tEvaluating... ", end='', flush=True)

    # 1. Generate all 100 Evolved Swarms simultaneously
    blue_queens = torch.arange(BATCH_SIZE, device=DEVICE)
    b_genes, b_sexes = gene_pool.create_swarm(GAME_SIZE, blue_queens)

    # 2. Generate the 1 Default Swarm
    red_queen = torch.tensor([0], device=DEVICE)
    r_genes_single, r_sexes_single = default.create_swarm(GAME_SIZE, red_queen)

    # 3. Broadcast the Default Swarm 100 times so it can fight every evolved team!
    r_genes = r_genes_single.expand(BATCH_SIZE, GAME_SIZE, -1)
    r_sexes = r_sexes_single.expand(BATCH_SIZE, GAME_SIZE)

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

    return steps_to_win.cpu().tolist()


def train():
    pool = GenePool(POPULATION, device=DEVICE, use_baseline=False)

    log = 'log.txt'
    best = MAX_GAME_LEN

    for e in range(1, 100_000):
        generation(pool)

        scores = evaluate(pool)
        avg = MAX_GAME_LEN - (sum(scores) / len(scores))
        max_v = MAX_GAME_LEN - min(scores)

        print(f'[{e}] Avg: {int(avg)}, Best: {max_v}', end='')
        if avg > best:
            best = avg
            print('*')
            pool.save('genes/best.pt')
        else:
            print()

        if e % 100 == 0:
            pool.save(f'genes/{e // 100}.pt')

        with open(log, 'a') as f:
            f.write(f'{e},{avg},{max_v}\n')

        pool.save('genes/current.pt')

if __name__ == '__main__':
    with open('log.txt', 'w+') as f:
        f.write('generation,avg,best\n')
    train()