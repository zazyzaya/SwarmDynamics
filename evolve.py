from argparse import ArgumentParser
from time import time

from joblib import Parallel, delayed
from tqdm import tqdm
import torch

from src.dna import GenePool
from src.env import Env

MAX_GAME_LEN = 5000
POPULATION = 10_000
GAME_SIZE = 100
WIN_BONUS = 1000 # Arbitrary
DEVICE = 0 # Why is this faster than my GPU??

def generation(gene_pool: GenePool):
    st = time()
    lineup = torch.randperm(gene_pool.population, device=DEVICE)

    # Calculate how many simultaneous games we are running
    BATCH_SIZE = gene_pool.population // (GAME_SIZE * 2)
    n_winners = GAME_SIZE // 10

    print("\tSimulating... ", end='', flush=True)

    # 1. Grab the flat IDs for both teams
    blue_idx_flat = lineup[:BATCH_SIZE * GAME_SIZE]
    red_idx_flat = lineup[BATCH_SIZE * GAME_SIZE : BATCH_SIZE * GAME_SIZE * 2]

    # 2. Extract phenotype tuples
    blue_genes, blue_sexes = gene_pool.phenotype(blue_idx_flat)
    red_genes, red_sexes = gene_pool.phenotype(red_idx_flat)

    # 3. Reshape into 3D Batches: (Batch, Game_Size, Features)
    env = Env(
        blue_genes.view(BATCH_SIZE, GAME_SIZE, -1),
        blue_sexes.view(BATCH_SIZE, GAME_SIZE),
        red_genes.view(BATCH_SIZE, GAME_SIZE, -1),
        red_sexes.view(BATCH_SIZE, GAME_SIZE)
    )

    # Trackers to catch when specific games finish within the massive batch
    final_game_over = torch.zeros(BATCH_SIZE, 2, dtype=torch.bool, device=DEVICE)
    finished = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=DEVICE)

    # 4. The Unified Physics Loop
    for _ in range(MAX_GAME_LEN):
        step_game_over = env.update() # Returns (B, 2)

        # Identify games that finished on this exact frame
        just_finished = step_game_over.any(dim=1) & ~finished

        # Lock in their final win/loss state
        final_game_over[just_finished] = step_game_over[just_finished]
        finished = finished | just_finished

        # Break only when ALL 50 games are done
        if finished.all():
            break

    # 5. Collect Winners
    scores = torch.empty(gene_pool.population, device=DEVICE)

    # Reshape the original IDs so we can slice them per-game
    blue_idx = blue_idx_flat.view(BATCH_SIZE, GAME_SIZE)
    red_idx = red_idx_flat.view(BATCH_SIZE, GAME_SIZE)

    # Rank agents
    for b in range(BATCH_SIZE):
        game_over = final_game_over[b]
        draw = game_over.sum() == 0 or game_over.sum() == 2

        blue_team = blue_idx[b]
        red_team = red_idx[b]

        if not draw and game_over[1]:
            # Team bonuses for winning quickly
            game_len = env.b_alive_time[b].max()
            speed_bonus = (MAX_GAME_LEN - game_len) / MAX_GAME_LEN
            speed_bonus *= (WIN_BONUS / 2)
            scores[blue_team] += WIN_BONUS + speed_bonus

            # Individual bonuses for valor and survival
            scores[blue_team] += env.b_kills[b] * WIN_BONUS / 20
            scores[blue_team] += (env.b_alive_time[b] < game_len).float() * (-WIN_BONUS/100)

        elif not draw and game_over[0]:
            # Team bonuses for winning quickly
            game_len = env.r_alive_time[b].max()
            speed_bonus = (MAX_GAME_LEN - game_len) / MAX_GAME_LEN
            speed_bonus *= (WIN_BONUS / 2)
            scores[red_team] += WIN_BONUS + speed_bonus

            # Individual bonuses for valor and survival
            scores[red_team] += env.r_kills[b] * WIN_BONUS / 20
            scores[red_team] += (env.r_alive_time[b] < game_len).float() * (-WIN_BONUS/100)


    en = time()
    print(f" ({en-st:0.2f}s)")

    winners = scores.sort(descending=True).indices[:n_winners]
    gene_pool.reproduce(winners)


def evaluate(gene_pool: GenePool):
    st = time()
    default = GenePool(GAME_SIZE, device=DEVICE, use_baseline=True)

    lineup = torch.randperm(gene_pool.population)
    BATCH_SIZE = gene_pool.population // GAME_SIZE

    print("\tEvaluating... ", end='', flush=True)

    # 1. Prepare Evolved Swarm
    blue_idx_flat = lineup[:BATCH_SIZE * GAME_SIZE]
    blue_genes, blue_sexes = gene_pool.phenotype(blue_idx_flat)

    blue_genes = blue_genes.view(BATCH_SIZE, GAME_SIZE, -1)
    blue_sexes = blue_sexes.view(BATCH_SIZE, GAME_SIZE)

    # 2. Prepare Default Swarm
    default_genes, default_sexes = default.phenotype()

    # Broadcast the single default swarm to fight against every evolved team simultaneously!
    red_genes = default_genes.unsqueeze(0).expand(BATCH_SIZE, GAME_SIZE, -1)
    red_sexes = default_sexes.unsqueeze(0).expand(BATCH_SIZE, GAME_SIZE)

    env = Env(blue_genes, blue_sexes, red_genes, red_sexes)

    # Trackers
    steps_to_win = torch.full((BATCH_SIZE,), MAX_GAME_LEN, device=DEVICE)
    finished = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=DEVICE)

    for step in range(MAX_GAME_LEN):
        step_game_over = env.update()

        just_finished = step_game_over.any(dim=1) & ~finished

        # If red lost [1] and blue didn't [0], it's a blue win!
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
    pool = GenePool(POPULATION, device=DEVICE)

    log = 'log.txt'
    best = MAX_GAME_LEN

    for e in range(1,100_000):
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