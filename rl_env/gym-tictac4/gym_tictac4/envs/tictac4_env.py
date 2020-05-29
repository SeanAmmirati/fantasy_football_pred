import gym
from gym import error, spaces, utils
from gym.utils import seeding


import pandas as pd
import numpy as np
from .ai_strategy import Team, BestOutOfOptions


class TicTac4(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_players=12, draft_pos=0, division=[1, 2, 3]):
        self.input_df = pd.read_csv(
            '/Users/seanammirati/dev/fantasy_football_predictions/data/external/projections_by_player.csv')
        self.incorporate_injuries()

        self.state = self._create_initial_state(n_players, draft_pos)
        self.n_players = n_players
        self.draft_pos = draft_pos

        self.teams = [f'opp_team_{i}' for i in range(1, self.n_players)]
        self.teams.insert(self.draft_pos, 'my_team')

        self.division = division

        self.team_objs = [Team(name) for name in self.teams]
        self.ai_strategies = [BestOutOfOptions(
            'ai_strat', t) for t in self.team_objs]
        self.round = 0
        self.position_in_round = 0
        self.action_space = spaces.Discrete(self.state.shape[0] - 1)

        total_shape = self.state.shape[1] * self.state.shape[0]

        obs_space_tup = spaces.Tuple([spaces.Tuple([spaces.Discrete(self.n_players + 1),
                                                    spaces.Discrete(16 + 1),
                                                    spaces.Discrete(6)])] * self.state.shape[0])
        obs_space_box = spaces.Box(
            low=np.repeat(-1, total_shape).reshape(self.state.shape),
            high=np.repeat(1000,  total_shape).reshape(self.state.shape))

        self.observation_space = obs_space_tup
        self.done = 0
        self.add = [0, 0]
        self.reward = 0
        self.current_team = ''

    def incorporate_injuries(self):
        inj_df = pd.read_excel(
            '/Users/seanammirati/dev/fantasy_football_predictions/data/external/injury_by_player.xlsx', header=2)
        inj_df['full_name'] = inj_df['Player'].str.extract('(.*),.*')
        inj_df = inj_df[['full_name', 'game_pct', 'ngames']]
        self.input_df['full_name'] = self.input_df['first_name'] + \
            ' ' + self.input_df['last_name']
        self.input_df = self.input_df.merge(inj_df, how='left')
        self.input_df.rename(columns={'game_pct': 'proj_probability_of_injury_game',
                                      'ngames': 'proj_number_of_games_if_injured'}, inplace=True)

        self.input_df.fillna(self.input_df.mean(), inplace=True)
        self.input_df.loc[self.input_df['position'] ==
                          'DST', 'proj_probability_of_injury_game'] = 0

    def _create_initial_state(self, n_players, draft_pos=0):

        df = self.input_df.copy()
        df['team'] = df['team'].rank(
            method='dense')
        df['position'] = df['position'].rank(
            method='dense')

        df['claimed_by'] = -1
        df['round_claimed'] = -1

        col_keep = [
            # 'team',
            # 'age', 'points',
            # 'position', 'points',
            # 'tier',
            # 'sd_points',
            'claimed_by', 'round_claimed',
            'position'
        ]

        df = df[col_keep].astype(
            float)

        return df

    def _simulate_actual_outcomes(self):
        division_schedule = (['division'] * 6) + (['non_division'] * 8)
        np.random.shuffle(division_schedule)
        non_division = [x for x in range(
            1, self.n_players) if x not in self.division]
        division = self.division * 2

        random_schedule = []
        for x in division_schedule:
            if x == 'division':
                l_to_use = division

            else:
                l_to_use = non_division

            p_idx = np.random.choice(np.arange(0, len(l_to_use)))
            random_schedule.append(l_to_use.pop(p_idx))

        byes = pd.read_excel(
            '/Users/seanammirati/dev/fantasy_football_predictions/references/bye_weeks.xlsx')
        byes_dict = byes.set_index('Abbreviation')['Week'].to_dict()
        weekly_dfs = [self.input_df.copy() for x in random_schedule]

        for df in weekly_dfs:
            df['injured'] = False

        for i, df in enumerate(weekly_dfs):
            # I'm going to alter the points
            # Use points from the last game as the points here
            if i > 0:
                df.loc[~(df['injured'] | weekly_dfs[i - 1]['injured']),
                       'points'] = weekly_dfs[i - 1]['act_points']
            else:
                # First, divide by the total number of games
                df['points'] /= 16

            # Now, incorporate byes
            week = df['team'].map(byes_dict)
            # No points, bye week
            df.loc[week == i, 'points'] = 0

            # Now, there is true points -- random variation in the points
            # Using sqroot of number of sources to try to estimate game variance
            game_sd = (((2 * df['sd_pts']) ** 2) / (16 ** 2)) ** .5
            df['act_points'] = np.random.normal(
                self.input_df['points'] / 16, game_sd.fillna(game_sd.mean()))

            # Finally, what if they get injured? And for how long?
            r = np.random.uniform(size=df.shape[0])
            injured = df['proj_probability_of_injury_game'] >= r
            n_days_injured = df['proj_number_of_games_if_injured']
            n_days_injured[~injured] = 0

            current_effect = n_days_injured.apply(lambda x: min(x, 1))
            df['act_points'] *= (1 - current_effect)
            df['injured'] = df['injured'] | (current_effect > 0)
            n_days_injured -= current_effect

            future_effects = n_days_injured[n_days_injured > 0]

            selected = i
            while (future_effects > 0).any():
                selected + - 1
                current_effect = future_effects.apply(lambda x: min(x, 1))
                try:
                    weekly_dfs[selected].loc[future_effects.index,
                                             'act_points'] *= (1 - current_effect)
                    weekly_dfs[selected].loc[future_effects.index,
                                             'injured'] = True
                except IndexError:
                    break

                future_effects -= current_effect
                future_effects = future_effects[future_effects > 0]
                df.fillna(0, inplace=True)

        return random_schedule, weekly_dfs

    def _determine_invalid_selection(self, state):

        possible_positions = ['QB', 'RB', 'TE', 'WR', 'DST', 'K']

        rnk_dict = pd.Series(possible_positions, index=possible_positions
                             ).rank(method='dense').to_dict()
        vc = self.input_df.loc[state['claimed_by']
                               == 0, 'position'].map(rnk_dict).value_counts()

        amt_off = 0
        pos_to_range = {rnk_dict['QB']: [1, 3],
                        rnk_dict['RB']: [2, np.inf],
                        rnk_dict['WR']: [2, np.inf],
                        rnk_dict['K']: [1, np.inf],
                        rnk_dict['TE']: [1, np.inf],
                        rnk_dict['DST']: [1, np.inf]}

        for k, v in pos_to_range.items():
            if k not in vc:
                amt_off += 1
            elif (vc[k] < v[0]):
                amt_off += abs(vc[k] - v[0])
            elif (vc[k] > v[1]):
                amt_off += abs(vc[k] - v[1])
        return amt_off

    def _determine_rank_complete(self, state):
        if self._determine_invalid_selection(state):
            return -100000 * self._determine_invalid_selection(state)
        else:
            simulated_sched, simulated_res = self._simulate_actual_outcomes()

            wins, losses = 0, 0

            reward = 0
            for i, game in enumerate(simulated_sched):

                teams = {'my_team': [],
                         f'opp_team_{game}': []}
                for team in [0, game]:
                    df = simulated_res[i]
                    team_name = 'my_team' if team == 0 else f'opp_team_{team}'
                    df = df[self.state['claimed_by'] == team]

                    for p in ['QB', 'RB', 'WR', 'K', 'TE', 'DST']:
                        position_index = self.input_df[self.input_df['position'] == p].index.intersection(
                            df.index)
                        wildcard_index = self.input_df[self.input_df['position'].isin(['TE', 'RB', 'WR'])].index.intersection(
                            df.index).intersection(self.input_df[~self.input_df['id'].isin(teams[team_name])].index)

                        n_to_select = 1 if p in ['Q', 'K', 'DST', 'TE'] else 2
                        try:
                            idx = df.sort_values(
                                by='points').loc[position_index, 'id'].head(n_to_select)
                        except:
                            import pdb
                            pdb.set_trace()
                        teams[team_name] += idx.tolist()
                        final_addition = df.loc[wildcard_index, 'id'].head(1)
                        teams[team_name] += final_addition.tolist()

                their_actual = df.loc[df['id'].isin(
                    teams[f'opp_team_{game}']), 'act_points'].sum()
                our_actual = df.loc[df['id'].isin(
                    teams[f'my_team']), 'act_points'].sum()

                if their_actual > our_actual:
                    losses += 1
                else:
                    wins += 1
                reward += our_actual - their_actual
            # reward = wins - losses

            return reward

    def check(self):

        if(self.round == 17):
            return self._determine_rank_complete(self.state)
        else:
            return 0

    def step(self, action):
        n_choices = 0

        while True:
            team = self.teams[self.position_in_round]
            if team == 'my_team':
                n_choices += 1
                pick = self.state.loc[action]
                while pick['round_claimed'] != -1:
                    action = action + 1
                    pick = self.state.loc[action]

                if n_choices == 2:
                    return [self.state, self.reward, self.done, {}]
            else:
                fake = self.input_df.copy()
                fake[['claimed_by', 'round_claimed']
                     ] = self.state[['claimed_by', 'round_claimed']]
                pick = self.ai_strategies[self.position_in_round].select(
                    fake).iloc[0]
            self.state.loc[pick.name, 'round_claimed'] = self.round

            self.state.loc[pick.name, 'claimed_by'] = 0 if team == 'my_team' else int(
                team.split('_')[-1])

            self.position_in_round += 1
            if self.position_in_round == self.n_players:
                self.position_in_round = 0
                self.round += 1
                self.teams.reverse()
                self.team_objs.reverse()
                self.ai_strategies.reverse()
            if self.round == 17:
                print('Game Over')
                self.done = 1
                self.reward = self.check()
                return [self.state, self.reward, self.done, {}]

    def reset(self):
        self.state = self._create_initial_state(self.n_players, self.draft_pos)

        self.teams = [f'opp_team_{i}' for i in range(1, self.n_players)]
        self.teams.insert(self.draft_pos, 'my_team')

        self.team_objs = [Team(name) for name in self.teams]
        self.ai_strategies = [BestOutOfOptions(
            'ai_strat', t) for t in self.team_objs]
        self.round = 0
        self.position_in_round = 0
        self.action_space = spaces.Discrete(self.state.shape[0] - 1)

        total_shape = self.state.shape[1] * self.state.shape[0]

        obs_space_box = spaces.Box(
            low=np.repeat(-1, total_shape).reshape(self.state.shape),
            high=np.repeat(1000,  total_shape).reshape(self.state.shape))

        self.observation_space = obs_space_box
        self.done = 0
        self.add = [0, 0]
        self.reward = 0
        self.current_team = ''
        return self.state

    def render(self, mode):
        print(self.state[self.state['claimed_by'] == 0])
        print(self.input_df[self.state['claimed_by'] != -
                            1].groupby(self.state['claimed_by'])['points'].sum())
        print(
            self.input_df.loc[self.state[self.state['claimed_by'] == 0].index])
