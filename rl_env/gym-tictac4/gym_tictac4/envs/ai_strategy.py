import pandas as pd
from copy import deepcopy


class Team:

    def __init__(self, name):
        self.name = name

        self.min_qb = 2
        self.min_rb = 2
        self.min_wr = 1
        self.min_te = 1
        self.min_k = 1
        self.min_dst = 1
        self.min_sum_rb_wr_te = 5

    def added(self, pos):
        if pos == 'QB':
            self.min_qb -= 1
        if pos == 'RB':
            self.min_rb -= 1
            self.min_sum_rb_wr_te -= 1
        if pos == 'WR':
            self.min_wr -= 1
            self.min_sum_rb_wr_te -= 1
        if pos == 'TE':
            self.min_te -= 1
            self.min_sum_rb_wr_te -= 1
        if pos == 'K':
            self.min_k -= 1
        if pos == 'DST':
            self.min_dst -= 1


class Strategy:

    def __init__(self, name, team, epsilon=.05):
        self.name = name
        self.epsilon = epsilon
        self.team = team

    def update_state(self, state):
        state['current_pick'] = self.team
        state['rnd'] += 1
        pick = self.select(state)
        state.drop(index=pick.index, inplace=True)
        state[team] = state[team].append(pick)

    def _after_filter_select(self, state):

        return state.iloc[0]

    def select(self, state):
        # Defines what player will be selected as a row in the dataframe

        revised_state = self.filter_state(state)
        revised_state = revised_state[revised_state['claimed_by'] == -1]
        return self._after_filter_select(revised_state)

    def filter_state(self, state):
        copied = deepcopy(state)
        growing_bool = state['position'].notnull()

        possible_positions = ['QB', 'RB', 'TE', 'WR', 'DST', 'K']

        rnk_dict = pd.Series(possible_positions, index=possible_positions
                             ).rank(method='dense').to_dict()

        if all([self.team.min_rb <= 0,
                self.team.min_wr <= 0,
                self.team.min_te <= 0,
                self.team.min_k <= 0,
                self.team.min_dst <= 0,
                self.team.min_sum_rb_wr_te <= 0]):
            if self.team.min_qb == 0:
                return copied
            elif self.team.min_qb == -1:
                copied = copied[copied['position'] != rnk_dict['QB']]
                return copied

        if self.team.min_qb == 0:
            growing_bool = growing_bool & (
                state['position'] != rnk_dict['QB'])
        if self.team.min_rb == 0:
            if self.team.min_sum_rb_wr_te == 0:
                growing_bool = growing_bool & (
                    state['position'] != rnk_dict['RB'])
        elif self.team.min_rb == -1:
            growing_bool = growing_bool & (
                state['position'] != rnk_dict['RB'])

        if self.team.min_wr == 0:
            if self.team.min_sum_rb_wr_te == 0:
                growing_bool = growing_bool & (
                    state['position'] != rnk_dict['WR'])
        elif self.team.min_wr == -1:
            growing_bool = growing_bool & (
                state['position'] != rnk_dict['WR'])

        if self.team.min_te == 0:
            if self.team.min_sum_rb_wr_te == 0:
                growing_bool = growing_bool & (
                    state['position'] != rnk_dict['TE'])
        elif self.team.min_te == -1:
            growing_bool = growing_bool & (
                state['position'] != rnk_dict['TE'])

        if self.team.min_k == 0:
            growing_bool = growing_bool & (
                state['position'] != rnk_dict['K'])
        if self.team.min_dst == 0:
            growing_bool = growing_bool & (
                state['position'] != rnk_dict['DST'])
        if self.team.min_rb == 0:
            growing_bool = growing_bool & (
                state['position'] != rnk_dict['RB'])

        if self.team.min_sum_rb_wr_te == 1:
            if self.team.min_rb > 0:
                growing_bool = growing_bool & (
                    ~state['position'].isin([rnk_dict['TE'], rnk_dict['WR']]))
            if self.team.min_wr > 0:
                growing_bool = growing_bool & (
                    ~state['position'].isin([rnk_dict['RB'], rnk_dict['TE']]))
            if self.team.min_te > 0:
                growing_bool = growing_bool & (
                    ~state['position'].isin([rnk_dict['RB'], rnk_dict['WR']]))

        if growing_bool.mean() == 0:
            import pdb
            pdb.set_trace()

        copied = copied[growing_bool]
        return copied


class Random(Strategy):

    def _after_filter_select(self, state):
        return state.sample(1)


class BestOutOfOptions(Strategy):

    def _after_filter_select(self, state):
        return state.sort_values(by='points', ascending=False).head(1)
