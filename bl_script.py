import numpy as np
import pandas as pd


def main():
    # Constants

    TAU = 0.05

    # Date of the quarter
    date = "2022-09-30"
    oneyearbefore = date[:2] + str(int(date[2:4]) - 1) + date[4:]
    nviews = 4


    result = pd.DataFrame(columns=["Portfolio", "Market pft excess return", "BL pft excess return",
                                   "GJI0", "JHUCXEHE", "JEEXXITE", "ER00", "GCXZ", "NDDLEMU", "MXWOHEUR"])

    indices_price = pd.read_excel("indices_values.xlsx", index_col=0)
    market_weights = pd.DataFrame([{"EGB0":0, "GJI0":0.0585, "JHUCXEHE":0.1819, "JEEXXITE":0.123, "ER00":0.1451, "GCXZ":0.1455, "NDDLEMU":0.0925, "MXWOHEUR":0.2535}], index=["Weight"])

    risk_free_realized_return = (indices_price.loc[date, 'EGB0'] /
                                 indices_price.loc[oneyearbefore, 'EGB0']) - 1

    risk_free_rtns = indices_price["EGB0"].pct_change()
    risk_free_rtns.dropna(inplace=True)
    risk_free_rtns.drop(risk_free_rtns[risk_free_rtns.index < "31-01-01"].index, inplace=True)
    risk_free_rtns = pd.DataFrame(risk_free_rtns)

    indices_price.drop(columns=['EGB0'], inplace=True)
    indices_rtns = indices_price.pct_change()
    indices_rtns.dropna(how='any', inplace=True)
    indices_rtns.drop(indices_rtns[indices_rtns.index > date].index, inplace=True)

    # From each monthly return of every indices subtract the corresponding risk-free monthly return 
    for i, row in indices_rtns.iterrows():
        rf_rtn = risk_free_rtns.at[i, 'EGB0']
        indices_rtns.at[i, 'GJI0'] -= rf_rtn
        indices_rtns.at[i, 'JHUCXEHE'] -= rf_rtn
        indices_rtns.at[i, 'JEEXXITE'] -= rf_rtn
        indices_rtns.at[i, 'ER00'] -= rf_rtn
        indices_rtns.at[i, 'GCXZ'] -= rf_rtn
        indices_rtns.at[i, 'NDDLEMU'] -= rf_rtn
        indices_rtns.at[i, 'MXWOHEUR'] -= rf_rtn

    # Drop the risk-free asset from the potfolio weights DataFrame
    portfolio_weights = market_weights.drop(columns=['EGB0'])

    # Calculate the benchmark excess return
    benchmark_er = np.dot(indices_rtns, np.transpose(portfolio_weights))

    # Compute historical price series of the portfolio
    benchmark_series = np.empty(len(benchmark_er) + 1)
    benchmark_series[0] = 100

    for i in range(benchmark_series.size):
        if i != 0:
            benchmark_series[i] = benchmark_series[i - 1] * (1 + benchmark_er[i - 1])

    # Compute the log monthly returns of the benchmark
    log_returns = np.empty(len(benchmark_er))

    for i in range(log_returns.size):
        log_returns[i] = np.log(benchmark_series[i + 1] / benchmark_series[i])

    # Mean monthly return of the optimal portfolio
    mean_monthly_return = np.mean(log_returns[:246])

    monthly_variance = np.var(log_returns[:246])

    # Portfolio Lambda
    lmd = mean_monthly_return / monthly_variance

    # Covariance Matrix 
    cov_matrix = indices_rtns.loc[:oneyearbefore, :].cov()

    # sigma matrix * tau
    cov_tau = cov_matrix * TAU

    market_implied_rtns = lmd * 12 * np.matmul(cov_matrix, np.transpose(np.array(portfolio_weights)))

    # print(market_implied_rtns)

    market_neutral_Q = np.zeros(nviews, dtype=float)

    market_neutral_Q[0] = market_implied_rtns.iloc[5] - market_implied_rtns.iloc[6]
    market_neutral_Q[1] = market_implied_rtns.iloc[0] - market_implied_rtns.iloc[2]
    market_neutral_Q[2] = market_implied_rtns.iloc[3] - \
                          (0.25 * market_implied_rtns.iloc[0] + 0.75 * market_implied_rtns.iloc[2])
    market_neutral_Q[3] = market_implied_rtns.iloc[5] - \
                          (0.25 * market_implied_rtns.iloc[0] + 0.75 * market_implied_rtns.iloc[2])

    views = pd.DataFrame([{"1": 0.0549, "2": 0, "3": 0.02, "4": 0.18}])

    # vector Q containig the relative/absolute views 

    i = 0

    for index, row in views.iterrows():

        Q = np.zeros(nviews, dtype=float)

        if row[1] != 9999:
            Q[0] = row[1]
        else:
            Q[0] = market_neutral_Q[0]

        if row[2] != 9999:
            Q[1] = row[2]
        else:
            Q[1] = market_neutral_Q[1]

        if row[3] != 9999:
            Q[2] = row[3]
        else:
            Q[2] = market_neutral_Q[2]

        if row[4] != 9999:
            Q[3] = row[4]
        else:
            Q[3] = market_neutral_Q[3]

        # print(Q)

        p_array = np.array([[0, 0, 0, 0, 0, 1, -1],
                            [1, 0, -1, 0, 0, 0, 0],
                            [-0.25, 0, -0.75, 1, 0, 0, 0],
                            [-0.25, 0, -0.75, 0, 0, 1, 0]])

        P_matrix = pd.DataFrame(p_array,
                                columns=['GJI0', 'JHUCXEHE', 'JEEXXITE', 'ER00', 'GCXZ', 'NDDLEMU', 'MXWOHEUR'])

        # print(P_matrix)

        Omega = np.matmul(np.matmul(np.array(P_matrix), cov_tau), np.transpose(np.array(P_matrix)))
        Omega = np.diag(np.diag(Omega))

        # print(Omega)

        bl1 = np.linalg.inv(np.linalg.inv(np.array(cov_tau)) + np.matmul(np.matmul(np.transpose(np.array(P_matrix)),
                                                                                   np.linalg.inv(Omega)), P_matrix))

        bl2 = np.matmul(np.linalg.inv(np.array(cov_tau)), np.array(market_implied_rtns)).flatten() + \
              np.matmul(np.matmul(np.transpose(np.array(P_matrix)), np.linalg.inv(Omega)), Q).flatten()

        bl_excess_returns = pd.DataFrame(np.matmul(bl1, bl2),
                                         index=['GJI0', 'JHUCXEHE', 'JEEXXITE', 'ER00', 'GCXZ', 'NDDLEMU', 'MXWOHEUR'],
                                         columns=['excess_return'])

        # print(bl_excess_returns)

        bl_cov = bl1 + cov_matrix

        opt_weight = pd.DataFrame(np.matmul(np.linalg.inv(12 * lmd * np.array(bl_cov)), np.array(bl_excess_returns)),
                                  index=['GJI0', 'JHUCXEHE', 'JEEXXITE', 'ER00', 'GCXZ', 'NDDLEMU', 'MXWOHEUR'],
                                  columns=['weight'])

        # print(opt_weight)

        ptf_annual_rtn = np.matmul(np.transpose(np.array(opt_weight['weight'])), np.array(bl_excess_returns))

        ptf_annual_variance = 12 * np.matmul(np.matmul(np.transpose(np.array(opt_weight['weight'])), bl_cov),
                                             np.array(opt_weight))

        ptf_annual_std = np.sqrt(ptf_annual_variance)

        lmbd = ptf_annual_rtn / ptf_annual_variance

        market_ptf_exp_return = np.matmul(np.array(portfolio_weights), np.array(market_implied_rtns))

        market_exp_variance = 12 * np.matmul(np.matmul(np.array(portfolio_weights), np.array(cov_matrix)),
                                             np.transpose(np.array(portfolio_weights)))

        market_exp_std = np.sqrt(market_exp_variance)

        market_lambda = market_ptf_exp_return / market_exp_variance

        constant_weights_adjusted = (1 - np.sum(np.array(opt_weight))) / 7

        bl_adjusted_weights = opt_weight + constant_weights_adjusted

        bl_ptf_exp_return = np.matmul(np.array(np.transpose(opt_weight)), np.array(bl_excess_returns))

        bl_exp_variance = 12 * np.matmul(np.matmul(np.array(np.transpose(opt_weight)), np.array(bl_cov)),
                                         np.array(opt_weight))

        bl_exp_std = np.sqrt(bl_exp_variance)

        bl_lambda = bl_ptf_exp_return / bl_exp_variance

        # Result

        realized_returns = (indices_price.loc[date] / indices_price.loc[oneyearbefore]) - 1

        realized_excess_returns = realized_returns - risk_free_realized_return

        market_ptf_excess_return = np.matmul(np.array(realized_excess_returns),
                                             np.array(np.transpose(portfolio_weights)))

        bl_ptf_excess_return = np.matmul(np.array(realized_excess_returns), np.array(bl_adjusted_weights))

        result.loc[i] = [index] + [market_ptf_excess_return[0]] + [bl_ptf_excess_return[0]] + \
                        [bl_adjusted_weights.weight.values[0]] + [bl_adjusted_weights.weight.values[1]] + \
                        [bl_adjusted_weights.weight.values[2]] + [bl_adjusted_weights.weight.values[3]] + \
                        [bl_adjusted_weights.weight.values[4]] + [bl_adjusted_weights.weight.values[5]] + \
                        [bl_adjusted_weights.weight.values[6]]

        i += 1

    result.to_excel("blresult.xlsx", index=False)


if __name__ == '__main__':
    main()
