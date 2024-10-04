import numpy as np
import matplotlib.pyplot as plt
import argparse

def sliding_window_average(data, window_size=11):
    result = np.zeros_like(data)
    half_window = window_size // 2
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        result[i] = np.mean(data[start:end])
    return result

def truncate_data(data_list, data_length):
    sub_lengths = [len(sublist) for sublist in data_list] 
    length = min(sub_lengths)
    length = min(length, data_length)
    truncated_data = [sublist[:length] for sublist in data_list]
    return np.array(truncated_data)


if __name__ == "__main__":
    """
    3-player Kuhn Poker ([0,2] vs [1])
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--expr", type=str, default="2v1", help="experiment to plot", choices=["2v1", "2v2"])
    parser.add_argument("--result_path", type=str, default="../results", help="path of results")
    parser.add_argument("--eval_interval", type=int, default=10, help="evaluation interval of the result data of 2v2Kuhn")
    args = parser.parse_args()

    expr = args.expr
    result_path = args.result_path

    if expr == "2v1":
        scenario_list = ['2v1Kuhn/r10m3', '2v1Kuhn/r13m2', '2v1Kuhn/r13m3', '2v1Leduc/r4s1m1m1', '2v1Leduc/r4s2m1m1', '2v1Leduc/r5s1m1m1']
        folders = []
        for scenario in scenario_list:
            folders.append(result_path + f"/{scenario}lr1.0")
        rows = 2
        cols = 3
        fig, axs = plt.subplots(rows, cols, figsize=(14, 8))

        for i, folder in enumerate(folders):

            ND_nashconvs = []
            IPGBR_nashconvs = []
            IBRBR_nashconvs = []
            Cycle_BR_nashconvs = []
            for seed in range(5):
                try:
                    ND_nashconvs.append(np.load(f'{folder}/seed{seed}/ND/nashconvs.npy'))
                    IPGBR_nashconvs.append(np.load(f'{folder}/seed{seed}/IPG-BR/nashconvs.npy'))
                    IBRBR_nashconvs.append(np.load(f'{folder}/seed{seed}/IBR-BR/nashconvs.npy'))
                    Cycle_BR_nashconvs.append(np.load(f'{folder}/seed{seed}/Cycle BR/nashconvs.npy'))
                except:
                    print(f"Skipping unexist seed {seed} of {folder.split('/')[-2]} {folder.split('/')[-1]}")

            if i < 3:
                data_length = 5000
            else:
                data_length = 10000

            try:
                ND_nashconvs = truncate_data(ND_nashconvs, data_length)
                ND_nashconvs = np.array([sliding_window_average(ND_nashconvs[i,:], window_size=1) for i in range(len(ND_nashconvs))])
                median_nc = np.median(ND_nashconvs, axis=0)
                axs[i//cols, i%cols].plot(median_nc, color='r', label=f'ND') # red
                lower_quartile = np.percentile(ND_nashconvs, 25, axis=0)
                upper_quartile = np.percentile(ND_nashconvs, 75, axis=0)
                axs[i//cols, i%cols].fill_between(range(len(median_nc)), lower_quartile, upper_quartile, color='r', alpha=0.2)

                IPGBR_nashconvs = truncate_data(IPGBR_nashconvs, data_length)
                IPGBR_nashconvs = np.array([sliding_window_average(IPGBR_nashconvs[i,:], window_size=9) for i in range(len(IPGBR_nashconvs))])
                median_nc = np.median(IPGBR_nashconvs, axis=0)
                axs[i//cols, i%cols].plot(median_nc, color='#0000FF', label=f'IPG-BR')
                lower_quartile = np.percentile(IPGBR_nashconvs, 25, axis=0)
                upper_quartile = np.percentile(IPGBR_nashconvs, 75, axis=0)
                axs[i//cols, i%cols].fill_between(range(len(median_nc)), lower_quartile, upper_quartile, color='#0000FF', alpha=0.2)

                IBRBR_nashconvs = truncate_data(IBRBR_nashconvs, data_length)
                IBRBR_nashconvs = np.array([sliding_window_average(IBRBR_nashconvs[i,:], window_size=5) for i in range(len(IBRBR_nashconvs))])
                median_nc = np.median(IBRBR_nashconvs, axis=0)
                axs[i//cols, i%cols].plot(median_nc, color='orange', label=f'IBR-BR')
                lower_quartile = np.percentile(IBRBR_nashconvs, 25, axis=0)
                upper_quartile = np.percentile(IBRBR_nashconvs, 75, axis=0)
                axs[i//cols, i%cols].fill_between(range(len(median_nc)), lower_quartile, upper_quartile, color='orange', alpha=0.2)

                Cycle_BR_nashconvs = truncate_data(Cycle_BR_nashconvs, data_length)
                Cycle_BR_nashconvs = np.array([sliding_window_average(Cycle_BR_nashconvs[i,:], window_size=5) for i in range(len(Cycle_BR_nashconvs))])
                median_nc = np.median(Cycle_BR_nashconvs, axis=0)
                axs[i//cols, i%cols].plot(median_nc, color='c', label=f'Cycle BR')
                lower_quartile = np.percentile(Cycle_BR_nashconvs, 25, axis=0)
                upper_quartile = np.percentile(Cycle_BR_nashconvs, 75, axis=0)
                axs[i//cols, i%cols].fill_between(range(len(median_nc)), lower_quartile, upper_quartile, color='c', alpha=0.2)
            except:
                pass

            axs[i//cols, i%cols].set_ylabel('NashConv')
            axs[i//cols, i%cols].set_xlabel('Iteration', fontsize=7)
            axs[i//cols, i%cols].grid()
            axs[i//cols, i%cols].set_yscale('log')
            axs[0, 0].legend(loc = 'lower left', fontsize=10)
            if i < 3:
                    axs[i//cols, i%cols].set_title(f"3K{scenario_list[i][-5:]}")
            else:
                axs[i//cols, i%cols].set_title(f"3L{scenario_list[i][-8:-4]}")


    elif expr == "2v2":
        scenario_list = ['r5m1', 'r5m2', 'r5m3', 'r6m2', 'r7m1', 'r7m2']
        folders = []
        for scenario in scenario_list:
            folders.append(result_path + f"/2v2Kuhn/{scenario}lr1.0")
        rows = 2
        cols = 3
        fig, axs = plt.subplots(rows, cols, figsize=(14, 8))

        for i, folder in enumerate(folders):        

            ND_nashconvs = []
            IPGIBR_nashconvs = []
            IPGminBR_nashconvs = []
            IPGminBR_dual_nashconvs = []
            IPGIBR_dual_nashconvs = []
            IBRIBR_nashconvs = []
            IBRminBR_nashconvs = []
            Cycle_BR_nashconvs = []
            for seed in range(5):
                try:
                    ND_nashconvs.append(np.load(f'{folder}/seed{seed}/ND/nashconvs.npy'))
                    IPGminBR_dual_nashconvs.append(np.load(f'{folder}/seed{seed}/IPG-MinBR Dual/nashconvs.npy'))
                    IPGminBR_nashconvs.append(np.load(f'{folder}/seed{seed}/IPG-MinBR/nashconvs.npy'))
                    IPGIBR_dual_nashconvs.append(np.load(f'{folder}/seed{seed}/IPG-IBR Dual/nashconvs.npy'))
                    IPGIBR_nashconvs.append(np.load(f'{folder}/seed{seed}/IPG-IBR/nashconvs.npy'))
                    IBRminBR_nashconvs.append(np.load(f'{folder}/seed{seed}/IBR-MinBR/nashconvs.npy'))
                    IBRIBR_nashconvs.append(np.load(f'{folder}/seed{seed}/IBR-IBR/nashconvs.npy'))
                    Cycle_BR_nashconvs.append(np.load(f'{folder}/seed{seed}/Cycle BR/nashconvs.npy'))         
                except Exception as e:
                    print(f"Skipping unexist seed {seed} of {folder.split('/')[-2]} {folder.split('/')[-1]}")
            
            data_length = 10000

            eval_interval = args.eval_interval

            try:
                ND_nashconvs = truncate_data(ND_nashconvs, data_length)
                ND_nashconvs = np.array([sliding_window_average(ND_nashconvs[i,:], window_size=1) for i in range(len(ND_nashconvs))])
                median_nc = np.median(ND_nashconvs, axis=0)
                axs[i//cols, i%cols].plot(median_nc, color='r', label=f'ND') # red
                lower_quartile = np.percentile(ND_nashconvs, 25, axis=0)
                upper_quartile = np.percentile(ND_nashconvs, 75, axis=0)
                axs[i//cols, i%cols].fill_between(range(len(median_nc)), lower_quartile, upper_quartile, color='r', alpha=0.2)

                IPGminBR_dual_nashconvs = truncate_data(IPGminBR_dual_nashconvs, data_length)
                IPGminBR_dual_nashconvs = np.array([sliding_window_average(IPGminBR_dual_nashconvs[i,:], window_size=1) for i in range(len(IPGminBR_dual_nashconvs))])
                median_nc = np.median(IPGminBR_dual_nashconvs, axis=0)
                axs[i//cols, i%cols].plot(range(0, len(median_nc)*eval_interval, eval_interval), median_nc, color='purple', label=f'IPG-MinBR Dual')
                lower_quartile = np.percentile(IPGminBR_dual_nashconvs, 25, axis=0)
                upper_quartile = np.percentile(IPGminBR_dual_nashconvs, 75, axis=0)
                axs[i//cols, i%cols].fill_between(range(0, len(median_nc)*eval_interval, eval_interval), lower_quartile, upper_quartile, color='purple', alpha=0.2)

                IPGminBR_nashconvs = truncate_data(IPGminBR_nashconvs, data_length)
                IPGminBR_nashconvs = np.array([sliding_window_average(IPGminBR_nashconvs[i,:], window_size=1) for i in range(len(IPGminBR_nashconvs))])
                median_nc = np.median(IPGminBR_nashconvs, axis=0)
                axs[i//cols, i%cols].plot(range(0, len(median_nc)*eval_interval, eval_interval), median_nc, color='#0000FF', label=f'IPG-MinBR') # blue
                lower_quartile = np.percentile(IPGminBR_nashconvs, 25, axis=0)
                upper_quartile = np.percentile(IPGminBR_nashconvs, 75, axis=0)
                axs[i//cols, i%cols].fill_between(range(0, len(median_nc)*eval_interval, eval_interval), lower_quartile, upper_quartile, color='#0000FF', alpha=0.2)
              
                IPGIBR_dual_nashconvs = truncate_data(IPGIBR_dual_nashconvs, data_length)
                IPGIBR_dual_nashconvs = np.array([sliding_window_average(IPGIBR_dual_nashconvs[i,:], window_size=1) for i in range(len(IPGIBR_dual_nashconvs))])
                median_nc = np.median(IPGIBR_dual_nashconvs, axis=0)
                axs[i//cols, i%cols].plot(range(0, len(median_nc)*eval_interval, eval_interval), median_nc, color='#C993C5', label=f'IPG-IBR Dual') # pink
                lower_quartile = np.percentile(IPGIBR_dual_nashconvs, 25, axis=0)
                upper_quartile = np.percentile(IPGIBR_dual_nashconvs, 75, axis=0)
                axs[i//cols, i%cols].fill_between(range(0, len(median_nc)*eval_interval, eval_interval), lower_quartile, upper_quartile, color='#C993C5', alpha=0.2)

                IPGIBR_nashconvs = truncate_data(IPGIBR_nashconvs, data_length)
                IPGIBR_nashconvs = np.array([sliding_window_average(IPGIBR_nashconvs[i,:], window_size=1) for i in range(len(IPGIBR_nashconvs))])
                median_nc = np.median(IPGIBR_nashconvs, axis=0)
                axs[i//cols, i%cols].plot(range(0, len(median_nc)*eval_interval, eval_interval), median_nc, color='#3CC33C', label=f'IPG-IBR') # green
                lower_quartile = np.percentile(IPGIBR_nashconvs, 25, axis=0)
                upper_quartile = np.percentile(IPGIBR_nashconvs, 75, axis=0)
                axs[i//cols, i%cols].fill_between(range(0, len(median_nc)*eval_interval, eval_interval), lower_quartile, upper_quartile, color='#3CC33C', alpha=0.2)

                IBRminBR_nashconvs = truncate_data(IBRminBR_nashconvs, data_length)
                IBRminBR_nashconvs = np.array([sliding_window_average(IBRminBR_nashconvs[i,:], window_size=1) for i in range(len(IBRminBR_nashconvs))])
                median_nc = np.median(IBRminBR_nashconvs, axis=0)
                axs[i//cols, i%cols].plot(range(0, len(median_nc)*eval_interval, eval_interval), median_nc, color='orange', label=f'IBR-MinBR') # interval
                lower_quartile = np.percentile(IBRminBR_nashconvs, 25, axis=0)
                upper_quartile = np.percentile(IBRminBR_nashconvs, 75, axis=0)
                axs[i//cols, i%cols].fill_between(range(0, len(median_nc)*eval_interval, eval_interval), lower_quartile, upper_quartile, color='orange', alpha=0.2)

                IBRIBR_nashconvs = truncate_data(IBRIBR_nashconvs, data_length)
                IBRIBR_nashconvs = np.array([sliding_window_average(IBRIBR_nashconvs[i,:], window_size=1) for i in range(len(IBRIBR_nashconvs))])
                median_nc = np.median(IBRIBR_nashconvs, axis=0)
                axs[i//cols, i%cols].plot(range(0, len(median_nc)*eval_interval, eval_interval), median_nc, color='#FD0AFF', label=f'IBR-IBR') # magenta
                lower_quartile = np.percentile(IBRIBR_nashconvs, 25, axis=0)
                upper_quartile = np.percentile(IBRIBR_nashconvs, 75, axis=0)
                axs[i//cols, i%cols].fill_between(range(0, len(median_nc)*eval_interval, eval_interval), lower_quartile, upper_quartile, color='#FD0AFF', alpha=0.2)
            
                Cycle_BR_nashconvs = truncate_data(Cycle_BR_nashconvs, data_length)
                Cycle_BR_nashconvs = np.array([sliding_window_average(Cycle_BR_nashconvs[i,::eval_interval], window_size=21) for i in range(len(Cycle_BR_nashconvs))])
                median_nc = np.median(Cycle_BR_nashconvs, axis=0)
                axs[i//cols, i%cols].plot(range(0, len(median_nc)*eval_interval, eval_interval), median_nc, color='c', label=f'Cycle BR')
                lower_quartile = np.percentile(Cycle_BR_nashconvs, 25, axis=0)
                upper_quartile = np.percentile(Cycle_BR_nashconvs, 75, axis=0)
                axs[i//cols, i%cols].fill_between(range(0, len(median_nc)*eval_interval, eval_interval), lower_quartile, upper_quartile, color='c', alpha=0.2)
            except:
                pass

            axs[i//cols, i%cols].set_ylabel('NashConv')
            axs[i//cols, i%cols].set_xlabel('Iteration')
            axs[i//cols, i%cols].set_title(f"4K{scenario_list[i]}")
            axs[0, 0].legend(loc='lower left', fontsize=10)
            axs[i//cols, i%cols].grid()
            axs[i//cols, i%cols].set_yscale('log')

    else:
        raise ValueError("Invalid experiment")


    fig.tight_layout()
    plt.show()
    fig.savefig(f'{result_path}/{expr}.png')
