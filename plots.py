import pandas as pd
import matplotlib.pyplot as plt

with open('omega.out', 'r') as f:
    df = pd.read_csv(f, delimiter='\t')
    df = df[:360]
    plt.plot(df['bin start'], df['hist_DD'])
    plt.plot(df['bin start'], df['hist_RR'])
    plt.plot(df['bin start'], df['hist_DR'])
    plt.legend(['hist_DD', 'hist_RR', 'hist_DR'], title='Histograms')
    plt.xlabel('Angle (in degree)')
    plt.ylabel('Amount of galaxies pairs')
    plt.xticks(range(0, 91, 10))
    plt.savefig('img/hist.png')
    plt.close()

    plt.bar(df['bin start'], df['omega'])
    plt.xlabel('Angle (in degree)')
    plt.ylabel('Difference')
    plt.xticks(range(0, 91, 10))
    plt.savefig('img/omega.png')
    plt.close()