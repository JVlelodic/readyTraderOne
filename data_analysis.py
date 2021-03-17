import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

lis = [10000,6000,300,6030,2002]
std = np.std(lis)
print(std)
# df = pd.read_csv(r'/home/posocer/Documents/projects/trader/readyTraderOne/match_events.csv')
# fig = df.plot(x="Time",y=["SMA-20","SMA-100"])
# plt.savefig("/home/posocer/Documents/projects/trader/readyTraderOne/file.jpg")
# trades = df.loc[(df['Competitor'] == "GoodTrader") & (df['Operation'] == "Trade")]

# buys = trades.loc[trades['Side'] == "B"]

# sells = trades.loc[trades['Side'] == "A"]

# plot1 = buys.plot(x="Time",y="Price", label="Buy Price")

# sells.plot(x="Time",y="Price",ax=plot1, label="Sell Price")
# plt.show()

# buys["Total"] = buys.Volume * buys.Price
# sells["Total"] = sells.Volume * sells.Price

# print(sells.sum(axis=0)['Total']-buys.sum(axis=0)['Total'])