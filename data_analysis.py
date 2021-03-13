import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'/home/posocer/Documents/projects/trader/readyTraderOne/match_events.csv')

trades = df.loc[(df['Competitor'] == "GoodTrader") & (df['Operation'] == "Trade")]

buys = trades.loc[trades['Side'] == "B"]

sells = trades.loc[trades['Side'] == "A"]

plot1 = buys.plot(x="Time",y="Price", label="Buy Price")

sells.plot(x="Time",y="Price",ax=plot1, label="Sell Price")
plt.show()

buys["Total"] = buys.Volume * buys.Price
sells["Total"] = sells.Volume * sells.Price

print(sells.sum(axis=0)['Total']-buys.sum(axis=0)['Total'])