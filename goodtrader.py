# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader One.
#
#     Ready Trader One is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader One is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader One.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools
import math
import pandas as pd
import heapq

import matplotlib.pyplot as plt

from struct import error
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

from statistics import mean
from typing import List

from ready_trader_one import BaseAutoTrader, Instrument, Lifespan, Side


LOT_SIZE = 10
POSITION_LIMIT = 1000
VOLUME_LIMIT = 200
TICK_SIZE_IN_CENTS = 100
UPDATE_LIST_SIZE = 5
ORDER_LIMIT = 10

class AutoTrader(BaseAutoTrader):
    """Example Auto-trader.

    When it starts this auto-trader places ten-lot bid and ask orders at the
    current best-bid and best-ask prices respectively. Thereafter, if it has
    a long position (it has bought more lots than it has sold) it reduces its
    bid and ask prices. Conversely, if it has a short position (it has sold
    more lots than it has bought) then it increases its bid and ask prices.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.order_book = OrderBook()
        
        # List of orders structured as [order_id, price, trade_side, volume]
        self.active_orders = [] 

        # Range from support and resistance s
        self.bound_range = 0.001

        # Current positions
        self.position = [0,0]

        # Current volume for ETF
        self.volume = 0

        # Ask prices
        self.ask_price = [0, 0]

        # Bid prices
        self.bid_price = [0, 0]

        # Current sequence number to update on order_book_updates function
        self.order_update_number = [1, 1]

        # Current sequence number to update the on_trade_ticks functiondxs  
        self.trade_update_number = [1,1]

        # List of bid/ask VWAPs for instruments, where list[i] is the bid/ask VWAP for instrument i
        self.vwaps = [[None, None], [None, None]]

        # List of market prices for instruments, where list[i] contains list of market prices for instrument i
        self.market_prices = [[], []]

        # Variables used for SMA BUY SELL strategy
        self.sma_20_prev = 0
        self.sma_100_prev = 0
        self.sma_list = []


        # orderbook for short and long position
        self.position_constitution = []
        heapq.heapify(self.position_constitution)

        # Most recent resistance line value
        self.resist = [0, 0]

        # Most recent support line value
        self.support = [0, 0]

        # Gradient of recent trend
        self.slope = [0, 0]

        # R2 Coefficient
        self.r2 = [0, 0]

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s",
                            client_order_id, error_message.decode())
        if client_order_id != 0:
            self.order_book.remove_order(client_order_id)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
       # Only recalculate average on new sequence number
        if sequence_number > self.order_update_number[instrument]:

            self.calculate_vwap(instrument, ask_prices,
                                ask_volumes, bid_prices, bid_volumes)
            self.calculate_resist(instrument)
            self.calculate_support(instrument)
            self.calculate_regression(instrument)

            if ask_prices[0] != 0:
                self.ask_price[instrument] = ask_prices[0] 
            if bid_prices[0] != 0:
                self.bid_price[instrument] = bid_prices[0]
            
            self.order_update_number[instrument] = sequence_number

            sma_20 = self.calculate_sma(20)
            sma_50 = self.calculate_sma(50)
            sma_100 = self.calculate_sma(100)
            sma_200 = self.calculate_sma(200)
            sma_500 = self.calculate_sma(500)
            self.sma_list.append([self.event_loop.time(),sma_20,sma_50,sma_100,sma_200,sma_500])
            df = pd.DataFrame(self.sma_list,columns=['Time','SMA-20','SMA-50','SMA-100','SMA-200','SMA-500'])
            #fig = df.plot(x="Time",y=["SMA-20","SMA-100"])
            #print(self.sma_list)
            df.to_csv(path_or_buf="/home/posocer/Documents/projects/trader/readyTraderOne/sma_10.csv")
            #plt.savefig("/home/posocer/Documents/projects/trader/readyTraderOne/file.jpg")
            
            self.sma_20_prev = sma_20
            self.sma_100_prev = sma_100
            #print(self.sma_list)
            #print("SMA20",sma_20)
            #print(sequence_number)
            #print("SMA100",sma_100)
            #print(ask_prices,bid_prices)
        
        self.calculate_resist(instrument)
        self.calculate_support(instrument)
        self.calculate_regression(instrument)

        

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when when of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """

        #currently we do not account for market orders as we do not do them
        self.order_book.amend_order(volume,client_order_id)

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:

        if sequence_number > self.trade_update_number[instrument] and (bid_prices[0] != 0 or ask_prices[0] != 0):
            self.calculate_market_price(
                instrument, ask_prices, ask_volumes, bid_prices, bid_volumes)
            self.trade_update_number[instrument] = sequence_number

        if bid_prices[0] != 0 and self.support[instrument] <= bid_prices[0] <= self.support[instrument] * (1 + self.bound_range):
            lot_size = 1
            if self.slope[instrument] > 0 and self.r2[instrument] >= 0.1:
                lot_size *= 2
            self.insert_order_buy(bid_prices[0], lot_size, instrument)

        if ask_prices[0] != 0 and self.resist[instrument] * (1 - self.bound_range) <= ask_prices[0] <= self.resist[instrument]:
            lot_size = 1
            if self.slope[instrument] < 0 and self.r2[instrument] >= 0.1:
                lot_size *= 2
            
            self.insert_order_sell(ask_prices[0], lot_size, instrument)

    def calculate_market_price(self, instrument: int, ask_prices: List[int], ask_volumes: List[int],
                               bid_prices: List[int], bid_volumes: List[int]) -> None:
        total_volume = 0
        total_price = 0
        for i in range(UPDATE_LIST_SIZE):
            total_volume += bid_volumes[i] + ask_volumes[i]
            total_price += bid_volumes[i] * \
                bid_prices[i] + ask_volumes[i] * ask_prices[i]

            average_price = total_price // total_volume
            self.market_prices[instrument].append(average_price)

    def calculate_vwap(self, instrument: int, ask_prices: List[int],
                       ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        bid_total_volume = 0
        bid_total_value = 0

        ask_total_volume = 0
        ask_total_value = 0

        for i in range(UPDATE_LIST_SIZE):
            # VWAP from bid orders
            bid_total_volume += bid_volumes[i]
            bid_total_value += bid_volumes[i] * bid_prices[i]

            # VWAP from ask orders
            ask_total_volume += ask_volumes[i]
            ask_total_value += ask_volumes[i] * ask_prices[i]

        bid_vwap = math.floor(bid_total_value / bid_total_volume) if (bid_total_volume != 0) else  self.vwaps[instrument][0]
        ask_vwap = math.ceil(ask_total_value / ask_total_volume)  if (ask_total_volume != 0) else  self.vwaps[instrument][1]

        self.vwaps[instrument][0] = bid_vwap
        self.vwaps[instrument][1] = ask_vwap

    def calculate_sma(self, period: int):
        # currently hardcoded for the ETF or something
        tail = self.market_prices[1][-period:]

        return mean(tail)

    def insert_order_buy(self, price: int, amount: int, instrument: int):
        buy_amount = min(LOT_SIZE*amount, POSITION_LIMIT - abs(self.position[0]) - abs(self.position[1]))
        if buy_amount <= 0 and self.position[0] + self.position[1] >= POSITION_LIMIT:
            return
        if self.order_book.num_orders == ORDER_LIMIT:
            print("Buy amount is: ", buy_amount, " Current volume is: ", self.order_book.volume, " Position is: ", self.order_book.position)
            print("-----------------------------")
            order_id = self.order_book.remove_least_useful_order(Side.BUY)
            self.send_cancel_order(order_id)
            return
        bid_id = next(self.order_ids)

        self.order_book.add_bid(price-100,buy_amount,bid_id)
        if(buy_amount != 0):
            self.send_insert_order(bid_id, Side.BUY, price - 100, buy_amount, Lifespan.GOOD_FOR_DAY)

    def insert_order_sell(self, price: int, amount: int, instrument: int):
        sell_amount = min(LOT_SIZE*amount, POSITION_LIMIT - abs(self.position[0]) - abs(self.position[1]))
        if sell_amount <= 0 and abs(self.position[0] + self.position[1]) >= POSITION_LIMIT:
            return
        if self.order_book.num_orders == ORDER_LIMIT:
            print("Sell amount is: ", sell_amount, " Current volume is: ", self.order_book.volume, " Position is: ", self.order_book.position)
            print("-----------------------------")
            order_id = self.order_book.remove_least_useful_order(Side.SELL)
            self.send_cancel_order(order_id)
            return

        sell_id = next(self.order_ids)

        self.order_book.add_ask(price+100,sell_amount,sell_id)
        if(sell_amount != 0):
            self.send_insert_order(sell_id, Side.SELL, price + 100, sell_amount, Lifespan.GOOD_FOR_DAY)

    def calculate_resist(self, instrument: int) -> None:
        if len(self.market_prices[instrument]) < 500:
            return

        prices = np.array(self.market_prices[instrument][-500:])
        midpoint = (self.vwaps[instrument][0] + self.vwaps[instrument][1]) / 2
        peaks, _ = find_peaks(prices, height=midpoint)

        res: np.ndarray = prices[peaks]
        if res.size == 0:
            return

        self.resist[instrument] = res.sum() / res.size

    def calculate_support(self, instrument: int) -> None:
        if len(self.market_prices[instrument]) < 500:
            return

        prices = np.array(self.market_prices[instrument][-500:])
        midpoint = (self.vwaps[instrument][0] + self.vwaps[instrument][1]) / 2
        reverse_prices = (prices - midpoint) * -1
        peaks, _ = find_peaks(reverse_prices, height=0)
        support: np.ndarray = prices[peaks]
        if support.size == 0:
            return

        self.support[instrument] = support.sum() / support.size

    def calculate_regression(self, instrument: int) -> None:
        if len(self.market_prices[instrument]) < 1000:
            return

        prices = np.array(self.market_prices[instrument][-1000:])
        result = linregress(list(range(1000)), prices)

        self.slope[instrument] = result.slope
        self.r2[instrument] = result.rvalue**2

class OrderBook():
    def __init__(self):
        #actual position
        self.position = 0

        #position if all orders in book are executed
        self.position_after_orders = 0

        #volume of current orders on the market
        self.volume = 0

        #number of orders on market
        self.num_orders = 0
        
        #some position trend variable


        #bids and asks will be structured [price, vol, order_id]
        #bids and ask list will be sorted lowest to highest price
        self.asks = []
        self.bids = []
    
    def add_bid(self, price: int, vol: int, order_id: int):
        """
        vol is updated with a value which is legal to be put into the exchange
        FUNCTION DOES NOT SEND AN INSERT ORDER TO EXCHANGE
        """
        if self.num_orders < ORDER_LIMIT:
            if vol + self.volume > VOLUME_LIMIT:
                vol = VOLUME_LIMIT - self.volume
            if vol <= 0:
                return
            if not self.bids:
                self.bids.append([price, vol, order_id])
            else:
                insert = False
                for i in range(len(self.bids)):
                    if(self.bids[i][0] >= price):
                        self.bids.insert(i,[price,vol,order_id])
                        insert = True
                        break
                if not insert:
                    self.bids.append([price,vol,order_id])

            self.position_after_orders += vol
            self.volume += vol
            self.num_orders += 1
        else:
            vol = 0

    def add_ask(self, price: int, vol: int, order_id: int):
        """
        FUNCTION DOES NOT SEND AN INSERT ORDER TO EXCHANGE
        """
        if self.num_orders < ORDER_LIMIT:
            if vol + self.volume > VOLUME_LIMIT:
                vol = VOLUME_LIMIT - self.volume
            if vol <= 0:
                return 
            if not self.asks:
                self.asks.append([price, vol, order_id])
            else:
                insert = False
                for i in range(len(self.asks)):
                    if(self.asks[i][0] >= price):
                        self.asks.insert(i,[price,vol,order_id])   
                        insert = True
                        break
                if not insert:
                    self.asks.append([price,vol,order_id])

            self.position_after_orders -= vol
            self.volume += vol
            self.num_orders += 1
        else:
            vol = 0

    def amend_order(self, vol: int, order_id: int):
        """reduces the volume of the order as it has been partially filled/filled, if volume reaches 0 order will be removed from orders

        FUNCTION DOES NOT SEND A CANCEL ORDER TO EXCHANGE"""
        for bid in self.bids:
            if bid[2] == order_id:
                self.position_after_orders += vol
                self.volume -= vol
                self.position += vol
                if bid[1]-vol == 0:
                    self.bids.remove(bid)
                    self.num_orders -= 1
                else:
                    bid[1] -= vol
                return
        
        for ask in self.asks:
            if ask[2] == order_id:
                self.position_after_orders -= vol
                self.volume -= vol
                self.position -= vol
                if ask[1]-vol == 0:
                    self.asks.remove(ask)
                    self.num_orders -= 1
                else:
                    ask[1] -= vol
                return

    def remove_order(self, order_id: int):
        for bid in self.bids:
            if bid[2] == order_id:
                self.num_orders -= 1
                self.position_after_orders -= bid[1]
                self.volume -= bid[1]
                self.bids.remove(bid)
                return
        
        for ask in self.asks:
            if ask[2] == order_id:
                self.num_orders -= 1
                self.position_after_orders += ask[1]
                self.volume -= ask[1]
                self.asks.remove(ask)
                return

    def remove_least_useful_order(self, side: int):
        """removes the order on the side specified which is furthest away, in terms of price, from the current market price

        returns: order_id of order removed
        
        FUNCTION DOES NOT SEND A CANCEL ORDER TO THE EXCHANGE
        """
        if(side == Side.BUY):
            #If not empty
            if self.bids:
                order = self.bids.pop(0)
                self.position_after_orders -= order[1]
            elif self.asks:
                order = self.asks.pop() #not specifying gives -1
                self.position_after_orders += order[1]
        else:
            if self.asks:
                order = self.asks.pop() #not specifying gives -1
                self.position_after_orders += order[1]
            elif self.bids:
                order = self.bids.pop(0)   
                self.position_after_orders -= order[1]
        
        self.volume -= order[1]
        self.num_orders -= 1
        return order[2]     