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
        self.bids = {}
        self.asks = {}
        
        # List of orders structured as [order_id, price, trade_side, volume]
        self.active_orders = [] 

        # Range from support and resistance lines
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

        # Current sequence number to update the on_trade_ticks function
        self.trade_update_number = [1, 1]

        # List of bid/ask VWAPs for instruments, where list[i] is the bid/ask VWAP for instrument i
        self.vwaps = [[None, None], [None, None]]

        # List of market prices for instruments, where list[i] contains list of market prices for instrument i
        self.market_prices = [[], []]

        # Variables used for SMA BUY SELL strategy
        self.sma_20_prev = 0
        self.sma_100_prev = 0

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
            self.on_order_status_message(client_order_id, 0, 0, 0)
            removed = self.remove_order_id(client_order_id)
            if removed:
                self.volume -= removed[3]

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


    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when when of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        if client_order_id in self.bids:
            self.volume -= volume
            self.position[self.bids.get(client_order_id)] += volume
        
        if client_order_id in self.asks:
            self.volume -= volume
            self.position[self.asks.get(client_order_id)] -= volume

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        if remaining_volume == 0:
            if client_order_id in self.bids:
                self.bids.pop(client_order_id)
            elif client_order_id in self.asks:
                self.asks.pop(client_order_id)
            
            self.remove_order_id(client_order_id)

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

    def cancel_all_orders(self, side: int):
        if side == Side.BUY:
            for order in self.bids:
                self.send_cancel_order(order)
            self.bids.clear()
        elif side == Side.SELL:
            for order in self.asks:
                self.send_cancel_order(order)
            self.asks.clear()

    def insert_order_buy(self, price: int, amount: int, instrument: int):
        buy_amount = min(VOLUME_LIMIT - self.volume, min(LOT_SIZE*amount, POSITION_LIMIT - abs(self.position[0]) - abs(self.position[1])))
        if len(self.active_orders) == 10 or buy_amount == 0:
            print("Sell amount is: ", buy_amount, " Current volume is: ", self.volume, " Position is: ", self.position)
            print("-----------------------------")
            self.free_order_space()
            return
        bid_id = next(self.order_ids)
        self.send_insert_order(bid_id, Side.BUY, price - 100, buy_amount, Lifespan.GOOD_FOR_DAY)
        self.volume += buy_amount
        self.bids[bid_id] = instrument
        self.active_orders.append([bid_id, price, Side.BUY, buy_amount])
        # print(len(self.active_orders))

    def insert_order_sell(self, price: int, amount: int, instrument: int):
        sell_amount = min(VOLUME_LIMIT - self.volume, min(LOT_SIZE*amount, POSITION_LIMIT - abs(self.position[0]) - abs(self.position[1])))
        if len(self.active_orders) == 10 or sell_amount == 0:
            print("Sell amount is: ", sell_amount, " Current volume is: ", self.volume, " Position is: ", self.position)
            print("-----------------------------")
            self.free_order_space()
            return
        sell_id = next(self.order_ids)
        self.send_insert_order(sell_id, Side.SELL, price + 100, sell_amount, Lifespan.GOOD_FOR_DAY)
        self.volume += sell_amount
        self.asks[sell_id] = instrument
        self.active_orders.append([sell_id, price, Side.SELL, sell_amount])
        # print(len(self.active_orders))
    
    def free_order_space(self):
        max_diff = 0
        index = 0
        for i in range(len(self.active_orders)):
            order = self.active_orders[i]
            order_id = order[0]
            price = order[1]
            trade_side = order[2]
            
            ref_price = self.bid_price[1] if (trade_side == Side.SELL) else self.ask_price[1]
            curr_diff = abs(ref_price - price)
            print("Order id is: ", order_id, "Order price is: $", price, "Diff is: ", curr_diff)
            
            if curr_diff > max_diff or (curr_diff == max_diff and order_id > self.active_orders[index][0]):
                max_diff = curr_diff
                index = i
        
        prev_length = len(self.active_orders)
        order = self.active_orders.pop(index)
        print("Popped Order: ", order)  
        assert(len(self.active_orders) + 1 == prev_length)
        self.send_cancel_order(order[0])
        self.volume -= order[3]
    
    def remove_order_id(self, order_id: int):
        for i in range(len(self.active_orders)):
                order = self.active_orders[i]
                if order[0] == order_id:
                    return self.active_orders.pop(i)

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
