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
        
        # Range from support and resistance s
        self.bound_range = 0.001

        # Current sequence number to update on order_book_updates function
        self.order_update_number = [1, 1]

        # Current sequence number to update the on_trade_ticks function
        self.trade_update_number = [1, 1]

        # Current bid/ask for etf
        self.bid = 0
        self.ask = 0
        
        self.bid_vwap = 0
        self.ask_vwap = 0
        
        # List of market price for futures
        self.future_market_price = []
        
        # List of market price for ETF
        self.etf_market_price = []

        # Variables used for SMA BUY SELL strategy
        self.sma_20_prev = 0
        self.sma_100_prev = 0

        # Most recent resistance line value
        self.resist = 0

        # Most recent support line value
        self.support = 0

        # Gradient of recent trend
        self.slope = 0

        # R2 Coefficient
        self.r2 = 0

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
            if instrument == Instrument.ETF:
                self.calculate_resist(instrument)
                self.calculate_support(instrument)
                self.calculate_regression(instrument)

                if ask_prices[0] != 0:
                    self.ask = ask_prices[0] 
                if bid_prices[0] != 0:
                    self.bid = bid_prices[0]
            
            self.order_update_number[instrument] = sequence_number


    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when when of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """

        #currently we do not account for market orders as we do not do them
        self.order_book.amend_order(volume,client_order_id)
        #print("Whats in here",self.order_book.asks,self.order_book.bids)
        #self.logger.info("Calling on_order_filled_message volume: %d client_order_id: %d", volume, client_order_id)

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

    def calculate_market_price(self, instrument: int, ask_prices: List[int], bid_prices: List[int]) -> None:

        last_traded = self.get_last_traded_price(ask_prices, bid_prices)
        if instrument == Instrument.FUTURE:
            self.future_market_price.append(last_traded)
        else:
            #Instrument is ETF
            last_future_price = self.future_market_price[-1]
            etf_price = round(max(last_future_price * 0.998, min(last_future_price * 1.002, last_traded)))
            self.etf_market_price.append(etf_price)
        
    def get_last_traded_price(self, ask_prices: List[int], bid_prices: List[int])-> int:
        last_price = 0
        if ask_prices[0] != 0 and bid_prices[0] != 0:
                if ask_prices[0] > bid_prices[0]:
                    last_price = round((ask_prices[0] + bid_prices[0]) / 2.0)
                else:
                    #Take the bid price
                    last_price = bid_prices[0]
        else:
            # Find latest bid 
            if ask_prices[0] == 0:
                for bids in bid_prices:
                    last_price = bids if bids != 0 else last_price
            # Find latest ask
            else:
                for asks in ask_prices:
                    last_price = asks if asks != 0 else last_price
                
        return last_price
    
    def calculate_sma(self, period: int):
        # currently hardcoded for the ETF or something
        tail = self.market_prices[1][-period:]

        return mean(tail)
    
    def calculate_vwap(self, bid_prices: List[int], ask_prices: List[int], bid_volumes: List[int], ask_volumes: List[int]):

    def calculate_resist(self, instrument: int) -> None:
        if len(self.etf_market_price) < 250:
            return

        prices = np.array(self.etf_market_price[-250:])
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
        #print("Current position: ",self.position, "PLUS ORDERS: ", self.position_after_orders)
        for bid in self.bids:
            if bid[2] == order_id:
                self.position_after_orders += vol
                self.volume -= vol
                self.position += vol
                if bid[1]-vol == 0:
                    #print("removing bid order")
                    self.bids.remove(bid)
                    self.num_orders -= 1
                    #print("Number of orders: ", self.num_orders)
                else:
                    bid[1] -= vol
                return
        
        for ask in self.asks:
            if ask[2] == order_id:
                self.position_after_orders -= vol
                self.volume -= vol
                self.position -= vol
                if ask[1]-vol == 0:
                    #print("removing ask order")
                    self.asks.remove(ask)
                    self.num_orders -= 1
                    #print("Number of orders: ", self.num_orders)
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
        if(side == Side.BUY): #this pop possibly broke
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
        #print("Number of orders: ", self.num_orders)
        return order[2]     