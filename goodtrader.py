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

from statistics import mean
from typing import List

from ready_trader_one import BaseAutoTrader, Instrument, Lifespan, Side


LOT_SIZE = 10
POSITION_LIMIT = 1000
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
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0

        # Current sequence number to update on order_book_updates function
        self.order_update_number = 1

        # Current sequence number to update the on_trade_ticks function
        self.trade_update_number = 1

        # List of bid/ask VWAPs for instruments, where list[i] is the bid/ask VWAP for instrument i
        self.vwaps = [[0,0], [0,0]]

        # List of market prices for instruments, where list[i] contains list of market prices for instrument i
        self.market_prices = [[],[]]

        # Variables used for SMA BUY SELL strategy
        self.sma_20_prev = 0
        self.sma_100_prev = 0

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s",
                            client_order_id, error_message.decode())
        if client_order_id != 0:
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
       # Only recalculate average on new sequence number
        if sequence_number > self.order_update_number: 
            self.calculate_vwap(instrument, ask_prices, ask_volumes, bid_prices, bid_volumes)
            self.order_update_number = sequence_number

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when when of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        if client_order_id in self.bids:
            self.position += volume
        elif client_order_id in self.asks:
            self.position -= volume

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
            if client_order_id == self.bid_id:
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0

            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int], 
                                ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
       
        if sequence_number > self.trade_update_number:
            self.calculate_market_price(instrument, ask_prices, ask_volumes, bid_prices, bid_volumes)
            #print(self.market_prices)

            #Calculate current and previous SMA
            sma_20 = self.calculate_sma(20)
            sma_100 = self.calculate_sma(200)
            #print("SMA20",sma_20)
            #print("SMA100",sma_100)
            if self.sma_20_prev < sma_20 and sma_20 >= sma_100:
                self.bid_id = next(self.order_ids)
                #print(bid_prices)
                self.send_insert_order(self.bid_id, Side.BUY, bid_prices[0]-200, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.bids.add(self.bid_id)
                #print("BUY")
                #BUY
            
            if self.sma_20_prev > sma_20 and sma_20 <= sma_100:
                #SELL
                self.bid_id = next(self.order_ids)
                self.send_insert_order(self.bid_id, Side.SELL, ask_prices[0]+200, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                self.bids.add(self.bid_id)
                #print("SELL")

            self.sma_20_prev = sma_20
            self.sma_100_prev = sma_100

            self.trade_update_number = sequence_number
            

    
    def calculate_market_price(self, instrument: int, ask_prices: List[int], ask_volumes: List[int], 
                                bid_prices: List[int], bid_volumes: List[int]) -> None:
        total_volume = 0
        total_price = 0
        for i in range(UPDATE_LIST_SIZE):
            total_volume += bid_volumes[i] + ask_volumes[i]
            total_price += bid_volumes[i] * bid_prices[i] + ask_volumes[i] * ask_prices[i]

            average_price = total_price // total_volume
            self.market_prices[instrument].append(average_price)
            
            #print("Average price is ", average_price)
            #print(self.market_prices)
            #print("\n")

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

            #VWAP from ask orders
            ask_total_volume += ask_volumes[i]
            ask_total_value += ask_volumes[i] * ask_prices[i]
        
        bid_vwap = math.floor(bid_total_value / bid_total_volume)
        ask_vwap = math.ceil(ask_total_value / ask_total_volume)
        
        self.vwaps[instrument][0] = bid_vwap
        self.vwaps[instrument][1] = ask_vwap
    
    def calculate_sma(self, period: int):
        tail = self.market_prices[1][-period:]#currently hardcoded for the ETF or something
        
        return mean(tail)