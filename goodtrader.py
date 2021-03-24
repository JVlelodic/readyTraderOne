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
import pandas as pd
from struct import error
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress

from statistics import mean
from typing import List, Optional, Dict

from ready_trader_one import BaseAutoTrader, Instrument, Lifespan, Side


LOT_SIZE = 40
POSITION_LIMIT = 1000
VOLUME_LIMIT = 200
TICK_SIZE_IN_CENTS = 100
UPDATE_LIST_SIZE = 5
ORDER_LIMIT = 10
RES_SUP_LENGTH = 250
ROLLING_LIMIT = 50


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

        self.execute_order = 0
        self.order_ids = itertools.count(1)
        self.order_book = OrderBook()
        
        # For rolling one second 50 order limit
        self.last_time_called = []

        # Range from support and resistance s
        self.bound_range = 0.00035
        self.scale_factor = 2

        # Current sequence number to update on order_book_updates function
        self.order_update_number = [1, 1]

        # Current sequence number to update the on_trade_ticks function
        self.trade_update_number = [1, 1]

        # Current price we should put for ETF
        self.bid = -1
        self.ask = -1

        self.bid_vwap = 0
        self.ask_vwap = 0

        # List of market price for futures
        self.future_market_price = [0]

        # List of market price for ETF
        self.etf_market_price = [0]

        # Variables used for SMA BUY SELL strategy
        self.sma_50 = []
        self.sma_200 = []
        self.prev_sma_diff = 0
        self.sma_intersections = []
        self.sma_list = []
        self.prev_ema_26 = 0
        self.prev_ema_50 = 0
        self.prev_ema_200 = 0
        self.macd = []
        self.macd_flag = False
        self.start_time = self.event_loop.time()

        # Most recent resistance line value
        self.resist = 0

        # Most recent support line value
        self.support = 0

        # Gradient of recent trend
        self.slope = 0

        # R2 Coefficient
        self.r2 = 0

        #DELETE
        self.pnl = []

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
                
                # sma_50 = self.calculate_sma(50)
                # sma_200 = self.calculate_sma(200)
                # sma_diff = sma_50 - sma_200
                # self.sma_50.append(sma_50)
                # self.sma_200.append(sma_200)
                # inter = False
                # if self.prev_sma_diff < 0:
                #     if sma_diff > 0:
                #         self.sma_intersections.append(self.event_loop.time())
                #         inter = True
                # elif self.prev_sma_diff > 0:
                #     if sma_diff < 0:
                #         self.sma_intersections.append(self.event_loop.time())
                #         inter = True
                # self.prev_sma_diff = sma_diff

                ema_26 = self.calculate_ema(26,self.prev_ema_26)
                ema_50 = self.calculate_ema(50,self.prev_ema_50)
                ema_200 = self.calculate_ema(200,self.prev_ema_200)
                macd = ema_200 - ema_50
                self.macd.append(macd)
                self.prev_ema_26 = ema_26
                self.prev_ema_50 = ema_50
                self.prev_ema_200 = ema_200
                if self.event_loop.time() - 200 > self.start_time:
                    if abs(self.macd[-1]) >= 90 and not self.macd_flag:
                        self.macd_flag = True
                    elif abs(self.macd[-1]) <= 10 and self.macd_flag:
                        self.macd_flag = False
                    #self.sma_list.append([self.event_loop.time(),self.etf_market_price[-1],sma_50,sma_200,inter,ema_26,ema_50,ema_200,macd])
                    #df = pd.DataFrame(self.sma_list,columns=['Time','Market','SMA-50','SMA-200','Intersection','EMA-26','EMA-50','EMA-200','MACD'])
                    #fig = df.plot(x="Time",y=["SMA-20","SMA-100"])
                    #print(self.sma_list)
                    #df.to_csv(path_or_buf="/home/posocer/Documents/projects/trader/readyTraderOne/example.csv")

                self.calculate_vwap(ask_prices, ask_volumes,
                                    bid_prices, bid_volumes)
                self.calculate_resist()
                self.calculate_support()
                self.calculate_regression()
                volume_limit = 0
                if ask_volumes[0] >= volume_limit:
                    self.ask = ask_prices[0]
                elif ask_volumes[1] >= volume_limit and ask_prices[1] - ask_prices[0] >= 200:
                    self.ask = ask_prices[1] - 100
                else:
                    self.ask = ask_prices[0] + 100
                if bid_volumes[0] >= volume_limit:
                    self.bid = bid_prices[0]
                elif bid_volumes[1] >= volume_limit and bid_prices[0] - bid_prices[1] >= 200:
                    self.bid = bid_prices[1] + 100
                else:
                    self.bid = bid_prices[0] - 100
            
                pnl = self.order_book.calc_profit_or_loss(self.future_market_price[-1], self.etf_market_price[-1])

                self.pnl.append([self.event_loop.time(), self.slope])
                df1 = pd.DataFrame(self.pnl, columns = ["Time", "Self PNL"])
                df1.to_csv("profit.csv")
                
                position = self.order_book.get_position()
                diff = self.etf_market_price[-1] - self.future_market_price[-1] 
                if self.macd_flag:
                    if self.macd[-1] < 0 and diff >= 0:
                        print("MACD")
                        print()
                        #self.logger.info("%f, MACD BUY Position: %d Volume: %d Bid Volume: %d Ask Volume: %d",self.event_loop.time()- self.start_time,self.order_book.position, self.order_book.volume, self.order_book.vol_bids, self.order_book.vol_asks)
                        self.send_buy_order(self.bid,LOT_SIZE,Lifespan.GOOD_FOR_DAY)
                    else:
                        print("MACD")
                        print()
                        #self.logger.info("%f, MACD SELL Position: %d Volume: %d Bid Volume: %d Ask Volume: %d",self.event_loop.time()-self.start_time,self.order_book.position, self.order_book.volume, self.order_book.vol_bids, self.order_book.vol_asks)
                        if diff <= 0:
                            self.send_sell_order(self.ask,LOT_SIZE,Lifespan.GOOD_FOR_DAY)
                else:
                    if pnl != 0 and (pnl/100 >= abs(position) * self.scale_factor or pnl/100 >= 200):
                        if position < 0:
                            print("Simple")
                            print()
                            #print("Price: ", self.bid)
                            #self.logger.info("%f, SIMPLE BUY Position: %d Volume: %d Bid Volume: %d Ask Volume: %d",self.event_loop.time()-self.start_time,self.order_book.position, self.order_book.volume, self.order_book.vol_bids, self.order_book.vol_asks)
                            self.send_buy_order(self.bid, min(abs(position), VOLUME_LIMIT), Lifespan.GOOD_FOR_DAY)
                        else: 
                            print("Simple")
                            print()
                            #print("Price: ", self.ask)
                            #self.logger.info("%f, SIMPLE SELL Position: %d Volume: %d Bid Volume: %d Ask Volume: %d",self.event_loop.time()-self.start_time,self.order_book.position, self.order_book.volume, self.order_book.vol_bids, self.order_book.vol_asks)
                            self.send_sell_order(self.ask, min(abs(position), VOLUME_LIMIT), Lifespan.GOOD_FOR_DAY)
                    else:
                        if self.r2 >= 0.1:
                            print("BUY: ", self.support, " <= ", self.bid, " <= ", self.support * (1 + self.bound_range))
                            print("SELL: ", self.resist * (1 - self.bound_range), " <= ", self.ask, " <= ", self.resist)

                            if self.support <= self.bid <= self.support * (1 + self.bound_range) and self.slope > 0.1 and diff >= 0:
                                print("Complex")
                                print()
                                #print("Price: ", self.bid)
                                #self.logger.info("%f, COMPLEX BUY Position: %d Volume: %d Bid Volume: %d Ask Volume: %d",self.event_loop.time()-self.start_time,self.order_book.position, self.order_book.volume, self.order_book.vol_bids, self.order_book.vol_asks)
                                self.send_buy_order(self.bid, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                                
                            elif self.resist * (1 - self.bound_range) <= self.ask <= self.resist and self.slope < -0.1 and diff <= 0:
                                print("Complex")
                                print()
                                #print("Price: ", self.ask)
                                #self.logger.info("%f, COMPLEX SELL Position: %d Volume: %d Bid Volume: %d Ask Volume: %d",self.event_loop.time()-self.start_time,self.order_book.position, self.order_book.volume, self.order_book.vol_bids, self.order_book.vol_asks)
                                self.send_sell_order(self.ask, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                        #print()
            #print(self.event_loop.time()-self.start_time,self.order_book.bids,self.order_book.asks)
            self.order_update_number[instrument] = sequence_number
            
    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when when of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        # currently we do not account for market orders as we do not do them
        #self.execute_order += 1
        self.logger.info("On order filled: Position before: %d Order ID: %d VOLUME: %d VOL_ASK: %d VOL_BID %d VOLUME_ORDERBOOK %d",self.order_book.position,client_order_id,volume,self.order_book.vol_asks, self.order_book.vol_bids, self.order_book.volume)
        self.order_book.amend_order(self,volume, client_order_id)
        self.logger.info("On order filled: Position after: %d Order ID: %d VOLUME: %d VOL_ASK: %d VOL_BID %d VOLUME_ORDERBOOK %d",self.order_book.position,client_order_id,volume,self.order_book.vol_asks, self.order_book.vol_bids, self.order_book.volume)

    # def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
    #                             fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:

        if sequence_number > self.trade_update_number[instrument]:
            self.calculate_market_price(
                instrument, ask_prices, bid_prices, ask_volumes, bid_volumes)
            self.trade_update_number[instrument] = sequence_number

    def send_buy_order(self, price: int, lot_size: int, order_type: Lifespan):
        if not self.rolling_period_limit():
            return
        
        id = next(self.order_ids)
        order = self.order_book.add_bid(price, lot_size, id)
        can = True
        # We could not enter an order here for two reasons. Either volume/position limit exceeded, OR too many orders.
        if not order[0]:
            if order[1] == 0 or order[1] == 1 or order[1] == 3: #need to change this to account for volume limits too
                if not self.rolling_period_limit():
                    return
                remove_id = self.order_book.remove_least_useful_order(self.etf_market_price[-1], price, Side.BID)
                if remove_id:
                    self.logger.info("Removed order: %d",remove_id)
                    self.send_cancel_order(remove_id)
                    order = self.order_book.add_bid(price, lot_size, id)
                    if not order[0]:
                        self.logger.info("Just straight up removed it")
                        can = False
                    else:
                        self.logger.info("We removed an order and then successfully added one into the orderbook so we should see an insert for order id: %d",id)
                else:
                    return
            # Volume exceed just dont do anything
            else:
                can = False

        if can:
            #self.logger.info("BUY Volume sent in is: %d Order ID is: %d Position: %d Volume: %d Bid Volume: %d Ask Volume: %d",order[1], id, self.order_book.position, self.order_book.volume, self.order_book.vol_bids, self.order_book.vol_asks)
            self.send_insert_order(id, Side.BUY, price, order[1], order_type)

    def send_sell_order(self, price: int, lot_size: int, order_type: Lifespan):
        if not self.rolling_period_limit():
            return

        id = next(self.order_ids)
        order = self.order_book.add_ask(price, lot_size, id)
        can = True
        # We could not enter an order here for two reasons. Either volume/position limit exceeded, OR too many orders.
        if not order[0]:
            # If too many orders we can remove one and see if it works
            if order[1] == 0 or order[1] == 1: #need to change this to account for volume limits too
                if not self.rolling_period_limit():
                    return
                remove_id = self.order_book.remove_least_useful_order(self.etf_market_price[-1],price,Side.ASK)
                if remove_id:
                    self.logger.info("Removed order: %d",remove_id)
                    self.send_cancel_order(remove_id)
                    order = self.order_book.add_ask(price, lot_size, id)
                    if not order[0]:
                        self.logger.info("Just straight up removed it")
                        can = False
                    else:
                        self.logger.info("We removed an order and then successfully added one into the orderbook so we should see an insert for order id: %d",id)
                else:
                    return
            # Volume exceed just dont do anything
            else:
                can = False

        if can:
            #self.logger.info("SELL Volume sent in is: %d Order ID is: %d Position: %d Volume: %d Bid Volume: %d Ask Volume: %d",order[1], id, self.order_book.position, self.order_book.volume, self.order_book.vol_bids, self.order_book.vol_asks)
            self.send_insert_order(id, Side.SELL, price, order[1], order_type)

    def calculate_market_price(self, instrument: int, ask_prices: List[int], bid_prices: List[int], ask_volumes: List[int], bid_volumes: List[int]) -> None:

        last_traded = self.get_last_traded_price(ask_prices, bid_prices)
        if last_traded: 
            if instrument == Instrument.FUTURE:
                self.future_market_price.append(last_traded)
            else:
                self.etf_market_price.append(last_traded)

    def get_last_traded_price(self, ask_prices: List[int], bid_prices: List[int]) -> Optional[int]:
        last_price = None
        if ask_prices[0] != 0 and bid_prices[0] != 0:
            last_price = round((ask_prices[0] + bid_prices[0]) / 2.0)
        else:
            # Find latest bid
            if ask_prices[0] != 0:
                last_price = ask_prices[0]
            elif bid_prices[0] != 0:
                last_price = bid_prices[0]

        return last_price

    def calculate_sma(self, period: int):
        # currently hardcoded for the ETF or something
        tail = self.etf_market_price[-period:]

        return mean(tail)

    def calculate_ema(self, period: int, prev_ema: float):
        multiplier = 2/(period +1)
        return self.etf_market_price[-1] * multiplier + prev_ema * (1-multiplier)

    def calculate_vwap(self, ask_prices: List[int], ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
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

        self.bid_vwap = round(
            bid_total_value / bid_total_volume) if (bid_total_volume != 0) else self.bid_vwap
        self.ask_vwap = round(
            ask_total_value / ask_total_volume) if (ask_total_volume != 0) else self.ask_vwap

    def calculate_resist(self) -> None:
        if len(self.etf_market_price) < RES_SUP_LENGTH:
            return

        prices = np.array(self.etf_market_price[-RES_SUP_LENGTH:])
        midpoint = (self.bid_vwap + self.ask_vwap) / 2
        peaks, _ = find_peaks(prices, height=midpoint)

        res: np.ndarray = prices[peaks]
        if res.size == 0:
            return

        self.resist = res.sum() / res.size

    def calculate_support(self) -> None:
        if len(self.etf_market_price) < RES_SUP_LENGTH:
            return

        prices = np.array(self.etf_market_price[-RES_SUP_LENGTH:])
        midpoint = (self.bid_vwap + self.ask_vwap) / 2
        reverse_prices = (prices - midpoint) * -1
        peaks, _ = find_peaks(reverse_prices, height=0)
        support: np.ndarray = prices[peaks]
        if support.size == 0:
            return

        self.support = support.sum() / support.size

    def calculate_regression(self) -> None:
        min_length = RES_SUP_LENGTH
        if len(self.etf_market_price) < min_length:
            return

        prices = np.array(self.etf_market_price[-min_length:])
        result = linregress(list(range(min_length)), prices)

        self.slope = result.slope
        self.r2 = result.rvalue**2

    def rolling_period_limit(self):
        # Orders for both buy and sell cannot exceed 50 in a 1 second rolling period
        time_limit = self.event_loop.time() - 1

        if self.last_time_called:
            while self.last_time_called[0] < time_limit:
                self.last_time_called.pop(0)
                if not self.last_time_called:
                    break
        if (len(self.last_time_called) == 50):
            return False
        else:
            self.last_time_called.append(self.event_loop.time())
        return True

class OrderBook():
    def __init__(self):
        # actual position
        self.position = 0

        # position if all orders in book are executed
        self.position_after_orders = 0

        # volume of current orders on the market
        self.volume = 0

        # number of orders on market
        self.num_orders = 0

        # bids and asks will be structured [price, vol, order_id]
        # bids and ask list will be sorted lowest to highest price
        self.asks = []
        self.bids = []

        self.vol_asks = 0
        self.vol_bids = 0

        self.average_price = 0
        self.total_buy_volume = 0
        self.total_sell_volume = 0
        
        self.future_position = 0
        self.last_buy = 0
        self.last_sell = 0

        self.cancelled_orders: Dict[int,Side] = {}

    def add_bid(self, price: int, vol: int, order_id: int):
        """
        vol is updated with a value which is legal to be put into the exchange

        Return: [Boolean, int] Boolean for whether it succeeded or not,if True int for volume OR reason for False if False

        Codes: 0 for ORDER_LIMIT, 1 for VOLUME_LIMIT, 2 for POSITION_LIMIT, 3 for BIDS + POSITION_LIMIT

        FUNCTION DOES NOT SEND AN INSERT ORDER TO EXCHANGE
        """
        
        if self.num_orders < ORDER_LIMIT:
            if self.volume == VOLUME_LIMIT:
                return [False, 1]
            if self.position == POSITION_LIMIT:
                return [False, 2]
            if self.position + self.vol_bids == POSITION_LIMIT:
                return [False, 3]
            if vol + self.volume > VOLUME_LIMIT:
                vol = VOLUME_LIMIT - self.volume
            if self.position + self.vol_bids + vol > POSITION_LIMIT:
                vol = POSITION_LIMIT - self.position - self.vol_bids
            insert = False
            for i in range(len(self.bids)):
                if(self.bids[i][0] >= price):
                    self.bids.insert(i, [price, vol, order_id])
                    insert = True
                    break
            if not insert:
                self.bids.append([price, vol, order_id])

            self.position_after_orders += vol
            self.volume += vol
            self.num_orders += 1
            self.vol_bids += vol
            return [True, vol]
        return [False, 0]

    def add_ask(self, price: int, vol: int, order_id: int):
        """
        vol is updated with a value which is legal to be put into the exchange

        Return: [Boolean, int] Boolean for whether it succeeded or not,if True int for volume OR reason for False if False

        Codes: 0 for ORDER_LIMIT, 1 for VOLUME_LIMIT, 2 for POSITION_LIMIT, 3 for ASKS + POSITION_LIMIT

        FUNCTION DOES NOT SEND AN INSERT ORDER TO EXCHANGE
        """
        #currently we are not implementing a hard 1000 position limit properly
        if self.num_orders < ORDER_LIMIT:
            if self.volume == VOLUME_LIMIT:
                return [False, 1]
            if self.position == -POSITION_LIMIT:
                return [False, 2]
            if -self.position + self.vol_asks == POSITION_LIMIT:
                return [False, 3]
            if vol + self.volume > VOLUME_LIMIT:
                vol = VOLUME_LIMIT - self.volume
            if -self.position + self.vol_asks + vol > POSITION_LIMIT:
                vol = POSITION_LIMIT - abs(self.position) - self.vol_asks
            insert = False
            for i in range(len(self.asks)):
                if(self.asks[i][0] >= price):
                    self.asks.insert(i, [price, vol, order_id])
                    insert = True
                    break
            if not insert:
                self.asks.append([price, vol, order_id])

            self.position_after_orders -= vol
            self.volume += vol
            self.num_orders += 1
            self.vol_asks += vol
            return [True, vol]
        return [False, 0]

    def amend_order(self, trader: AutoTrader, vol: int, order_id: int):
        """reduces the volume of the order as it has been partially filled/filled, if volume reaches 0 order will be removed from orders

        FUNCTION DOES NOT SEND A CANCEL ORDER TO EXCHANGE"""
        for i in range(len(self.bids)):
            bid = self.bids[i]
            if bid[2] == order_id:
                self.calc_average_price(bid[0], vol, Side.BUY)
                self.volume -= vol
                self.position += vol
                
                self.future_position -= vol
                self.last_buy = bid[0]

                self.vol_bids -= vol
                if bid[1]-vol == 0:
                    self.bids.pop(i)
                    self.num_orders -= 1
                else:
                    bid[1] -= vol
                return

        for i in range(len(self.asks)):
            ask = self.asks[i]
            if ask[2] == order_id:
                self.calc_average_price(ask[0], vol, Side.SELL)
                self.volume -= vol
                self.position -= vol

                self.future_position += vol
                self.last_sell = ask[0]

                self.vol_asks -= vol
                if ask[1]-vol == 0:
                    self.asks.pop(i)
                    self.num_orders -= 1
                else:
                    ask[1] -= vol
                return
        trader.logger.error("Order not found %d",order_id)

        if order_id in self.cancelled_orders:
            trader.logger.error("But then it got fixed")
            order = self.cancelled_orders.get(order_id)
            if order == Side.ASK:
                self.position -= vol
                self.future_position += vol
            else:
                self.position += vol
                self.future_position -= vol
            return

    # HAVENT ACCOUNTED FOR CASE WHERE WE REMOVE AN ORDER AND OUR SELF.POSITION_AFTER_ORDERS goes over -+ 1000

    def remove_order(self, order_id: int):
        for i in range(len(self.bids)):
            bid = self.bids[i]
            if bid[2] == order_id:
                self.num_orders -= 1
                self.position_after_orders -= bid[1]
                self.volume -= bid[1]
                self.vol_bids -= bid[1]
                self.bids.pop(i)
                self.cancelled_orders[order_id] = Side.BID
                return 


        for i in range(len(self.asks)):
            ask = self.asks[i]
            if ask[2] == order_id:
                self.num_orders -= 1
                self.position_after_orders += ask[1]
                self.volume -= ask[1]
                self.vol_asks -= ask[1]
                self.asks.pop(i)
                self.cancelled_orders[order_id] = Side.ASK
                return


    def remove_least_useful_order(self, market_price: int, price: int, side: Side):
        """removes the order on the side specified which is furthest away, in terms of price, from the current market price

        returns: order_id of order removed

        FUNCTION DOES NOT SEND A CANCEL ORDER TO THE EXCHANGE
        """
        max_diff = -1
        order_id = -1
        order = None
        remove_side = None
        
        # bids and asks will be structured [price, vol, order_id]

        for i in range(len(self.bids)):
            bid = self.bids[i]
            curr_diff = abs(market_price - bid[0])
            if curr_diff > max_diff or (curr_diff == max_diff and bid[2] > order_id):
                max_diff = curr_diff
                order_id  = bid[2]
                order = bid
                remove_side = Side.BID
        
        for i in range(len(self.asks)):
            ask = self.asks[i]
            curr_diff = abs(market_price - ask[0])
            if curr_diff > max_diff or (curr_diff == max_diff and ask[2] > order_id):
                max_diff = curr_diff
                order_id = ask[2]
                order = ask
                remove_side = Side.ASK

        if order[0] == price and side == remove_side:
            return None    

        self.remove_order(order_id)
        return order_id

    def calc_average_price(self, price: int, vol: int, side: Side):
        if side == Side.BUY:
            if self.position > 0:
                total_price = self.total_buy_volume * self.average_price + price * vol
                self.total_buy_volume += vol
                self.average_price = round(total_price / self.total_buy_volume)
            else:
                new_position = self.position + vol
                if new_position > 0:
                    self.total_sell_volume = 0
                    self.total_buy_volume = new_position
                    self.average_price = price
                elif new_position < 0:
                    self.total_sell_volume -= vol
                else:
                    self.total_buy_volume = 0
                    self.total_sell_volume = 0
                    self.average_price = 0
        else:
            if self.position < 0:
                total_price = self.total_sell_volume * self.average_price + price * vol
                self.total_sell_volume += vol
                self.average_price = round(
                    total_price / self.total_sell_volume)
            else:
                new_position = self.position - vol
                if new_position < 0:
                    self.total_buy_volume = 0
                    self.total_sell_volume = abs(new_position)
                    self.average_price = price
                elif new_position > 0:
                    self.total_buy_volume -= vol
                else:
                    self.total_buy_volume = 0
                    self.total_sell_volume = 0
                    self.average_price = 0

    def calc_profit_or_loss(self, future_price: int, etf_price: int):
        delta: int = round(0.02 * future_price)
        delta -= delta % 100
        min_price: int = future_price - delta
        max_price: int = future_price + delta
        clamped: int = min_price if etf_price < min_price else max_price if etf_price > max_price else etf_price
        profit_or_loss = self.future_position * future_price + self.position * clamped
        return profit_or_loss

    def get_average_price(self):
        return self.average_price

    def get_position(self):
        return self.position
