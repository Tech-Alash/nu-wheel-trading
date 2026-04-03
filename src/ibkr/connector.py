"""
IBKR Connector — base connection layer for IB Gateway (paper trading).

Account: DUP595019
Port   : 4002 (IB Gateway paper)
"""

import logging
import threading
import time
from typing import Optional

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 4002


class _IBKRApp(EWrapper, EClient):
    """Low-level IBKR app — handles all callbacks from the Gateway."""

    def __init__(self):
        EClient.__init__(self, self)

        # State
        self.next_order_id: Optional[int] = None
        self.account_id: Optional[str] = None
        self._ready = threading.Event()       # Set when nextValidId arrives
        self._lock = threading.Lock()

        # Data stores (filled by callbacks)
        self.account_values: dict = {}        # tag -> value
        self.positions: dict = {}             # ticker -> position dict
        self.market_data: dict = {}           # reqId -> latest tick
        self.option_chain: dict = {}          # reqId -> chain data
        self.historical_data: dict = {}       # reqId -> list of bars
        self.order_status: dict = {}          # orderId -> status

        # Request tracking
        self._req_id = 100
        self._pending: dict = {}              # reqId -> threading.Event

    # ── Helpers ────────────────────────────────────────────────────────────
    def next_req_id(self) -> int:
        with self._lock:
            self._req_id += 1
            return self._req_id

    def wait_for(self, req_id: int, timeout: float = 15.0) -> bool:
        event = threading.Event()
        self._pending[req_id] = event
        return event.wait(timeout)

    def _done(self, req_id: int):
        if req_id in self._pending:
            self._pending[req_id].set()

    # ── Connection callbacks ────────────────────────────────────────────────
    def connectAck(self):
        logger.info("Connected to IB Gateway")

    def nextValidId(self, orderId: int):
        self.next_order_id = orderId
        # Request delayed-frozen data (free on paper accounts)
        # 1=live(paid), 2=frozen, 3=delayed(free), 4=delayed-frozen(last known price)
        self.reqMarketDataType(4)
        self._ready.set()
        logger.info(f"Ready — next order ID: {orderId}")

    def error(self, reqId, errorTime=None, errorCode=None,
              errorString=None, advancedOrderRejectJson=""):
        # Ignore informational status messages
        if errorCode in (2104, 2106, 2107, 2103, 2158, 2119):
            return
        if errorCode == 326:   # Duplicate client ID (harmless on reconnect)
            return
        logger.warning(f"IBKR [{errorCode}] req={reqId}: {errorString}")
        # Unblock any waiting request
        if reqId in self._pending:
            self._done(reqId)

    # ── Account callbacks ───────────────────────────────────────────────────
    def accountSummary(self, reqId, account, tag, value, currency):
        self.account_id = account
        self.account_values[tag] = {"value": value, "currency": currency}
        logger.debug(f"Account {account} | {tag}: {value} {currency}")

    def accountSummaryEnd(self, reqId):
        self._done(reqId)

    # ── Position callbacks ──────────────────────────────────────────────────
    def position(self, account, contract, pos, avgCost):
        ticker = contract.symbol
        self.positions[ticker] = {
            "ticker": ticker,
            "sec_type": contract.secType,
            "exchange": contract.exchange,
            "currency": contract.currency,
            "position": pos,
            "avg_cost": avgCost,
        }

    def positionEnd(self):
        self._done(-1)     # positions use sentinel -1

    # ── Market data callbacks ───────────────────────────────────────────────
    def tickPrice(self, reqId, tickType, price, attrib):
        if reqId not in self.market_data:
            self.market_data[reqId] = {}
        # tickType: 1=bid, 2=ask, 4=last, 6=high, 7=low, 9=close
        tick_map = {1: "bid", 2: "ask", 4: "last", 6: "high", 7: "low", 9: "close"}
        if tickType in tick_map:
            self.market_data[reqId][tick_map[tickType]] = price

    def tickSize(self, reqId, tickType, size):
        if reqId not in self.market_data:
            self.market_data[reqId] = {}
        tick_map = {0: "bid_size", 3: "ask_size", 5: "last_size", 8: "volume"}
        if tickType in tick_map:
            self.market_data[reqId][tick_map[tickType]] = size

    def tickOptionComputation(self, reqId, tickType, tickAttrib,
                              impliedVol, delta, optPrice, pvDividend,
                              gamma, vega, theta, undPrice):
        if reqId not in self.market_data:
            self.market_data[reqId] = {}
        self.market_data[reqId].update({
            "iv": impliedVol,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "opt_price": optPrice,
            "und_price": undPrice,
        })

    def tickSnapshotEnd(self, reqId):
        self._done(reqId)

    # ── Historical data callbacks ───────────────────────────────────────────
    def historicalData(self, reqId, bar):
        if reqId not in self.historical_data:
            self.historical_data[reqId] = []
        self.historical_data[reqId].append({
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        })

    def historicalDataEnd(self, reqId, start, end):
        self._done(reqId)

    # ── Option chain callbacks ──────────────────────────────────────────────
    def securityDefinitionOptionalParameter(
        self, reqId, exchange, underlyingConId, tradingClass,
        multiplier, expirations, strikes
    ):
        if reqId not in self.option_chain:
            self.option_chain[reqId] = {"expirations": set(), "strikes": set()}
        self.option_chain[reqId]["expirations"].update(expirations)
        self.option_chain[reqId]["strikes"].update(strikes)

    def securityDefinitionOptionalParameterEnd(self, reqId):
        self._done(reqId)

    # ── Order callbacks ─────────────────────────────────────────────────────
    def orderStatus(self, orderId, status, filled, remaining,
                    avgFillPrice, permId, parentId, lastFillPrice,
                    clientId, whyHeld, mktCapPrice):
        self.order_status[orderId] = {
            "status": status,
            "filled": filled,
            "remaining": remaining,
            "avg_fill_price": avgFillPrice,
        }
        logger.info(f"Order {orderId}: {status} | filled={filled} @ {avgFillPrice}")
        if status in ("Filled", "Cancelled", "Inactive"):
            self._done(orderId)

    def openOrder(self, orderId, contract, order, orderState):
        logger.info(f"Open order {orderId}: {contract.symbol} {order.action} {order.totalQuantity}")

    def execDetails(self, reqId, contract, execution):
        logger.info(
            f"Execution: {contract.symbol} {execution.side} "
            f"{execution.shares} @ {execution.price}"
        )


class IBKRConnector:
    """
    High-level interface to IB Gateway.

    Usage:
        ibkr = IBKRConnector()
        ibkr.connect()
        summary = ibkr.get_account_summary()
        price   = ibkr.get_stock_price("AAPL")
        ibkr.disconnect()
    """

    def __init__(self, host: str = HOST, port: int = PORT, client_id: int = 1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self._app: Optional[_IBKRApp] = None
        self._thread: Optional[threading.Thread] = None

    # ── Connection management ───────────────────────────────────────────────
    def connect(self, timeout: float = 10.0) -> bool:
        self._app = _IBKRApp()
        self._app.connect(self.host, self.port, clientId=self.client_id)
        self._thread = threading.Thread(target=self._app.run, daemon=True)
        self._thread.start()
        ready = self._app._ready.wait(timeout)
        if ready:
            logger.info(f"IBKR connected — account: {self._app.account_id}")
        else:
            logger.error("IBKR connection timed out")
        return ready

    def disconnect(self):
        if self._app:
            self._app.disconnect()

    @property
    def app(self) -> _IBKRApp:
        if self._app is None:
            raise RuntimeError("Not connected — call connect() first")
        return self._app

    # ── Account ─────────────────────────────────────────────────────────────
    def get_account_summary(self) -> dict:
        """Return cash, net liquidation, available funds."""
        req_id = self.app.next_req_id()
        tags = "TotalCashValue,NetLiquidation,AvailableFunds,UnrealizedPnL,RealizedPnL"
        self.app.reqAccountSummary(req_id, "All", tags)
        self.app.wait_for(req_id, timeout=10)
        self.app.cancelAccountSummary(req_id)
        return {k: v["value"] for k, v in self.app.account_values.items()}

    def get_positions(self) -> dict:
        """Return all current positions."""
        self.app.positions.clear()
        self.app.reqPositions()
        self.app.wait_for(-1, timeout=10)
        self.app.cancelPositions()
        return self.app.positions

    # ── Market data ─────────────────────────────────────────────────────────
    def get_stock_price(self, ticker: str, wait: float = 4.0) -> Optional[dict]:
        """Get current bid/ask/last for a stock (delayed data, free on paper)."""
        import time
        contract = self._stock_contract(ticker)
        req_id = self.app.next_req_id()
        # snapshot=False → streaming; we cancel after wait seconds
        self.app.reqMktData(req_id, contract, "", False, False, [])
        time.sleep(wait)
        self.app.cancelMktData(req_id)
        data = self.app.market_data.get(req_id)
        return data if data else None

    def get_option_price(
        self, ticker: str, expiration: str, strike: float, right: str,
        wait: float = 5.0,
    ) -> Optional[dict]:
        """
        Get current market data for a specific option (delayed).

        Args:
            ticker    : e.g. "AAPL"
            expiration: e.g. "20260424"  (YYYYMMDD)
            strike    : e.g. 170.0
            right     : "P" for put, "C" for call
        """
        import time
        contract = self._option_contract(ticker, expiration, strike, right)
        req_id = self.app.next_req_id()
        self.app.reqMktData(req_id, contract, "100,101", False, False, [])
        time.sleep(wait)
        self.app.cancelMktData(req_id)
        return self.app.market_data.get(req_id)

    def get_option_chain_strikes(self, ticker: str) -> dict:
        """Return all available expirations and strikes for a ticker."""
        contract = self._stock_contract(ticker)
        req_id = self.app.next_req_id()
        self.app.reqSecDefOptParams(req_id, ticker, "", "STK", 0)
        self.app.wait_for(req_id, timeout=15)
        return self.app.option_chain.get(req_id, {})

    # ── Historical data ─────────────────────────────────────────────────────
    def get_stock_history(
        self,
        ticker: str,
        duration: str = "1 Y",
        bar_size: str = "1 day",
    ) -> list:
        """
        Fetch historical OHLCV bars.

        Args:
            duration: "1 Y", "6 M", "3 M", "1 W", etc.
            bar_size: "1 day", "1 hour", "5 mins", etc.
        """
        import datetime as dt
        contract = self._stock_contract(ticker)
        req_id = self.app.next_req_id()
        end_dt = dt.datetime.now().strftime("%Y%m%d %H:%M:%S")
        self.app.reqHistoricalData(
            req_id, contract, end_dt, duration,
            bar_size, "TRADES", 1, 1, False, []
        )
        self.app.wait_for(req_id, timeout=30)
        return self.app.historical_data.get(req_id, [])

    # ── Orders ───────────────────────────────────────────────────────────────
    def sell_put(
        self,
        ticker: str,
        expiration: str,
        strike: float,
        contracts: int,
        limit_price: float,
    ) -> int:
        """
        Sell a cash-secured put.

        Returns the order ID.
        """
        from ibapi.order import Order
        contract = self._option_contract(ticker, expiration, strike, "P")
        order = Order()
        order.action = "SELL"
        order.orderType = "LMT"
        order.totalQuantity = contracts
        order.lmtPrice = round(limit_price, 2)
        order.tif = "DAY"
        order.transmit = True

        order_id = self.app.next_order_id
        self.app.next_order_id += 1
        self.app.placeOrder(order_id, contract, order)
        logger.info(
            f"Placed SELL PUT: {ticker} {strike}P exp={expiration} "
            f"x{contracts} @ ${limit_price:.2f} (orderId={order_id})"
        )
        return order_id

    def sell_put_market(
        self,
        ticker: str,
        expiration: str,
        strike: float,
        contracts: int,
    ) -> int:
        """Sell a cash-secured put at MARKET price (paper trading)."""
        from ibapi.order import Order
        contract = self._option_contract(ticker, expiration, strike, "P")
        order = Order()
        order.action = "SELL"
        order.orderType = "MKT"
        order.totalQuantity = contracts
        order.tif = "DAY"
        order.transmit = True

        order_id = self.app.next_order_id
        self.app.next_order_id += 1
        self.app.placeOrder(order_id, contract, order)
        logger.info(
            f"Placed MKT SELL PUT: {ticker} {strike}P exp={expiration} "
            f"x{contracts} (orderId={order_id})"
        )
        return order_id

    def sell_call(
        self,
        ticker: str,
        expiration: str,
        strike: float,
        contracts: int,
        limit_price: float,
    ) -> int:
        """Sell a covered call."""
        from ibapi.order import Order
        contract = self._option_contract(ticker, expiration, strike, "C")
        order = Order()
        order.action = "SELL"
        order.orderType = "LMT"
        order.totalQuantity = contracts
        order.lmtPrice = round(limit_price, 2)
        order.tif = "DAY"
        order.transmit = True

        order_id = self.app.next_order_id
        self.app.next_order_id += 1
        self.app.placeOrder(order_id, contract, order)
        logger.info(
            f"Placed SELL CALL: {ticker} {strike}C exp={expiration} "
            f"x{contracts} @ ${limit_price:.2f} (orderId={order_id})"
        )
        return order_id

    def buy_to_close(
        self,
        ticker: str,
        expiration: str,
        strike: float,
        right: str,
        contracts: int,
        limit_price: float,
    ) -> int:
        """Buy back an option to close the position (take profit / stop loss)."""
        from ibapi.order import Order
        contract = self._option_contract(ticker, expiration, strike, right)
        order = Order()
        order.action = "BUY"
        order.orderType = "LMT"
        order.totalQuantity = contracts
        order.lmtPrice = round(limit_price, 2)
        order.tif = "DAY"
        order.transmit = True

        order_id = self.app.next_order_id
        self.app.next_order_id += 1
        self.app.placeOrder(order_id, contract, order)
        logger.info(
            f"Placed BUY TO CLOSE: {ticker} {strike}{right} exp={expiration} "
            f"x{contracts} @ ${limit_price:.2f} (orderId={order_id})"
        )
        return order_id

    def wait_for_fill(self, order_id: int, timeout: float = 30.0) -> dict:
        """Block until an order is filled or cancelled."""
        self.app.wait_for(order_id, timeout)
        return self.app.order_status.get(order_id, {})

    def cancel_order(self, order_id: int):
        self.app.cancelOrder(order_id, "")

    # ── Contract helpers ────────────────────────────────────────────────────
    @staticmethod
    def _stock_contract(ticker: str) -> Contract:
        c = Contract()
        c.symbol = ticker
        c.secType = "STK"
        c.exchange = "SMART"
        c.currency = "USD"
        return c

    @staticmethod
    def _option_contract(
        ticker: str, expiration: str, strike: float, right: str
    ) -> Contract:
        """
        Args:
            expiration: YYYYMMDD format e.g. "20260424"
            right     : "P" or "C"
        """
        c = Contract()
        c.symbol = ticker
        c.secType = "OPT"
        c.exchange = "SMART"
        c.currency = "USD"
        c.lastTradeDateOrContractMonth = expiration
        c.strike = strike
        c.right = right
        c.multiplier = "100"
        return c
