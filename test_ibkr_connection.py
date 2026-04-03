"""Quick test to verify IB Gateway connection on port 4002 (paper trading)."""

import time
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract


class IBKRTest(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.account_id = None
        self.next_order_id = None
        self.positions = []
        self.cash = None

    # ── Connection callbacks ────────────────────────────────────────────────
    def connectAck(self):
        print("✅ Connected to IB Gateway!")
        self.connected = True

    def nextValidId(self, orderId: int):
        self.next_order_id = orderId
        print(f"✅ Next valid order ID: {orderId}")
        # Kick off account data request
        self.reqAccountSummary(1, "All", "TotalCashValue,NetLiquidation,AvailableFunds")

    def error(self, reqId, errorTime=None, errorCode=None, errorString=None, advancedOrderRejectJson=""):
        # New ibapi 10.37+ signature adds errorTime as 2nd argument
        # Informational codes (market data farm status etc)
        if errorCode in (2104, 2106, 2158, 2107, 2103, 2119):
            print(f"[INFO {errorCode}]: {errorString}")
        else:
            print(f"[ERROR {errorCode}]: {errorString}")

    # ── Account callbacks ───────────────────────────────────────────────────
    def accountSummary(self, reqId, account, tag, value, currency):
        self.account_id = account
        print(f"📊 Account: {account} | {tag}: {value} {currency}")

    def accountSummaryEnd(self, reqId):
        print("─" * 50)
        print("✅ Account summary received. Connection fully working!")
        self.cancelAccountSummary(1)
        # Gracefully disconnect after getting data
        self.disconnect()


def test_connection(host="127.0.0.1", port=4002, client_id=99):
    print("=" * 50)
    print(f"  Testing IB Gateway connection")
    print(f"  Host: {host}  Port: {port}  ClientID: {client_id}")
    print("=" * 50)

    app = IBKRTest()
    app.connect(host, port, clientId=client_id)

    # Run in a background thread
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()

    # Wait up to 10 seconds for data
    timeout = 10
    for _ in range(timeout * 2):
        time.sleep(0.5)
        if not thread.is_alive():
            break

    if not app.connected:
        print()
        print("❌ Could NOT connect to IB Gateway.")
        print()
        print("Check the following in IB Gateway:")
        print("  1. Gateway is running and logged in")
        print("  2. Settings → API → Enable ActiveX and Socket Clients ✅")
        print("  3. Port is set to 4002")
        print("  4. 'Allow connections from localhost only' is checked")
        print("  5. Trusted IPs includes 127.0.0.1")
    else:
        print()
        print("🎉 IB Gateway is working correctly!")
        print(f"   Account ID : {app.account_id}")
        print(f"   Order ID   : {app.next_order_id}")


if __name__ == "__main__":
    test_connection()
