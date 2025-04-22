# main.py

import io
import os
import sys
import json
import contextlib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import import_module
from matcher import match_buy_order, match_sell_order
from datamodel import TradingState, Listing, OrderDepth, Trade, Observation, ConversionObservation

# Defaults; will be overridden by command-line args
ROUND_NUMBER = 0
SHOW_PLOT = True

PRODUCTS = [
    "RAINFOREST_RESIN", "KELP", "SQUID_INK", "CROISSANTS", "DJEMBES", "JAMS",
    "PICNIC_BASKET1", "PICNIC_BASKET2",
    "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250",
    "VOLCANIC_ROCK_VOUCHER_10500", "VOLCANIC_ROCK_VOUCHER_9500",
    "VOLCANIC_ROCK_VOUCHER_9750", "VOLCANIC_ROCK"
]

RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"
CROISSANTS = "CROISSANTS"
DJEMBES = "DJEMBES"
JAMS = "JAMS"
PCB1 = "PICNIC_BASKET1"
PCB2 = "PICNIC_BASKET2"
VOLCANIC_ROCK = "VOLCANIC_ROCK"
VOLCANIC_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
VOLCANIC_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
VOLCANIC_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
VOLCANIC_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
VOLCANIC_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9500"

CONSOLE_PRINT = False
POSITION_LIMITS = {
    RESIN: 50,
    KELP: 50,
    SQUID_INK: 50,
    CROISSANTS: 250,
    JAMS: 350,
    DJEMBES: 60,
    PCB1: 60,
    PCB2: 100,
    VOLCANIC_ROCK: 400,
    VOLCANIC_VOUCHER_9500: 200,
    VOLCANIC_VOUCHER_9750: 200,
    VOLCANIC_VOUCHER_10000: 200,
    VOLCANIC_VOUCHER_10250: 200,
    VOLCANIC_VOUCHER_10500: 200
}

def load_trading_states(log_path: str):
    """Load trading states from a JSON log file and convert each dictionary into a TradingState object."""
    with open(log_path, "r") as f:
        trading_states_data = json.load(f)
    
    def convert_trading_state(d):
        listings = {}
        for sym, data in d.get("listings", {}).items():
            listings[sym] = Listing(
                symbol=data["symbol"],
                product=data["product"],
                denomination=data["denomination"]
            )
        
        order_depths = {}
        for sym, data in d.get("order_depths", {}).items():
            od = OrderDepth()
            od.buy_orders = {int(k): int(v) for k, v in data.get("buy_orders", {}).items()}
            od.sell_orders = {int(k): int(v) for k, v in data.get("sell_orders", {}).items()}
            order_depths[sym] = od
        
        def convert_trades(trades):
            return [
                Trade(
                    symbol=t["symbol"],
                    price=int(t["price"]),
                    quantity=int(t["quantity"]),
                    buyer=t.get("buyer"),
                    seller=t.get("seller"),
                    timestamp=int(t["timestamp"])
                ) for t in trades
            ]
        market_trades = {sym: convert_trades(trades)
                         for sym, trades in d.get("market_trades", {}).items()}
        own_trades = {sym: convert_trades(trades)
                      for sym, trades in d.get("own_trades", {}).items()}
        
        position = {prod: int(val) for prod, val in d.get("position", {}).items()}
        
        obs = d.get("observations", {})
        plain_obs = {prod: int(val) for prod, val in obs.get("plainValueObservations", {}).items()}
        conv_obs_data = obs.get("conversionObservations", {})
        conv_obs = {}
        for prod, details in conv_obs_data.items():
            conv_obs[prod] = ConversionObservation(
                bidPrice=float(details.get("bidPrice", 0.0)),
                askPrice=float(details.get("askPrice", 0.0)),
                transportFees=float(details.get("transportFees", 0.0)),
                exportTariff=float(details.get("exportTariff", 0.0)),
                importTariff=float(details.get("importTariff", 0.0)),
                sugarPrice=float(details.get("sugarPrice", 0.0)),
                sunlightIndex=float(details.get("sunlightIndex", 0.0))
            )
        observations = Observation(
            plainValueObservations=plain_obs,
            conversionObservations=conv_obs
        )
        return TradingState(
            traderData=d.get("traderData", ""),
            timestamp=int(d.get("timestamp", 0)),
            listings=listings,
            order_depths=order_depths,
            own_trades=own_trades,
            market_trades=market_trades,
            position=position,
            observations=observations
        )
    return [convert_trading_state(d) for d in trading_states_data]

def parse_algorithm(algo_path: str):
    algorithm_path = Path(algo_path).expanduser().resolve()
    if not algorithm_path.is_file():
        raise ModuleNotFoundError(f"{algorithm_path} is not a file.")
    
    sys.path.append(str(algorithm_path.parent))
    return import_module(algorithm_path.stem)

def print_self_trade(trade):
    if trade.seller == "SUBMISSION":
        print(f"Sold {trade.quantity} {trade.symbol} at {trade.price}.")
    elif trade.buyer == "SUBMISSION":
        print(f"Bought {trade.quantity} {trade.symbol} at {trade.price}.")

def plot_pnl(per_product_pnl_over_time):
    """
    Plots the PnL over time for each product on the same axes.
    """
    if not per_product_pnl_over_time:
        print("No PnL data available to plot.")
        return

    plt.figure(figsize=(10, 6))
    for product, pnl_data in per_product_pnl_over_time.items():
        if pnl_data:
            timestamps, pnl_values = zip(*pnl_data)
            plt.plot(timestamps, pnl_values, marker="o", markersize=2, label=product)
    plt.xlabel("Timestamp")
    plt.ylabel("Profit and Loss")
    plt.title("PnL Over Time per Product")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    # uses globals ROUND_NUMBER and day_number
    out_dir = Path(f"results/round-{ROUND_NUMBER}/day-{day_number}")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "pnl_over_time.png")
    if SHOW_PLOT:
        plt.show()

def main(algo_path=None) -> None:
    if not algo_path:
        print("No algo path provided, using algorithms/algo.py")
        algo_path = "algorithms/algo.py"
    
    trader_module = parse_algorithm(algo_path)
    
    trader = trader_module.Trader()  # trader instance
    trader.cash = {prod: 0 for prod in PRODUCTS}
    trader.pnl = {prod: 0 for prod in PRODUCTS}
    trader.aggregate_cash = 0
    trader.aggregate_pnl = 0
    
    market_conditions = []
    trade_history_list = []
    sandbox_logs = []
    per_product_pnl = {prod: [] for prod in PRODUCTS}
    
    position = {prod: 0 for prod in PRODUCTS}
    traderData = ""

    for i, state in enumerate(trading_states):
        next_state = trading_states[i + 1] if i < len(trading_states) - 1 else None
        timestamp = state.timestamp
        traded = False
        all_trades_executed = []
        
        mid_prices = {}
        for product in state.listings:
            if (not state.order_depths[product].buy_orders
                or not state.order_depths[product].sell_orders):
                mid_prices[product] = -1
            else:
                mid_prices[product] = (
                    min(state.order_depths[product].sell_orders) +
                    max(state.order_depths[product].buy_orders)
                ) // 2
        
        if LOG_LENGTH and timestamp > LOG_LENGTH * 100:
            break
        
        state.position = position
        state.traderData = traderData
        
        if CONSOLE_PRINT:
            result, conversions, traderData = trader.run(state)
            lambda_log = ""
        else:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result, conversions, traderData = trader.run(state)
            lambda_log = buf.getvalue()
        
        sandbox_logs.append({
            "sandboxLog": "",
            "lambdaLog": lambda_log,
            "timestamp": timestamp
        })
        
        for product, orders_list in result.items():
            current_position = position.get(product, 0)
            total_buy = sum(o.quantity for o in orders_list if o.quantity > 0)
            total_sell = sum(-o.quantity for o in orders_list if o.quantity < 0)
            pos_limit = POSITION_LIMITS.get(product, 0)

            if current_position + total_buy > pos_limit or current_position - total_sell < -pos_limit:
                if VERBOSE:
                    print(f"[{timestamp}] Position limit exceeded for {product}.")
                continue

            for order in orders_list:
                if order.quantity > 0:
                    trades = match_buy_order(state, next_state, order)
                    filled = sum(t.quantity for t in trades)
                    position[product] += filled
                    change = -sum(t.price * t.quantity for t in trades)
                else:
                    trades = match_sell_order(state, next_state, order)
                    filled = sum(t.quantity for t in trades)
                    position[product] -= filled
                    change = sum(t.price * t.quantity for t in trades)

                trader.cash[product] += change
                trader.aggregate_cash += change

                for t in trades:
                    trade_history_list.append({
                        "timestamp": t.timestamp,
                        "buyer": t.buyer,
                        "seller": t.seller,
                        "symbol": t.symbol,
                        "currency": "SEASHELLS",
                        "price": t.price,
                        "quantity": t.quantity
                    })
                if trades:
                    all_trades_executed.extend(trades)
                    traded = True
        
        trader.pnl = trader.cash.copy()
        trader.aggregate_pnl = trader.aggregate_cash
        for prod, pos in position.items():
            trader.pnl[prod] += pos * mid_prices.get(prod, 0)
            trader.aggregate_pnl += pos * mid_prices.get(prod, 0)
        
        for prod in PRODUCTS:
            per_product_pnl[prod].append((timestamp, trader.pnl.get(prod, 0)))

        if traded and LOG_LENGTH and timestamp < LOG_LENGTH * 100:
            print(f"[{timestamp}]")
            for t in all_trades_executed:
                print_self_trade(t)
            print(f"Positions: {position}")
            print(f"Cash: {trader.aggregate_cash}")
            print(f"PNL: {trader.aggregate_pnl}\n")
        
        for prod in PRODUCTS:
            od = state.order_depths.get(prod, None)
            if od:
                bids = sorted(od.buy_orders.items(), key=lambda x: x[0], reverse=True)
                asks = sorted(od.sell_orders.items(), key=lambda x: x[0])
            else:
                bids = asks = []
            def pick(xs, i):
                return xs[i] if len(xs) > i else ("", "")
            bid1, bidvol1 = pick(bids, 0)
            bid2, bidvol2 = pick(bids, 1)
            bid3, bidvol3 = pick(bids, 2)
            ask1, askvol1 = pick(asks, 0)
            ask2, askvol2 = pick(asks, 1)
            ask3, askvol3 = pick(asks, 2)
            mid = (bid1 + ask1) / 2 if bids and asks else ""
            market_conditions.append({
                "day": day_number,
                "timestamp": timestamp,
                "product": prod,
                "bid_price_1": bid1,
                "bid_volume_1": bidvol1,
                "bid_price_2": bid2,
                "bid_volume_2": bidvol2,
                "bid_price_3": bid3,
                "bid_volume_3": bidvol3,
                "ask_price_1": ask1,
                "ask_volume_1": askvol1,
                "ask_price_2": ask2,
                "ask_volume_2": askvol2,
                "ask_price_3": ask3,
                "ask_volume_3": askvol3,
                "mid_price": mid,
                "profit_and_loss": trader.aggregate_pnl
            })
    
    out_base = Path(f"results/round-{ROUND_NUMBER}/day-{day_number}")
    out_base.mkdir(parents=True, exist_ok=True)

    # Export CSVs
    mc_df = pd.DataFrame(market_conditions)
    mc_df.to_csv(out_base / "orderbook.csv", sep=";", index=False)
    th_df = pd.DataFrame(trade_history_list)
    if not th_df.empty:
        th_df = th_df[["timestamp","buyer","seller","symbol","currency","price","quantity"]]
    th_df.to_csv(out_base / "trade_history.csv", sep=";", index=False)

    print("-----------------------------------------------------------------------------------")
    print("TOTAL PNL:", trader.aggregate_pnl)
    for prod, pnl in trader.pnl.items():
        print(f"  {prod}: {pnl}")
    print("Exported orderbook.csv and trade_history.csv.")
    print("-----------------------------------------------------------------------------------")
    
    combined = out_base / "combined_results.log"
    with open(combined, "w") as f:
        f.write("Sandbox logs:\n")
        for log in sandbox_logs:
            f.write(json.dumps(log, indent=2) + "\n")
        f.write("\nActivities log:\n")
        header = ("day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;"
                  "bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;"
                  "ask_price_3;ask_volume_3;mid_price;profit_and_loss\n")
        f.write(header)
        for cond in market_conditions:
            line = ("{day};{timestamp};{product};"
                    "{bid_price_1};{bid_volume_1};{bid_price_2};{bid_volume_2};"
                    "{bid_price_3};{bid_volume_3};{ask_price_1};{ask_volume_1};"
                    "{ask_price_2};{ask_volume_2};{ask_price_3};{ask_volume_3};"
                    "{mid_price};{profit_and_loss}\n").format(**cond)
            f.write(line)
        f.write("\nTrade History:\n")
        f.write(json.dumps(trade_history_list, indent=2))
    
    (Path("grid_search_data") / "pnl.txt").write_text(str(trader.aggregate_pnl))

    plot_pnl(per_product_pnl)


if __name__ == "__main__":
    # Expected args:
    #   1. Round number (int 0–5, default 0)
    #   2. Day number (int ≥0, default 0)
    #   3. Algorithm path (default "algorithms/algo.py")
    #   4. Log length (int, default all)
    #   5. Verbose (true/false or 1/0/yes/no, default false)

    # Parse round number
    if len(sys.argv) > 1:
        try:
            round_number = int(sys.argv[1])
            if round_number < 0 or round_number > 5:
                raise ValueError("must be between 0 and 5.")
        except ValueError as e:
            print(f"Invalid round number: {sys.argv[1]}. {e}")
            sys.exit(1)
    else:
        round_number = 0

    # Parse day number
    if len(sys.argv) > 2:
        try:
            day_number = int(sys.argv[2])
            if day_number < 0:
                raise ValueError("must be ≥ 0.")
        except ValueError as e:
            print(f"Invalid day number: {sys.argv[2]}. {e}")
            sys.exit(1)
    else:
        day_number = 0

    # Override globals
    ROUND_NUMBER = round_number

    # Parse algorithm path
    if len(sys.argv) > 3:
        algo_path = sys.argv[3]
        algo_file = Path(algo_path).expanduser().resolve()
        if not algo_file.is_file():
            print(f"Algorithm file not found: {algo_path}")
            sys.exit(1)
        algo_path = str(algo_file)
    else:
        algo_path = "algorithms/algo.py"
        default_algo = Path(algo_path).expanduser().resolve()
        if not default_algo.is_file():
            print(f"Default algorithm file not found: {algo_path}")
            sys.exit(1)

    # Parse log length
    if len(sys.argv) > 4:
        try:
            LOG_LENGTH = int(sys.argv[4])
            if LOG_LENGTH <= 0:
                raise ValueError("must be positive.")
        except ValueError as e:
            print(f"Invalid log length: {sys.argv[4]}. {e}")
            sys.exit(1)
    else:
        LOG_LENGTH = None

    # Parse verbose flag
    if len(sys.argv) > 5:
        v = sys.argv[5].lower()
        if v in ("true","1","yes","是"):
            VERBOSE = True
        elif v in ("false","0","no","否"):
            VERBOSE = False
        else:
            print(f"Invalid verbose flag: {sys.argv[5]}. Use true/false, 1/0, yes/no.")
            sys.exit(1)
    else:
        VERBOSE = False

    # Load trading states
    trading_states_file = Path(f"data/round-{ROUND_NUMBER}/day-{day_number}/trading_states.json")
    if not trading_states_file.is_file():
        print(f"Trading states file not found: {trading_states_file}")
        sys.exit(1)

    trading_states = load_trading_states(str(trading_states_file))
    main(algo_path)
